import argparse
import logging
import math
import os
import sys
import torch
import wandb
import json
import numpy as np
import pandas as pd
import sys
import tifffile
import cv2
from collections import Counter, defaultdict
import openslide
import zarr
from fvcore.common.checkpoint import PeriodicCheckpointer
from pathlib import Path
from PIL import Image
from pathlib import Path
import gc
import torch
gc.collect()
torch.cuda.empty_cache()
torch.cuda.ipc_collect()
from functools import partial
sys.path.append(".")
import dinov2.distributed as distributed
from dinov2.data import (
    DataAugmentationDINO,
    DataAugmentationHEMA,
    MaskingGenerator,
    SamplerType,
    collate_data_and_cast,
    make_data_loader,
    make_dataset,)
import random
from dinov2.fsdp import FSDPCheckpointer
from dinov2.logging import MetricLogger
from dinov2.train.ssl_meta_arch_v0 import SSLMetaArch
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler, smooth_rank_measure
from torch.utils.data import Dataset
from datasets import  PreExtractedCellDataset, MIDOGCellDataset,Balanced_TCGACellCropDataset, Balanced_TCGACellCropDataset1, Balanced_TCGACellCropDataset2, Balanced_TCGACellCropDataset3,ImmunoctoHEDataset, CONICCellDataset,PanNukeCellDataset
import math as _math
import cv2 as _cv2
import numpy as _np
import torch.nn.functional as _F
from PIL import Image as _Image
torch.backends.cuda.matmul.allow_tf32 = True 
logger = logging.getLogger("dinov2")

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)

import functools
_orig_torch_load = torch.load
torch.load = functools.partial(_orig_torch_load, weights_only=False)
def _find_last_block(backbone):
    blocks = backbone.blocks
    last_block = None
    for chunk in blocks:
        sub_blocks = list(chunk) if hasattr(chunk, "__iter__") else [chunk]
        for b in sub_blocks:
            if b.__class__.__name__ == "Identity":
                continue
            if hasattr(b, "attn") and hasattr(b.attn, "qkv"):
                last_block = b
    if last_block is None:
        raise RuntimeError(
            "Attention visualization: could not find a transformer block "
            "with an `attn.qkv` submodule on this backbone."
        )
    return last_block


class _AttentionExtractor:
    """
    Context manager that captures the qkv projection output of a backbone's
    last transformer block and reconstructs standard multi-head
    scaled-dot-product attention weights from it.

        with _AttentionExtractor(backbone) as extractor:
            backbone(images, is_training=True)
        attn = extractor.get_attention()   # (B, num_heads, N, N)
    """

    def __init__(self, backbone):
        self.backbone = backbone
        self.block = _find_last_block(backbone)
        self.attn_module = self.block.attn
        self.num_heads = self.attn_module.num_heads
        self._qkv_output = None
        self._handle = None

    def _hook(self, module, inp, out):
        self._qkv_output = out.detach()

    def __enter__(self):
        self._qkv_output = None
        self._handle = self.attn_module.qkv.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_attention(self):
        if self._qkv_output is None:
            raise RuntimeError(
                "_AttentionExtractor.get_attention() called before a "
                "forward pass was run inside the `with` block."
            )
        B, N, three_c = self._qkv_output.shape
        C = three_c // 3
        head_dim = C // self.num_heads

        qkv = self._qkv_output.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, heads, N, head_dim)
        q, k = qkv[0], qkv[1]

        scale = head_dim ** -0.5
        attn = (q.float() @ k.float().transpose(-2, -1)) * scale
        attn = attn.softmax(dim=-1)
        return attn  # (B, num_heads, N, N)


def _get_cls_attention_maps(attn, img_size, num_register_tokens=0):
    """
    attn: (B, num_heads, N, N), N = 1 (CLS) + num_register_tokens + num_patches.
    Returns the CLS token's attention to every patch, reshaped to a square
    grid and upsampled (nearest) to (img_size, img_size).
    Output: numpy array (B, num_heads, img_size, img_size)
    """
    B, num_heads, N, _ = attn.shape
    num_patches = N - 1 - num_register_tokens
    grid = int(_math.sqrt(num_patches))
    if grid * grid != num_patches:
        raise ValueError(
            f"Token count {N} (minus 1 CLS token and {num_register_tokens} "
            f"register tokens) does not form a square grid of patches "
            f"({num_patches} patches). Check num_register_tokens."
        )
    cls_attn = attn[:, :, 0, 1 + num_register_tokens:]  # (B, heads, num_patches)
    cls_attn = cls_attn.reshape(B, num_heads, grid, grid)
    cls_attn = _F.interpolate(cls_attn, size=(img_size, img_size), mode="nearest")
    return cls_attn.cpu().numpy()


def _unnormalize_image(tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD):
    """(3, H, W) normalized image tensor -> (H, W, 3) uint8 numpy array."""
    mean_t = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor * std_t + mean_t).clamp(0, 1)
    img = (img.permute(1, 2, 0).cpu().numpy() * 255).astype(_np.uint8)
    return img


def _overlay_attention(image_rgb_uint8, attn_map, alpha=0.5, colormap=_cv2.COLORMAP_JET):
    """image_rgb_uint8: (H, W, 3) uint8. attn_map: (H, W) float, min-max normalized internally."""
    attn_norm = attn_map - attn_map.min()
    denom = attn_norm.max() + 1e-8
    attn_norm = (attn_norm / denom * 255).astype(_np.uint8)
    heatmap = _cv2.applyColorMap(attn_norm, colormap)
    heatmap = _cv2.cvtColor(heatmap, _cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(_np.uint8)
    return overlay


def _build_attention_grid(image_rgb_uint8, head_attn_maps, alpha=0.5, max_heads=8):
    """
    Returns a single PIL Image:
        [original | mean-of-heads overlay | head_0 overlay | head_1 overlay | ...]
    `max_heads` caps how many individual head panels are drawn.
    """
    panels = [_Image.fromarray(image_rgb_uint8)]

    mean_map = head_attn_maps.mean(axis=0)
    panels.append(_Image.fromarray(_overlay_attention(image_rgb_uint8, mean_map, alpha)))

    num_heads_to_show = min(head_attn_maps.shape[0], max_heads)
    for h in range(num_heads_to_show):
        panels.append(_Image.fromarray(_overlay_attention(image_rgb_uint8, head_attn_maps[h], alpha)))

    widths, heights = zip(*(p.size for p in panels))
    grid = _Image.new("RGB", (sum(widths), max(heights)))
    x_offset = 0
    for p in panels:
        grid.paste(p, (x_offset, 0))
        x_offset += p.width
    return grid


@torch.no_grad()
def log_attention_visualization(
    model,
    images,
    iteration,
    img_size,
    wandb_key="attention_maps",
    num_images=2,
    use_teacher=True,
    alpha=0.5,
    max_heads=8,
):
    """
    model: SSLMetaArch instance.
    images: normalized image tensor (B, 3, img_size, img_size) on the
        correct device, e.g. `data["collated_global_crops"].cuda()`.
    iteration: current training iteration -- used as the wandb step and in
        the image caption.
    use_teacher: visualize the (EMA, more stable) teacher backbone by
        default; set False to inspect the student instead.

    Runs a small, isolated forward pass (no_grad, eval mode) purely for
    visualization -- it does not affect gradients, or the teacher EMA
    update.
    """
    backbone = model.teacher.backbone if use_teacher else model.student.backbone
    was_training = backbone.training
    backbone.eval()

    sample = images[:num_images]

    try:
        with _AttentionExtractor(backbone) as extractor:
            backbone(sample, is_training=True)
        attn = extractor.get_attention()  # (B, heads, N, N)
    finally:
        if was_training:
            backbone.train()

    num_register_tokens = getattr(backbone, "num_register_tokens", 0)
    attn_maps = _get_cls_attention_maps(attn, img_size, num_register_tokens=num_register_tokens)

    logged = {}
    for i in range(sample.shape[0]):
        img_rgb = _unnormalize_image(sample[i])
        grid = _build_attention_grid(img_rgb, attn_maps[i], alpha=alpha, max_heads=max_heads)
        logged[f"{wandb_key}/sample_{i}"] = wandb.Image(grid, caption=f"iter {iteration}")

    wandb.log(logged, step=iteration)
    logger.info(f"Logged attention visualization for {sample.shape[0]} sample(s) at iteration {iteration}")

# ==========================================================================
# End attention visualization
# ==========================================================================
sys.path.append('/tia/phd-data/u5552013/ImmunoDINO_datasets/PathoCellbench/Pathology-Foundation-Model-Benchmark/src')
from benchmark.dataset.lmdb_dataset import LMDBDataset
import os
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")

def get_args_parser(add_help: bool = True):
    parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
    parser.add_argument(
        "--config-file", default="./dinov2/configs/train/custom.yaml", metavar="FILE", help="path to config file"
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Whether to not attempt to resume from the checkpoint directory. ",
    )
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
    parser.add_argument(
        "opts",
        help="""
Modify config options at the end of the command. For Yacs configs, use
space-separated "PATH.KEY VALUE" pairs.
For python-based LazyConfig, use "path.key=value".
        """.strip(),
        default=None,
        nargs=argparse.REMAINDER,
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        default="",
        type=str,
        help="Output directory to save logs and checkpoints",
    )
    parser.add_argument("--name", type=str, default="debug", help="Name of the run for logging")
    parser.add_argument("--local-rank", type=int, help="Variable for distributed computing.")

    return parser


def build_optimizer(cfg, params_groups):
    return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


def build_schedulers(cfg):
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    lr = dict(
        base_value=cfg.optim["lr"],
        final_value=cfg.optim["min_lr"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=int(cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH),
        start_warmup_value=0,
    )
    wd = dict(
        base_value=cfg.optim["weight_decay"],
        final_value=cfg.optim["weight_decay_end"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    momentum = dict(
        base_value=cfg.teacher["momentum_teacher"],
        final_value=cfg.teacher["final_momentum_teacher"],
        total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
    )
    teacher_temp = dict(
        base_value=cfg.teacher["teacher_temp"],
        final_value=cfg.teacher["teacher_temp"],
        total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
        start_warmup_value=cfg.teacher["warmup_teacher_temp"],
    )

    lr_schedule = CosineScheduler(**lr)
    wd_schedule = CosineScheduler(**wd)
    momentum_schedule = CosineScheduler(**momentum)
    teacher_temp_schedule = CosineScheduler(**teacher_temp)
    last_layer_lr_schedule = CosineScheduler(**lr)

    last_layer_lr_schedule.schedule[
        : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
    ] = 0  # mimicking the original schedules

    logger.info("Schedulers ready.")

    return (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    )


def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
    for param_group in optimizer.param_groups:
        is_last_layer = param_group["is_last_layer"]
        lr_multiplier = param_group["lr_multiplier"]
        wd_multiplier = param_group["wd_multiplier"]
        param_group["weight_decay"] = wd * wd_multiplier
        param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


def do_test(cfg, model, iteration):
    new_state_dict = model.teacher.state_dict()

    if distributed.is_main_process():
        iterstring = str(iteration)
        eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
        os.makedirs(eval_dir, exist_ok=True)
        # save teacher checkpoint
        teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
        torch.save({"teacher": new_state_dict}, teacher_ckp_path)


def do_train(cfg, model, resume=False):
    model.train()
    inputs_dtype = torch.half
    fp16_scaler = model.fp16_scaler  # for mixed precision training
    optimizer = build_optimizer(cfg, model.get_params_groups())
    (
        lr_schedule,
        wd_schedule,
        momentum_schedule,
        teacher_temp_schedule,
        last_layer_lr_schedule,
    ) = build_schedulers(cfg)

    # checkpointer
    checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
    if cfg.MODEL.WEIGHTS:
        ckpt = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
    else:
        logger.info("No MODEL.WEIGHTS and no existing checkpoint in output_dir -- "
                     "starting fresh from the pretrained backbone loaded in SSLMetaArch.")
        ckpt = {}
    start_iter = ckpt.get("iteration", -1) + 1 if isinstance(ckpt, dict) else 37500
    start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    # start_iter=0
    OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
    max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer,
        period=3 * OFFICIAL_EPOCH_LENGTH,
        max_iter=max_iter,
        max_to_keep=3,
    )

    # setup data preprocessing

    img_size = cfg.crops.global_crops_size
    patch_size = cfg.student.patch_size
    n_tokens = (img_size // patch_size) ** 2
    mask_generator = MaskingGenerator(
        input_size=(img_size // patch_size, img_size // patch_size),
        max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
    )
    if cfg.data_transform == "default":
        data_transform = DataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )
    elif cfg.data_transform == "hema":
        data_transform = DataAugmentationHEMA(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
        )

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
        mask_probability=cfg.ibot.mask_sample_probability,
        n_tokens=n_tokens,
        mask_generator=mask_generator,
        dtype=inputs_dtype,
    )
    dataset1 = ImmunoctoHEDataset(
            root=cfg.train.dataset_path,
            transform=data_transform,)
    dataset2 = CONICCellDataset(transform=data_transform)
    dataset3 = PanNukeCellDataset(transform=data_transform)
    dataset5 = PreExtractedCellDataset(
        root="/tia/phd-data/u5552013/ImmunoDINO_datasets/PUMA",
        transform=data_transform,
        dataset_name="PUMA"
    )
    dataset6 = PreExtractedCellDataset(
        root="/tia/phd-data/u5552013/ImmunoDINO_datasets/Monkey",
        transform=data_transform,
        dataset_name="Monkey"
    )
    dataset7 = MIDOGCellDataset(
    csv_file="/tia/lab-share/esha/MIDOG/Classification/Final/seven.csv",
    root_dir="/tia/lab-share/esha/MIDOG/Classification/Final",
    transform=data_transform,
    dataset_name="MIDOG")
    dataset8 = Balanced_TCGACellCropDataset(transform=data_transform)
    dataset9 = Balanced_TCGACellCropDataset1(transform=data_transform)
    dataset10 = Balanced_TCGACellCropDataset2(transform=data_transform)
    dataset11 = Balanced_TCGACellCropDataset3(transform=data_transform)
    dataset =  torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3, dataset5,dataset6, dataset7, dataset8,dataset9, dataset10, dataset11])
    # Print dataset sizes
    print("\n=== Dataset sizes before training ===")
    print(f"ImmunoctoHE: {len(dataset1)} samples")
    print(f"CONIC:       {len(dataset2)} samples")
    print(f"PanNuke:     {len(dataset3)} samples")
    print(f"PUMA:   {len(dataset5)} samples")
    print(f"MONKEY:   {len(dataset6)} samples")
    print(f"MIDOG:   {len(dataset7)} samples")
    print(f"TCGA:   {len(dataset8)} samples")
    print(f"TCGA1:   {len(dataset9)} samples")
    print(f"TCGA2:   {len(dataset10)} samples")
    print(f"TCGA3:   {len(dataset11)} samples")
    # + len(dataset4)
    total_samples = len(dataset1) + len(dataset2) + len(dataset3)  + len(dataset5) +len(dataset6) + len(dataset7)+len(dataset8)+len(dataset9)+len(dataset10)+len(dataset11)
    print(f"TOTAL samples: {total_samples}\n")
    sampler_type = SamplerType.SHARDED_INFINITE
    data_loader = make_data_loader(
        dataset=dataset,
        batch_size=cfg.train.batch_size_per_gpu,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        seed=start_iter,  # TODO: Fix this -- cfg.train.seed
        sampler_type=sampler_type,
        sampler_advance=0,  # TODO (qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
        drop_last=True,
        collate_fn=collate_fn,
    )
    iteration = start_iter
    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    batch_collection = []
    total_tokens_collected = 0
    attention_viz_period=0

    for data in metric_logger.log_every(
        data_loader,
        10,
        header,
        max_iter,
        start_iter,
    ):
        current_batch_size = (data["collated_global_crops"].shape[0] / 2) * torch.cuda.device_count()
        if iteration > max_iter:
            return
        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)
        optimizer.zero_grad(set_to_none=True)
        loss_dict, class_tokens = model.forward_backward(data, teacher_temp=teacher_temp)

        if fp16_scaler is not None:
            if cfg.optim.clip_grad:
                fp16_scaler.unscale_(optimizer)
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
        else:
            if cfg.optim.clip_grad:
                for v in model.student.values():
                    v.clip_grad_norm_(cfg.optim.clip_grad)
            optimizer.step()
        model.update_teacher(mom)
        if distributed.get_global_size() > 1:
            for v in loss_dict.values():
                torch.distributed.all_reduce(v)
        loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

        if math.isnan(sum(loss_dict_reduced.values())):
            logger.info("NaN detected")
            raise AssertionError
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        metric_logger.update(lr=lr)
        metric_logger.update(wd=wd)
        metric_logger.update(mom=mom)
        metric_logger.update(last_layer_lr=last_layer_lr)
        metric_logger.update(current_batch_size=current_batch_size)
        metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

        wandb.log(
            {
                "lr": lr,
                "wd": wd,
                "mom": mom,
                "last_layer_lr": last_layer_lr,
                "current_batch_size": current_batch_size,
                "total_loss": losses_reduced,
                **loss_dict_reduced,
            }
        )
        if (
            distributed.is_main_process()
            and attention_viz_period > 0
            and iteration % attention_viz_period == 0
        ):
            try:
                log_attention_visualization(
                    model,
                    data["collated_global_crops"].cuda(non_blocking=True),
                    iteration,
                    img_size=img_size,
                    num_images=attention_viz_num_images,
                )
            except Exception as e:
                logger.info(f"Attention visualization failed at iteration {iteration}: {e}")

        desired_tokens = 2500
        tokens_needed = desired_tokens - total_tokens_collected
        batch_size = class_tokens.shape[0]  # Current batch size from class_tokens shape
        if tokens_needed > 0 and iteration % 1000 < (int(desired_tokens / batch_size) + 1):
            # If the whole batch can be added without exceeding 1000 tokens
            if batch_size <= tokens_needed:
                batch_collection.append(class_tokens.detach())
                total_tokens_collected += batch_size
            else:
                # Add only the required number of tokens from the batch
                batch_collection.append(class_tokens.detach()[:tokens_needed, :])
                total_tokens_collected += tokens_needed

            print(f"{tokens_needed} tokens needed, {total_tokens_collected }tokens collected")
        if total_tokens_collected == desired_tokens:
            embedding_matrix = torch.cat(batch_collection, dim=0)
            smooth_rank = smooth_rank_measure(embedding_matrix)  # Assuming this function is defined elsewhere
            wandb.log({"smooth_rank": smooth_rank})
            # Reset for the next tokens
            batch_collection = []
            total_tokens_collected = 0

        if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
            do_test(cfg, model, f"training_{iteration}")
            torch.cuda.synchronize()
        periodic_checkpointer.step(iteration)

        iteration = iteration + 1
    metric_logger.synchronize_between_processes()
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args):
    cfg = setup(args)
    model = SSLMetaArch(cfg).to(torch.device("cuda"))
    model.prepare_for_distributed_training()
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        iteration = (
            FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
            .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
            .get("iteration", -1)
            + 1
        )
        return do_test(cfg, model, f"manual_{iteration}")

    do_train(cfg, model, resume=not args.no_resume)


if __name__ == "__main__":
    args = get_args_parser(add_help=True).parse_args()
    # name = args.name if args.name != "immunocto" else args.name
    name="dinov2_p4"
    args.output_dir = os.path.join(args.output_dir, name)
    wandb.init(
    entity="eshasadia2015-university-of-warwick",  # your own account
    project="dinov2",
    name=name,
    mode="online",
    config=args)

    # wandb.init(entity="histo-collab", project="dinov2", name=name, mode="online", config=args)
    main(args)