# import argparse
# import logging
# import math
# import os
# import sys
# import torch
# import wandb
# import json
# import numpy as np
# import pandas as pd
# import sys
# import tifffile
# import cv2
# from collections import Counter, defaultdict
# import openslide
# import zarr
# from fvcore.common.checkpoint import PeriodicCheckpointer
# from pathlib import Path
# from PIL import Image
# from pathlib import Path
# import gc
# import torch
# gc.collect()
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()
# import matplotlib
# matplotlib.use("Agg")  # headless -- no display needed, we only save into wandb.Image
# import matplotlib.pyplot as plt
# from functools import partial
# sys.path.append(".")
# import dinov2.distributed as distributed
# from dinov2.data import (DataAugmentationDINO,DataAugmentationHEMA,MaskingGenerator,SamplerType,collate_data_and_cast,make_data_loader,make_dataset,)
# import random
# from dinov2.fsdp import FSDPCheckpointer
# from dinov2.logging import MetricLogger
# from dinov2.train.ssl_meta_arch import SSLMetaArch
# from dinov2.utils.config import setup
# from dinov2.utils.utils import CosineScheduler, smooth_rank_measure
# from torch.utils.data import Dataset
# from datasets_w_masks import CONICCellDataset, MonkeyCellDataset, ImmunoctoHEDataset, PanNukeRawFoldDataset
# torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
# logger = logging.getLogger("dinov2")
# local_rank = int(os.environ["LOCAL_RANK"])
# torch.cuda.set_device(local_rank)
# device = torch.device(f"cuda:{local_rank}")

# def get_args_parser(add_help: bool = True):
#     parser = argparse.ArgumentParser("DINOv2 training", add_help=add_help)
#     parser.add_argument(
#         "--config-file", default="./dinov2/configs/train/custom.yaml", metavar="FILE", help="path to config file"
#     )
#     parser.add_argument(
#         "--no-resume",
#         action="store_true",
#         help="Whether to not attempt to resume from the checkpoint directory. ",
#     )
#     parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
#     parser.add_argument("--eval", type=str, default="", help="Eval type to perform")
#     parser.add_argument(
#         "opts",
#         help="""
#         """.strip(),
#         default=None,
#         nargs=argparse.REMAINDER,
#     )
#     parser.add_argument(
#         "--output-dir",
#         "--output_dir",
#         default="",
#         type=str,
#         help="Output directory to save logs and checkpoints",
#     )
#     parser.add_argument("--name", type=str, default="debug", help="Name of the run for logging")
#     parser.add_argument("--local-rank", type=int, help="Variable for distributed computing.")

#     return parser


# def build_optimizer(cfg, params_groups):
#     return torch.optim.AdamW(params_groups, betas=(cfg.optim.adamw_beta1, cfg.optim.adamw_beta2))


# def build_schedulers(cfg):
#     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
#     lr = dict(
#         base_value=cfg.optim["lr"],
#         final_value=cfg.optim["min_lr"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#         warmup_iters=int(cfg.optim["warmup_epochs"] * OFFICIAL_EPOCH_LENGTH),
#         start_warmup_value=0,
#     )
#     wd = dict(
#         base_value=cfg.optim["weight_decay"],
#         final_value=cfg.optim["weight_decay_end"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#     )
#     momentum = dict(
#         base_value=cfg.teacher["momentum_teacher"],
#         final_value=cfg.teacher["final_momentum_teacher"],
#         total_iters=cfg.optim["epochs"] * OFFICIAL_EPOCH_LENGTH,
#     )
#     teacher_temp = dict(
#         base_value=cfg.teacher["teacher_temp"],
#         final_value=cfg.teacher["teacher_temp"],
#         total_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
#         warmup_iters=cfg.teacher["warmup_teacher_temp_epochs"] * OFFICIAL_EPOCH_LENGTH,
#         start_warmup_value=cfg.teacher["warmup_teacher_temp"],
#     )

#     lr_schedule = CosineScheduler(**lr)
#     wd_schedule = CosineScheduler(**wd)
#     momentum_schedule = CosineScheduler(**momentum)
#     teacher_temp_schedule = CosineScheduler(**teacher_temp)
#     last_layer_lr_schedule = CosineScheduler(**lr)

#     last_layer_lr_schedule.schedule[
#         : cfg.optim["freeze_last_layer_epochs"] * OFFICIAL_EPOCH_LENGTH
#     ] = 0  # mimicking the original schedules

#     logger.info("Schedulers ready.")

#     return (
#         lr_schedule,
#         wd_schedule,
#         momentum_schedule,
#         teacher_temp_schedule,
#         last_layer_lr_schedule,
#     )


# def apply_optim_scheduler(optimizer, lr, wd, last_layer_lr):
#     for param_group in optimizer.param_groups:
#         is_last_layer = param_group["is_last_layer"]
#         lr_multiplier = param_group["lr_multiplier"]
#         wd_multiplier = param_group["wd_multiplier"]
#         param_group["weight_decay"] = wd * wd_multiplier
#         param_group["lr"] = (last_layer_lr if is_last_layer else lr) * lr_multiplier


# def log_morphology_visualization(model, data, wandb_key="morphology_visualization", n_samples=4):
#     if not distributed.is_main_process():
#         return {}

#     get_vis = getattr(model, "get_morphology_visualization", None)
#     if get_vis is None:
#         logger.warning(
#             "model has no get_morphology_visualization method (FSDP wrapping may be "
#             "hiding it) -- skipping morphology visualization this step"
#         )
#         return {}

#     vis = get_vis(data, n_samples=n_samples)
#     if vis is None:  # cfg.morphology.enabled is False
#         return {}
#     crops_np, raw_mask, morph_cov, attn_grid, attn_mask_corr, lam = vis
#     get_backbone_attn = getattr(model, "get_backbone_attention_visualization", None)
#     cls_attn_grid = get_backbone_attn(data, n_samples=n_samples) if get_backbone_attn is not None else None

#     images_for_wandb = []
#     for i in range(crops_np.shape[0]):
#         crop_img = Image.fromarray(crops_np[i])
#         proc_size = crop_img.size[0]

#         # struct-token attention, upsampled from the coarse patch grid
#         attn_i = attn_grid[i]
#         attn_i = (attn_i - attn_i.min()) / (attn_i.max() - attn_i.min() + 1e-8)
#         attn_up = np.array(
#             Image.fromarray((attn_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.BILINEAR)
#         ) / 255.0

#         # coarse patch-coverage channel -- literally what the model sees
#         # as morph_feat, blocky by construction (nearest, no smoothing)
#         cov_i = np.clip(morph_cov[i], 0, 1)
#         cov_up = np.array(
#             Image.fromarray((cov_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.NEAREST)
#         ) / 255.0

#         if cls_attn_grid is not None:
#             cls_i = cls_attn_grid[i]
#             cls_i = (cls_i - cls_i.min()) / (cls_i.max() - cls_i.min() + 1e-8)
#             cls_up = np.array(
#                 Image.fromarray((cls_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.BILINEAR)
#             ) / 255.0

#         n_panels = 3 + (1 if raw_mask is not None else 0) + (1 if cls_attn_grid is not None else 0)
#         fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))

#         axes[0].imshow(crop_img)
#         axes[0].set_title("crop", fontsize=9)
#         axes[0].axis("off")

#         panel = 1
#         if raw_mask is not None:
#             # full-resolution ground-truth mask -- the real check that
#             # image and mask geometry are actually aligned
#             mask_i = raw_mask[i]  # already (H, W) in {0,1}, same size as crop_img
#             axes[panel].imshow(crop_img)
#             axes[panel].imshow(mask_i, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
#             axes[panel].set_title("raw mask (full res)", fontsize=9)
#             axes[panel].axis("off")
#             panel += 1

#         axes[panel].imshow(crop_img)
#         axes[panel].imshow(cov_up, cmap="jet", alpha=0.45)
#         axes[panel].set_title("morph_feat coverage (coarse)", fontsize=9)
#         axes[panel].axis("off")
#         panel += 1

#         axes[panel].imshow(crop_img)
#         axes[panel].imshow(attn_up, cmap="jet", alpha=0.45)
#         axes[panel].set_title(f"struct-token attn (corr={attn_mask_corr[i]:+.2f})", fontsize=9)
#         axes[panel].axis("off")
#         panel += 1
#         if cls_attn_grid is not None:
#             axes[panel].imshow(crop_img)
#             axes[panel].imshow(cls_up, cmap="jet", alpha=0.45)
#             axes[panel].set_title("ViT CLS attention (backbone)", fontsize=9)
#             axes[panel].axis("off")

#         plt.tight_layout()
#         images_for_wandb.append(wandb.Image(fig))
#         plt.close(fig)

#     return {
#         wandb_key: images_for_wandb,
#         "structure_block_lambda": lam,
#         "struct_attn_mask_corr_mean": float(attn_mask_corr.mean()),
#     }


# def do_test(cfg, model, iteration):
#     new_state_dict = model.teacher.state_dict()

#     if distributed.is_main_process():
#         iterstring = str(iteration)
#         eval_dir = os.path.join(cfg.train.output_dir, "eval", iterstring)
#         os.makedirs(eval_dir, exist_ok=True)
#         # save teacher checkpoint
#         teacher_ckp_path = os.path.join(eval_dir, "teacher_checkpoint.pth")
#         torch.save({"teacher": new_state_dict}, teacher_ckp_path)


# def do_train(cfg, model, resume=False):
#     model.train()
#     inputs_dtype = torch.half
#     fp16_scaler = model.fp16_scaler  # for mixed precision training

#     # setup optimizer

#     optimizer = build_optimizer(cfg, model.get_params_groups())
#     (
#         lr_schedule,
#         wd_schedule,
#         momentum_schedule,
#         teacher_temp_schedule,
#         last_layer_lr_schedule,
#     ) = build_schedulers(cfg)

#     # checkpointer
#     checkpointer = FSDPCheckpointer(model, cfg.train.output_dir, optimizer=optimizer, save_to_disk=True)
#     if cfg.MODEL.WEIGHTS:
#         ckpt = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume)
#     else:
#         logger.info("No MODEL.WEIGHTS and no existing checkpoint in output_dir -- "
#                      "starting fresh from the pretrained backbone loaded in SSLMetaArch.")
#         ckpt = {}
#     # start_iter = ckpt.get("iteration", -1) + 1 if isinstance(ckpt, dict) else 37500
#     # start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
#     start_iter=0
#     OFFICIAL_EPOCH_LENGTH = cfg.train.OFFICIAL_EPOCH_LENGTH
#     max_iter = cfg.optim.epochs * OFFICIAL_EPOCH_LENGTH

#     periodic_checkpointer = PeriodicCheckpointer(
#         checkpointer,
#         period=3 * OFFICIAL_EPOCH_LENGTH,
#         max_iter=max_iter,
#         max_to_keep=3,
#     )
#     img_size = cfg.crops.global_crops_size
#     patch_size = cfg.student.patch_size
#     n_tokens = (img_size // patch_size) ** 2
#     mask_generator = MaskingGenerator(
#         input_size=(img_size // patch_size, img_size // patch_size),
#         max_num_patches=0.5 * img_size // patch_size * img_size // patch_size,
#     )
#     if cfg.morphology.enabled:
#         from dinov2.data.morph_augmentations import MorphAwareDataAugmentationDINO
#         from dinov2.data.collate import collate_morph_data_and_cast

#         data_transform = MorphAwareDataAugmentationDINO(
#             cfg.crops.global_crops_scale,
#             cfg.crops.local_crops_scale,
#             cfg.crops.local_crops_number,
#             global_crops_size=cfg.crops.global_crops_size,
#             local_crops_size=cfg.crops.local_crops_size,
#             patch_size=cfg.student.patch_size,
#         )
#         collate_fn = partial(
#             collate_morph_data_and_cast,
#             mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#             mask_probability=cfg.ibot.mask_sample_probability,
#             n_tokens=n_tokens,
#             mask_generator=mask_generator,
#             dtype=inputs_dtype,
#         )
#     elif cfg.data_transform == "default":
#         data_transform = DataAugmentationDINO(
#             cfg.crops.global_crops_scale,
#             cfg.crops.local_crops_scale,
#             cfg.crops.local_crops_number,
#             global_crops_size=cfg.crops.global_crops_size,
#             local_crops_size=cfg.crops.local_crops_size,
#         )
#         collate_fn = partial(
#             collate_data_and_cast,
#             mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#             mask_probability=cfg.ibot.mask_sample_probability,
#             n_tokens=n_tokens,
#             mask_generator=mask_generator,
#             dtype=inputs_dtype,
#         )
#     elif cfg.data_transform == "hema":
#         data_transform = DataAugmentationHEMA(
#             cfg.crops.global_crops_scale,
#             cfg.crops.local_crops_scale,
#             cfg.crops.local_crops_number,
#             global_crops_size=cfg.crops.global_crops_size,
#             local_crops_size=cfg.crops.local_crops_size,
#         )
#         collate_fn = partial(
#             collate_data_and_cast,
#             mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
#             mask_probability=cfg.ibot.mask_sample_probability,
#             n_tokens=n_tokens,
#             mask_generator=mask_generator,
#             dtype=inputs_dtype,
#         )
#     else:
#         raise ValueError(
#             f"Unrecognized cfg.data_transform: {cfg.data_transform!r} "
#             f"(and cfg.morphology.enabled is False) -- set cfg.data_transform to "
#             f"'default' or 'hema', or set cfg.morphology.enabled: true"
#         )

#     if not cfg.morphology.enabled:
#         raise ValueError(
#             "This training script only uses mask-bearing datasets (Immunocto, CONIC, "
#             "PanNuke, Monkey) and expects cfg.morphology.enabled: true. If you want a "
#             "non-morphology run, use the original dataset set and script instead."
#         )

#     # setup data loader -- ONLY mask-bearing datasets. PUMA, MIDOG, and the
#     # three TCGA Balanced_* datasets have no mask source and were removed
#     # rather than fed dummy zero masks.
#     dataset1 = ImmunoctoHEDataset(
#         root=cfg.train.dataset_path,
#         transform=data_transform,
#         return_masks=True,
#     )
#     dataset2 = CONICCellDataset(
#         transform=data_transform,
#         return_masks=True,
#     )
#     dataset3 = PanNukeRawFoldDataset(
#         root="/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke",
#         transform=data_transform,
#         return_masks=True,
#     )
#     # dataset6 = MonkeyCellDataset(
#     #     root="/tia/phd-data/u5552013/ImmunoDINO_datasets/Monkey",
#     #     transform=data_transform,
#     #     return_masks=True,
#     # )

#     dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])

#     # Print dataset sizes
#     print("\n=== Dataset sizes before training (mask-bearing datasets only) ===")
#     print(f"ImmunoctoHE: {len(dataset1)} samples")
#     print(f"CONIC:       {len(dataset2)} samples")
#     print(f"PanNuke:     {len(dataset3)} samples")
#     # print(f"MONKEY:      {len(dataset6)} samples")
#     total_samples = len(dataset1) + len(dataset2) + len(dataset3) 
#     print(f"TOTAL samples: {total_samples}\n")

#     # sampler_type = SamplerType.DISTRIBUTED
#     sampler_type = SamplerType.SHARDED_INFINITE
#     data_loader = make_data_loader(
#         dataset=dataset,
#         batch_size=cfg.train.batch_size_per_gpu,
#         num_workers=cfg.train.num_workers,
#         shuffle=True,
#         seed=start_iter,  # TODO: Fix this -- cfg.train.seed
#         sampler_type=sampler_type,
#         sampler_advance=0,  # TODO (qas): fix this -- start_iter * cfg.train.batch_size_per_gpu,
#         drop_last=True,
#         collate_fn=collate_fn,
#     )

#     # training loop

#     iteration = start_iter

#     logger.info("Starting training from iteration {}".format(start_iter))
#     metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
#     metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
#     header = "Training"

#     batch_collection = []
#     total_tokens_collected = 0

#     for data in metric_logger.log_every(
#         data_loader,
#         10,
#         header,
#         max_iter,
#         start_iter,
#     ):
#         current_batch_size = (data["collated_global_crops"].shape[0] / 2) * torch.cuda.device_count()
#         if iteration > max_iter:
#             return

#         # apply schedules

#         lr = lr_schedule[iteration]
#         wd = wd_schedule[iteration]
#         mom = momentum_schedule[iteration]
#         teacher_temp = teacher_temp_schedule[iteration]
#         last_layer_lr = last_layer_lr_schedule[iteration]
#         apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

#         # compute losses

#         optimizer.zero_grad(set_to_none=True)
#         loss_dict, class_tokens = model.forward_backward(data, teacher_temp=teacher_temp)

#         if fp16_scaler is not None:
#             if cfg.optim.clip_grad:
#                 fp16_scaler.unscale_(optimizer)
#                 for v in model.student.values():
#                     v.clip_grad_norm_(cfg.optim.clip_grad)
#             fp16_scaler.step(optimizer)
#             fp16_scaler.update()
#         else:
#             if cfg.optim.clip_grad:
#                 for v in model.student.values():
#                     v.clip_grad_norm_(cfg.optim.clip_grad)
#             optimizer.step()

#         # perform teacher EMA update

#         model.update_teacher(mom)

#         # logging

#         if distributed.get_global_size() > 1:
#             for v in loss_dict.values():
#                 torch.distributed.all_reduce(v)
#         loss_dict_reduced = {k: v.item() / distributed.get_global_size() for k, v in loss_dict.items()}

#         if math.isnan(sum(loss_dict_reduced.values())):
#             logger.info("NaN detected")
#             raise AssertionError
#         losses_reduced = sum(loss for loss in loss_dict_reduced.values())

#         metric_logger.update(lr=lr)
#         metric_logger.update(wd=wd)
#         metric_logger.update(mom=mom)
#         metric_logger.update(last_layer_lr=last_layer_lr)
#         metric_logger.update(current_batch_size=current_batch_size)
#         metric_logger.update(total_loss=losses_reduced, **loss_dict_reduced)

#         wandb_log_dict = {
#             "lr": lr,
#             "wd": wd,
#             "mom": mom,
#             "last_layer_lr": last_layer_lr,
#             "current_batch_size": current_batch_size,
#             "total_loss": losses_reduced,
#             **loss_dict_reduced,
#         }
#         vis_period = cfg.evaluation.get("wandb_vis_period_iterations", 250)
#         if cfg.morphology.enabled and vis_period > 0 and iteration % vis_period == 0:
#             wandb_log_dict.update(log_morphology_visualization(model, data))

#         wandb.log(wandb_log_dict)

#         desired_tokens = 2500
#         tokens_needed = desired_tokens - total_tokens_collected
#         batch_size = class_tokens.shape[0]  # Current batch size from class_tokens shape

#         if tokens_needed > 0 and iteration % 1000 < (int(desired_tokens / batch_size) + 1):
#             # If the whole batch can be added without exceeding 1000 tokens
#             if batch_size <= tokens_needed:
#                 batch_collection.append(class_tokens.detach())
#                 total_tokens_collected += batch_size
#             else:
#                 # Add only the required number of tokens from the batch
#                 batch_collection.append(class_tokens.detach()[:tokens_needed, :])
#                 total_tokens_collected += tokens_needed

#             print(f"{tokens_needed} tokens needed, {total_tokens_collected }tokens collected")

#         # Once desired_tokens are collected, process them
#         if total_tokens_collected == desired_tokens:
#             embedding_matrix = torch.cat(batch_collection, dim=0)
#             smooth_rank = smooth_rank_measure(embedding_matrix)  # Assuming this function is defined elsewhere
#             wandb.log({"smooth_rank": smooth_rank})
#             # Reset for the next tokens
#             batch_collection = []
#             total_tokens_collected = 0

#         if cfg.evaluation.eval_period_iterations > 0 and (iteration + 1) % cfg.evaluation.eval_period_iterations == 0:
#             do_test(cfg, model, f"training_{iteration}")
#             torch.cuda.synchronize()
#         periodic_checkpointer.step(iteration)

#         iteration = iteration + 1
#     metric_logger.synchronize_between_processes()
#     return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


# def main(args):
#     cfg = setup(args)

#     model = SSLMetaArch(cfg).to(device)

#     model.prepare_for_distributed_training()

#     logger.info("Model:\n{}".format(model))
#     if args.eval_only:
#         iteration = (
#             FSDPCheckpointer(model, save_dir=cfg.train.output_dir)
#             .resume_or_load(cfg.MODEL.WEIGHTS, resume=not args.no_resume)
#             .get("iteration", -1)
#             + 1
#         )
#         return do_test(cfg, model, f"manual_{iteration}")

#     do_train(cfg, model, resume=not args.no_resume)


# if __name__ == "__main__":
#     args = get_args_parser(add_help=True).parse_args()
#     # name = args.name if args.name != "immunocto" else args.name
#     name="dino_morph_mask"
#     args.output_dir = os.path.join(args.output_dir, name)
#     wandb.init(
#     entity="eshasadia2015-university-of-warwick",  # your own account
#     project="dinov2",
#     name=name,
#     mode="online",
#     config=args)

#     # wandb.init(entity="histo-collab", project="dinov2", name=name, mode="online", config=args)
#     main(args)

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.

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
import matplotlib
matplotlib.use("Agg")  # headless -- no display needed, we only save into wandb.Image
import matplotlib.pyplot as plt
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
from dinov2.train.ssl_meta_arch import SSLMetaArch
from dinov2.utils.config import setup
from dinov2.utils.utils import CosineScheduler, smooth_rank_measure
from torch.utils.data import Dataset
from datasets_w_masks import CONICCellDataset, MonkeyCellDataset, ImmunoctoHEDataset, PanNukeRawFoldDataset
torch.backends.cuda.matmul.allow_tf32 = True  # PyTorch 1.12 sets this to False by default
logger = logging.getLogger("dinov2")
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


def log_morphology_visualization(model, data, wandb_key="morphology_visualization", n_samples=4):
    """Renders a crop / raw-mask / patch-coverage / structure-attention
    panel for a few samples from the current batch and returns a dict
    ready to merge into the same wandb.log(...) call already made each
    iteration. Returns {} (nothing to log) if morphology is disabled or
    the model doesn't expose get_morphology_visualization directly.

    Only called on the main process -- avoid duplicate/racing wandb writes
    and duplicate matplotlib rendering work across ranks.
    """
    if not distributed.is_main_process():
        return {}

    get_vis = getattr(model, "get_morphology_visualization", None)
    if get_vis is None:
        logger.warning(
            "model has no get_morphology_visualization method (FSDP wrapping may be "
            "hiding it) -- skipping morphology visualization this step"
        )
        return {}

    vis = get_vis(data, n_samples=n_samples)
    if vis is None:  # cfg.morphology.enabled is False
        return {}
    crops_np, raw_mask, morph_cov, attn_grid, attn_mask_corr, lam = vis

    # Backbone's own CLS-token attention -- separate no-grad forward pass,
    # unrelated to structure_block. None if unavailable, in which case
    # this panel is silently skipped rather than failing the whole log.
    # DISABLED as of the CUDA illegal-memory-access crash on 2026-08-17 --
    # get_backbone_attention_visualization issues an extra out-of-band
    # forward call on a LIVE, actively-training FSDP-wrapped module while
    # (a) monkeypatching the module-level XFORMERS_AVAILABLE flag, which
    # changes which attention kernel path every layer takes mid-training,
    # and (b) persisting full dense (B, heads, N, N) attention tensors as
    # plain Python attributes on FSDP submodules via return_attention,
    # outliving the forward call they came from. Both are plausible
    # sources of memory corruption that can surface later as an
    # unrelated-looking illegal-memory-access in a subsequent flash
    # attention kernel call -- which is exactly what happened here after
    # ~33 minutes of otherwise-healthy training. Not worth the risk to the
    # real run for a diagnostic-only panel. Re-enable only after
    # redesigning this to run against a fully separate, non-FSDP-wrapped
    # snapshot of the backbone (e.g. load student.backbone's state_dict
    # into a plain unwrapped nn.Module copy periodically, and run the
    # attention-capture forward pass on THAT instead of the live model).
    cls_attn_grid = None

    images_for_wandb = []
    for i in range(crops_np.shape[0]):
        crop_img = Image.fromarray(crops_np[i])
        proc_size = crop_img.size[0]

        # struct-token attention, upsampled from the coarse patch grid
        attn_i = attn_grid[i]
        attn_i = (attn_i - attn_i.min()) / (attn_i.max() - attn_i.min() + 1e-8)
        attn_up = np.array(
            Image.fromarray((attn_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.BILINEAR)
        ) / 255.0
        cov_i = np.clip(morph_cov[i], 0, 1)
        cov_up = np.array(
            Image.fromarray((cov_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.NEAREST)
        ) / 255.0

        if cls_attn_grid is not None:
            cls_i = cls_attn_grid[i]
            cls_i = (cls_i - cls_i.min()) / (cls_i.max() - cls_i.min() + 1e-8)
            cls_up = np.array(
                Image.fromarray((cls_i * 255).astype(np.uint8)).resize((proc_size, proc_size), Image.BILINEAR)
            ) / 255.0

        n_panels = 3 + (1 if raw_mask is not None else 0) + (1 if cls_attn_grid is not None else 0)
        fig, axes = plt.subplots(1, n_panels, figsize=(3 * n_panels, 3))

        axes[0].imshow(crop_img)
        axes[0].set_title("crop", fontsize=9)
        axes[0].axis("off")

        panel = 1
        if raw_mask is not None:
            mask_i = raw_mask[i]  # already (H, W) in {0,1}, same size as crop_img
            axes[panel].imshow(crop_img)
            axes[panel].imshow(mask_i, cmap="Reds", alpha=0.45, vmin=0, vmax=1)
            axes[panel].set_title("raw mask (full res)", fontsize=9)
            axes[panel].axis("off")
            panel += 1

        axes[panel].imshow(crop_img)
        axes[panel].imshow(cov_up, cmap="jet", alpha=0.45)
        axes[panel].set_title("morph_feat coverage (coarse)", fontsize=9)
        axes[panel].axis("off")
        panel += 1

        axes[panel].imshow(crop_img)
        axes[panel].imshow(attn_up, cmap="jet", alpha=0.45)
        axes[panel].set_title(f"struct-token attn (corr={attn_mask_corr[i]:+.2f})", fontsize=9)
        axes[panel].axis("off")
        panel += 1

        if cls_attn_grid is not None:
            axes[panel].imshow(crop_img)
            axes[panel].imshow(cls_up, cmap="jet", alpha=0.45)
            axes[panel].set_title("ViT CLS attention (backbone)", fontsize=9)
            axes[panel].axis("off")

        plt.tight_layout()
        images_for_wandb.append(wandb.Image(fig))
        plt.close(fig)

    return {
        wandb_key: images_for_wandb,
        "structure_block_lambda": lam,
        "struct_attn_mask_corr_mean": float(attn_mask_corr.mean()),
    }


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

    # setup optimizer

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
    # start_iter = ckpt.get("iteration", -1) + 1 if isinstance(ckpt, dict) else 37500
    # start_iter = checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    start_iter=0
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

    # ---- data_transform + collate_fn: exactly one branch fires, each sets
    # BOTH data_transform and collate_fn together, so there's no way for a
    # stray unconditional block to clobber collate_fn after the fact -------
    if cfg.morphology.enabled:
        from dinov2.data.morph_augmentations import MorphAwareDataAugmentationDINO
        from dinov2.data.collate import collate_morph_data_and_cast

        data_transform = MorphAwareDataAugmentationDINO(
            cfg.crops.global_crops_scale,
            cfg.crops.local_crops_scale,
            cfg.crops.local_crops_number,
            global_crops_size=cfg.crops.global_crops_size,
            local_crops_size=cfg.crops.local_crops_size,
            patch_size=cfg.student.patch_size,
        )
        collate_fn = partial(
            collate_morph_data_and_cast,
            mask_ratio_tuple=cfg.ibot.mask_ratio_min_max,
            mask_probability=cfg.ibot.mask_sample_probability,
            n_tokens=n_tokens,
            mask_generator=mask_generator,
            dtype=inputs_dtype,
        )
    elif cfg.data_transform == "default":
        data_transform = DataAugmentationDINO(
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
    else:
        raise ValueError(
            f"Unrecognized cfg.data_transform: {cfg.data_transform!r} "
            f"(and cfg.morphology.enabled is False) -- set cfg.data_transform to "
            f"'default' or 'hema', or set cfg.morphology.enabled: true"
        )

    if not cfg.morphology.enabled:
        raise ValueError(
            "This training script only uses mask-bearing datasets (Immunocto, CONIC, "
            "PanNuke, Monkey) and expects cfg.morphology.enabled: true. If you want a "
            "non-morphology run, use the original dataset set and script instead."
        )

    # setup data loader -- ONLY mask-bearing datasets. PUMA, MIDOG, and the
    # three TCGA Balanced_* datasets have no mask source and were removed
    # rather than fed dummy zero masks.
    dataset1 = ImmunoctoHEDataset(
        root=cfg.train.dataset_path,
        transform=data_transform,
        return_masks=True,
    )
    dataset2 = CONICCellDataset(
        transform=data_transform,
        return_masks=True,
    )
    dataset3 = PanNukeRawFoldDataset(
        root="/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke",
        transform=data_transform,
        return_masks=True,
    )
    # dataset6 = MonkeyCellDataset(
    #     root="/tia/phd-data/u5552013/ImmunoDINO_datasets/Monkey",
    #     transform=data_transform,
    #     return_masks=True,
    # )

    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2, dataset3])

    # Print dataset sizes
    print("\n=== Dataset sizes before training (mask-bearing datasets only) ===")
    print(f"ImmunoctoHE: {len(dataset1)} samples")
    print(f"CONIC:       {len(dataset2)} samples")
    print(f"PanNuke:     {len(dataset3)} samples")
    # print(f"MONKEY:      {len(dataset6)} samples")
    total_samples = len(dataset1) + len(dataset2) + len(dataset3) 
    print(f"TOTAL samples: {total_samples}\n")

    # sampler_type = SamplerType.DISTRIBUTED
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

    # training loop

    iteration = start_iter

    logger.info("Starting training from iteration {}".format(start_iter))
    metrics_file = os.path.join(cfg.train.output_dir, "training_metrics.json")
    metric_logger = MetricLogger(delimiter="  ", output_file=metrics_file)
    header = "Training"

    batch_collection = []
    total_tokens_collected = 0

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

        # apply schedules

        lr = lr_schedule[iteration]
        wd = wd_schedule[iteration]
        mom = momentum_schedule[iteration]
        teacher_temp = teacher_temp_schedule[iteration]
        last_layer_lr = last_layer_lr_schedule[iteration]
        apply_optim_scheduler(optimizer, lr, wd, last_layer_lr)

        # compute losses

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

        # perform teacher EMA update

        model.update_teacher(mom)

        # logging

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

        wandb_log_dict = {
            "lr": lr,
            "wd": wd,
            "mom": mom,
            "last_layer_lr": last_layer_lr,
            "current_batch_size": current_batch_size,
            "total_loss": losses_reduced,
            **loss_dict_reduced,
        }

        # periodic crop / morphology-mask / structure-attention panel --
        # matplotlib rendering + an extra no-grad forward pass, so gated to
        # a coarse period rather than every iteration. Falls back to 250 if
        # cfg.evaluation.wandb_vis_period_iterations isn't set in the yaml.
        vis_period = cfg.evaluation.get("wandb_vis_period_iterations", 250)
        vis_period = 0
        if cfg.morphology.enabled and vis_period > 0 and iteration % vis_period == 0:
            wandb_log_dict.update(log_morphology_visualization(model, data))

        wandb.log(wandb_log_dict)

        # compute smooth rank measure

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

        # Once desired_tokens are collected, process them
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

    model = SSLMetaArch(cfg).to(device)

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
    name="dino_morph_maskonly_immunocto_conic_pannuke_monkey"
    args.output_dir = os.path.join(args.output_dir, name)
    wandb.init(
    entity="eshasadia2015-university-of-warwick",  # your own account
    project="dinov2",
    name=name,
    mode="online",
    config=args)

    # wandb.init(entity="histo-collab", project="dinov2", name=name, mode="online", config=args)
    main(args)