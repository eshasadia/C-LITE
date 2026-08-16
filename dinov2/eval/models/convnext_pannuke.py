#!/usr/bin/env python3
"""
run_pannuke_pipeline.py

Standalone, single-file, local-machine pipeline. Does everything the
notebook did across several cells, in one script:

  1. Downloads PanNuke (RationAI/PanNuke on Hugging Face -- public,
     non-gated) and extracts 64x64 cell-centered crops from its per-
     nucleus instance masks. Trains on fold1+fold2, holds fold3 out
     entirely for evaluation (no data leakage between the two).
  2. Trains an atrous-adapted DINOv3-ConvNeXt-Tiny backbone via DINO EMA
     self-distillation (Caron et al. 2021) + multi-stage self-distillation
     ("Be Your Own Teacher", Zhang et al. 2019), using the validated
     recipe from prior runs: backbone-freeze warmup + differential
     backbone/head learning rates (see WHAT THIS IS below for why).
  3. Runs a frozen linear-probe evaluation (logistic regression on pooled
     features) against the held-out fold3 crops.

This has NO Colab/notebook dependencies -- pure CLI. Each stage can be
skipped independently (e.g. --skip-extraction if crops already exist on
disk, --skip-training if you just want to re-probe an existing
checkpoint).

WHAT THIS IS, PLAINLY (condensed from the individual scripts this merges):
  - Architecture: DINOv3-ConvNeXt-Tiny's stem/stage-1 downsample stride is
    halved (pretrained weights untouched -- kernel size, not shape,
    changes), with TRUE atrous/dilated-convolution compensation (DeepLab,
    Chen et al. 2015-16) applied to the depthwise 7x7 block convs
    downstream. The patchify downsample convs themselves do NOT get true
    atrous treatment -- they have no overlap/padding structure to
    preserve in the first place; this is a pragmatic stride-halving, not
    an exact substitution.
  - Multi-stage self-distillation (BYOT) is a forced adaptation of Zhang
    et al.'s supervised method to this label-free SSL setting -- the
    deep DINO head's own output stands in for the true-label softmax the
    original paper uses. Not a direct transcription.
  - Backbone-freeze warmup + differential backbone LR exist because an
    earlier run without them produced full representational collapse the
    instant the backbone unfroze (loss pinned at ln(4096), the exact
    value for a uniform/constant output) -- randomly-initialized DINO/BYOT
    heads sent large, noisy gradients into the pretrained backbone with
    zero optimizer-state history behind it. This recipe fixed that; it is
    NOT a default DINO practice, it's a fix for a specific observed
    failure mode.

FIX APPLIED (this version): DataLoader workers now spawn via the `spawn`
multiprocessing start method instead of Linux's default `fork`. Without
this, the training loop hung indefinitely and silently on the first
`next(data_iter)` call -- forking a process that already has an
initialized CUDA context (from `model.to(device)`, which runs before the
DataLoader is ever iterated) is a well-known deadlock source: no error,
no progress, no output, forever. `spawn` avoids inheriting the parent's
CUDA state in worker processes entirely. This must be set before any
CUDA call happens, hence it's done at true module import time, guarded
by `if __name__ == "__main__"`.

STATUS: the merge/orchestration logic in this file (argument wiring,
extraction-then-train-then-probe sequencing, checkpoint auto-discovery)
was smoke-tested end-to-end with random-init weights and synthetic data
standing in for a real PanNuke download, since real network/gated-weight
access isn't available in the environment this was built in. Every
individual piece (atrous surgery, DINO/BYOT losses, freeze/unfreeze,
differential LR, extraction math) was independently verified against
real and synthetic data in prior, separate testing -- see the git history
/ prior conversation this was built from for those specifics if you have
it. Run --skip-training --skip-probe first to sanity check extraction
alone before committing to a full run.

Prerequisites:
    pip install -U transformers huggingface_hub datasets torch torchvision \\
        zarr pandas pillow numpy scikit-learn
    Accept the license at
    https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    and either `huggingface-cli login` beforehand, or pass --hf-token.

Usage:
    python run_pannuke_pipeline.py \\
        --output-root ./runs/pannuke_run1 \\
        --hf-token hf_xxx \\
        --epochs 20 --steps-per-epoch 200 --batch-size 64

    # Re-probe an existing checkpoint without retraining:
    python run_pannuke_pipeline.py \\
        --output-root ./runs/pannuke_run1 \\
        --skip-extraction --skip-training \\
        --checkpoint ./runs/pannuke_run1/checkpoint_epoch10.pth
"""

import multiprocessing as mp

if __name__ == "__main__":
    # Must happen before any CUDA call in this process. See "FIX APPLIED"
    # in the module docstring above -- this avoids the fork-after-CUDA-init
    # DataLoader worker deadlock.
    mp.set_start_method("spawn", force=True)

import argparse
import copy
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset, DataLoader
from torchvision import transforms

from transformers import AutoModel
from transformers.models.dinov3_convnext.modeling_dinov3_convnext import (
    DINOv3ConvNextConfig,
    DINOv3ConvNextModel,
)

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

MODEL_NAME = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"
FEAT_DIMS = {0: 96, 1: 192, 2: 384, 3: 768}
DEEP_STAGE = 3


# ============================================================================
# SECTION 1: Backbone -- atrous-adapted DINOv3 ConvNeXt-Tiny.
# ============================================================================

def adapt_for_lowres_atrous(model: DINOv3ConvNextModel,
                             reduce_stages: tuple = (0, 1)) -> DINOv3ConvNextModel:
    stages = model.model.stages
    dilation_multiplier = 1

    for stage_idx in range(len(stages)):
        stage = stages[stage_idx]
        downsample_conv = (stage.downsample_layers[0] if stage_idx == 0
                            else stage.downsample_layers[-1])
        orig_stride = downsample_conv.stride[0]

        if stage_idx in reduce_stages and orig_stride > 1:
            new_stride = max(1, orig_stride // 2)
            reduction_factor = orig_stride // new_stride
            downsample_conv.stride = (new_stride, new_stride)
            k = downsample_conv.kernel_size[0]
            pad = max(0, (k - new_stride) // 2)
            downsample_conv.padding = (pad, pad)
            dilation_multiplier *= reduction_factor

        if dilation_multiplier > 1:
            for layer in stage.layers:
                dw = layer.depthwise_conv
                dw.dilation = (dilation_multiplier, dilation_multiplier)
                k = dw.kernel_size[0]
                dw.padding = (dilation_multiplier * (k - 1) // 2,) * 2

    return model


def load_dinov3_convnext_atrous(pretrained: bool = True, hf_token: str = None,
                                 reduce_stages: tuple = (0, 1)):
    if pretrained:
        model = AutoModel.from_pretrained(MODEL_NAME, token=hf_token)
    else:
        cfg = DINOv3ConvNextConfig(hidden_sizes=[96, 192, 384, 768], depths=[3, 3, 9, 3])
        model = DINOv3ConvNextModel(cfg)
    return adapt_for_lowres_atrous(model, reduce_stages=reduce_stages)


# ============================================================================
# SECTION 2: DINO head, multi-stage self-distillation (BYOT), DINO EMA loss.
# ============================================================================

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim=4096, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        # original0 = magnitude (g), original1 = direction (v) -- confirmed against
        # PyTorch's own migration docs for nn.utils.parametrizations.weight_norm
        # ("The magnitude (weight_g) and direction (weight_v) are now expressed as
        # parametrizations.weight.original0 and .original1 respectively"). Freezing
        # original0 here at a fixed magnitude of 1 while leaving original1 (direction)
        # trainable is exactly DINOv2's own last-layer freeze (weight_g fixed, weight_v
        # trainable, under the old pre-parametrize weight_norm API) -- not a partial or
        # incorrect version of it.
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(x, dim=-1, p=2)
        return self.last_layer(x)


class MultiStageBackbone(nn.Module):
    def __init__(self, pretrained: bool, hf_token, reduce_stages=(0, 1),
                 aux_stages=(1, 2), out_dim=4096):
        super().__init__()
        self.backbone = load_dinov3_convnext_atrous(
            pretrained=pretrained, hf_token=hf_token, reduce_stages=reduce_stages)
        self.aux_stages = aux_stages
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.deep_head = DINOHead(FEAT_DIMS[DEEP_STAGE], out_dim)
        self.aux_heads = nn.ModuleDict({
            str(s): DINOHead(FEAT_DIMS[s], out_dim) for s in aux_stages
        })

    def forward(self, x):
        out = self.backbone(x, output_hidden_states=True)
        hidden_states = out.hidden_states

        deep_feat = self.pool(hidden_states[DEEP_STAGE + 1]).flatten(1)
        deep_out = self.deep_head(deep_feat)

        aux_outs = {}
        for s in self.aux_stages:
            feat = self.pool(hidden_states[s + 1]).flatten(1)
            aux_outs[s] = self.aux_heads[str(s)](feat)

        return deep_out, aux_outs


class StudentTeacherMultiStage(nn.Module):
    def __init__(self, pretrained: bool, hf_token, reduce_stages=(0, 1),
                 aux_stages=(1, 2), out_dim=4096):
        super().__init__()
        self.student = MultiStageBackbone(pretrained, hf_token, reduce_stages, aux_stages, out_dim)

        self.teacher_backbone = copy.deepcopy(self.student.backbone)
        self.teacher_pool = nn.AdaptiveAvgPool2d(1)
        self.teacher_deep_head = copy.deepcopy(self.student.deep_head)
        for p in list(self.teacher_backbone.parameters()) + list(self.teacher_deep_head.parameters()):
            p.requires_grad = False

    def teacher_forward(self, x):
        out = self.teacher_backbone(x, output_hidden_states=True)
        feat = self.teacher_pool(out.hidden_states[DEEP_STAGE + 1]).flatten(1)
        return self.teacher_deep_head(feat)

    @torch.no_grad()
    def update_teacher(self, momentum: float):
        for ps, pt in zip(self.student.backbone.parameters(), self.teacher_backbone.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)
        for ps, pt in zip(self.student.deep_head.parameters(), self.teacher_deep_head.parameters()):
            pt.data.mul_(momentum).add_(ps.data, alpha=1 - momentum)


class DINOLoss(nn.Module):
    def __init__(self, out_dim, student_temp=0.1, teacher_temp=0.04, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_out, teacher_out, teacher_temp=None):
        temp = teacher_temp if teacher_temp is not None else self.teacher_temp
        s = [x / self.student_temp for x in student_out]
        t = [F.softmax((x - self.center) / temp, dim=-1).detach() for x in teacher_out]

        total_loss, n_terms = 0.0, 0
        for ti, tq in enumerate(t):
            for si, sq in enumerate(s):
                if si == ti:
                    continue
                loss = torch.sum(-tq * F.log_softmax(sq, dim=-1), dim=-1)
                total_loss = total_loss + loss.mean()
                n_terms += 1
        total_loss = total_loss / n_terms

        self._update_center(torch.cat(teacher_out))
        return total_loss

    @torch.no_grad()
    def _update_center(self, teacher_output):
        """
        NOT distributed-safe as written: on multiple GPUs under DDP, each
        process would compute its own local batch_center from only the
        samples it sees, with no torch.distributed.all_reduce to average
        across ranks first -- so self.center would diverge per-process.
        This is currently a non-issue: this script has no torch.distributed
        setup anywhere (single GPU only, confirmed by the #SBATCH --gres=gpu:1
        SLURM config this has been run under). If multi-GPU DDP support is
        ever added, this method needs:
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(batch_center)
                batch_center /= torch.distributed.get_world_size()
        added before the momentum update below -- but adding that now, with
        no process group ever initialized, would either no-op uselessly or
        error out. Left as single-GPU-only until DDP is actually wired up.
        """
        batch_center = teacher_output.mean(dim=0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)


class BYOTLoss(nn.Module):
    """KL divergence: auxiliary (shallow) heads vs. detached deep-head output, same view."""

    def __init__(self, temp=0.1):
        super().__init__()
        self.temp = temp

    def forward(self, aux_outs: dict, deep_out: torch.Tensor, aux_weights: dict):
        deep_target = F.softmax(deep_out.detach() / self.temp, dim=-1)
        total = torch.zeros((), device=deep_out.device, dtype=deep_out.dtype)
        for stage_idx, aux_out in aux_outs.items():
            log_p = F.log_softmax(aux_out / self.temp, dim=-1)
            kl = F.kl_div(log_p, deep_target, reduction="batchmean")
            total = total + aux_weights.get(stage_idx, 1.0) * kl
        return total


# ============================================================================
# SECTION 3: Multi-crop augmentation for 64x64 cell crops.
# ============================================================================

class MultiCropTransform:
    """
    Picklable multi-crop transform. Must be a module-level class rather than
    the closure this used to be (`def apply(img): ...` nested inside
    make_multicrop_transform) -- under the `fork` multiprocessing start
    method, closures worked fine because child worker processes inherit the
    parent's memory directly, no pickling involved. Under `spawn` (which we
    now use to avoid the CUDA-before-fork deadlock), the entire Dataset --
    including its `.transform` attribute -- must be pickled to hand off to
    each new worker process, and Python's pickle module cannot serialize
    local/nested functions, only module-level ones.
    """

    def __init__(self, global_size=64, local_size=32, n_global=2, n_local=1):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.global_t = transforms.Compose([
            transforms.RandomResizedCrop(global_size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(), normalize,
        ])
        self.local_t = transforms.Compose([
            transforms.RandomResizedCrop(local_size, scale=(0.3, 0.6)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.2, 0.2, 0.1, 0.05),
            transforms.ToTensor(), normalize,
        ])
        self.n_global = n_global
        self.n_local = n_local

    def __call__(self, img):
        return ([self.global_t(img) for _ in range(self.n_global)]
                + [self.local_t(img) for _ in range(self.n_local)])


def make_multicrop_transform(global_size=64, local_size=32, n_global=2, n_local=1):
    return MultiCropTransform(global_size, local_size, n_global, n_local)


def multicrop_collate(batch):
    views_lists = [item[0] for item in batch]
    n_views = len(views_lists[0])
    return [torch.stack([v[i] for v in views_lists]) for i in range(n_views)]


# ============================================================================
# SECTION 4: Training dataset (folder-of-classes layout).
# ============================================================================

class PreExtractedCellDataset(Dataset):
    """root/ClassX/Images/*.{png,jpg,jpeg} -- label folders ignored for SSL."""
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, transform=None, dataset_name="Dataset"):
        self.root = Path(root)
        self.transform = transform
        self.dataset_name = dataset_name
        self.samples = []
        class_counts = Counter()

        for class_folder in sorted(self.root.iterdir()):
            if not class_folder.is_dir():
                continue
            images_dir = class_folder / "Images"
            search_dir = images_dir if images_dir.exists() else class_folder
            for img_path in sorted(search_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)
                    class_counts[class_folder.name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No crops found in {root}")

        print(f"[{self.dataset_name}] {len(self.samples)} crops: "
              f"{dict(sorted(class_counts.items()))}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img = Image.open(self.samples[idx]).convert("RGB")
        views = self.transform(img) if self.transform is not None else img
        return views, ()


# ============================================================================
# SECTION 5: PanNuke download + extraction.
#
#    RationAI/PanNuke on the HF Hub -- public, non-gated. Each row has an
#    image (256x256 RGB), a list of per-nucleus binary instance masks, and
#    aligned category labels (0=Neoplastic, 1=Inflammatory, 2=Connective,
#    3=Dead, 4=Epithelial).
#
#    Trains on fold1+fold2, holds fold3 out entirely for evaluation -- a
#    genuinely unseen test set, not a split of the training pool.
# ============================================================================

CATEGORY_NAMES = {
    0: "Neoplastic", 1: "Inflammatory", 2: "Connective",
    3: "Dead", 4: "Epithelial",
}


def extract_crops_from_row(image, instances, categories, row_id,
                            output_dir, patch_size=64, min_mask_pixels=10):
    """
    For each nucleus instance: compute its mask centroid, crop a
    patch_size x patch_size window centered there from the RGB image
    (zero-padded at borders), save to output_dir/<CategoryName>/Images/.

    Mask decoding uses .convert("L") + threshold rather than
    np.array(mask_img, dtype=bool) directly -- the latter can silently
    return an all-False array depending on how the mode-'1' image was
    encoded (confirmed this failure mode while building this extraction
    logic against synthetic data matching PanNuke's schema).
    """
    img_arr = np.array(image.convert("RGB"))
    H, W = img_arr.shape[:2]
    half = patch_size // 2
    n_saved, n_skipped = 0, 0

    for inst_idx, (mask_img, cat) in enumerate(zip(instances, categories)):
        mask = np.asarray(mask_img.convert("L")) > 0
        ys, xs = np.nonzero(mask)
        if len(ys) < min_mask_pixels:
            n_skipped += 1
            continue

        cy, cx = int(ys.mean()), int(xs.mean())
        y1, y2 = cy - half, cy + half
        x1, x2 = cx - half, cx + half
        y1_src, y2_src = max(0, y1), min(H, y2)
        x1_src, x2_src = max(0, x1), min(W, x2)

        patch = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        y_off, x_off = y1_src - y1, x1_src - x1
        patch[y_off:y_off + (y2_src - y1_src), x_off:x_off + (x2_src - x1_src)] = \
            img_arr[y1_src:y2_src, x1_src:x2_src]

        cls_name = CATEGORY_NAMES.get(int(cat), f"class{cat}")
        out_dir = output_dir / cls_name / "Images"
        out_dir.mkdir(parents=True, exist_ok=True)
        Image.fromarray(patch).save(out_dir / f"{row_id}_{inst_idx}.png")
        n_saved += 1

    return n_saved, n_skipped


def extract_fold_set(fold_dict, output_dir, log_every=200):
    output_dir.mkdir(parents=True, exist_ok=True)
    total_saved, total_skipped = 0, 0
    class_counts = Counter()
    for fold_name, ds in fold_dict.items():
        t0 = time.time()
        for row_idx, row in enumerate(ds):
            saved, skipped = extract_crops_from_row(
                row["image"], row["instances"], row["categories"],
                row_id=f"{fold_name}_{row_idx}", output_dir=output_dir,
            )
            total_saved += saved
            total_skipped += skipped
            for cat in row["categories"]:
                class_counts[CATEGORY_NAMES.get(int(cat), f"class{cat}")] += 1
            if row_idx % log_every == 0:
                print(f"  [{fold_name} {row_idx}/{len(ds)}] saved={total_saved} "
                      f"skipped={total_skipped} ({time.time()-t0:.1f}s)")
        print(f"  [{fold_name}] done: {len(ds)} rows in {time.time()-t0:.1f}s")
    print(f"{output_dir}: {total_saved} crops saved, {total_skipped} skipped")
    print("Per-class counts:", dict(class_counts))
    return total_saved, class_counts


def download_and_extract_pannuke(train_dir: Path, eval_dir: Path):
    from datasets import load_dataset

    print("Downloading PanNuke folds from RationAI/PanNuke...")
    fold1 = load_dataset("RationAI/PanNuke", split="fold1")
    fold2 = load_dataset("RationAI/PanNuke", split="fold2")
    fold3 = load_dataset("RationAI/PanNuke", split="fold3")
    print(f"fold1: {len(fold1)}  fold2: {len(fold2)}  fold3: {len(fold3)}")

    print("\n=== Extracting TRAIN crops (fold1 + fold2) ===")
    extract_fold_set({"fold1": fold1, "fold2": fold2}, train_dir)

    print("\n=== Extracting EVAL crops (fold3, held out from training) ===")
    extract_fold_set({"fold3": fold3}, eval_dir)


# ============================================================================
# SECTION 6: Cosine schedules.
# ============================================================================

def cosine_schedule(base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0):
    warmup_iters = min(warmup_iters, total_iters)
    warmup = np.linspace(start_warmup_value, base_value, warmup_iters) if warmup_iters > 0 else np.array([])
    remaining = total_iters - warmup_iters
    if remaining > 0:
        iters = np.arange(remaining)
        cosine = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / max(remaining - 1, 1)))
    else:
        cosine = np.array([])
    schedule = np.concatenate([warmup, cosine])
    assert len(schedule) == total_iters, (len(schedule), total_iters)
    return schedule


def set_seed(seed: int):
    """
    Seeds python's random, numpy, and torch (CPU + all CUDA devices) so runs
    are reproducible given the same seed and hardware. Note: full bit-for-bit
    reproducibility also requires torch.use_deterministic_algorithms(True),
    which is NOT set here since it can meaningfully slow down some ops (and
    isn't compatible with every op this model uses) -- this gives you
    seeded-but-not-strictly-deterministic runs, which is what args.seed was
    presumably intended for given it's also used to seed the train/test
    split in the linear probe.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# SECTION 7: Training.
# ============================================================================

def training_step(model, dino_criterion, byot_criterion, views, n_global, momentum,
                   teacher_temp, aux_weights, byot_weight, optimizer, clip_grad,
                   verbose_timing=False, amp_dtype=None, scaler=None):
    """
    amp_dtype: None disables AMP entirely (full fp32, original behavior).
        torch.bfloat16 or torch.float16 enables torch.autocast with that dtype.
    scaler: a torch.cuda.amp.GradScaler, required when amp_dtype=torch.float16
        (fp16's narrow exponent range can under/overflow gradients, so loss
        scaling is needed). Not needed for bfloat16 -- same exponent range as
        fp32, so no scaling required; pass scaler=None for the bf16 path.
    """
    optimizer.zero_grad(set_to_none=True)
    use_amp = amp_dtype is not None

    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
        student_deep_outs, student_aux_outs_per_view = [], []
        for vi, v in enumerate(views):
            if verbose_timing:
                torch.cuda.synchronize()
                t_v0 = time.time()
            deep_out, aux_outs = model.student(v)
            if verbose_timing:
                torch.cuda.synchronize()
                print(f"    [timing] student view {vi} ({v.shape[0]}x{v.shape[-1]}px): "
                      f"{time.time() - t_v0:.2f}s")
            student_deep_outs.append(deep_out)
            student_aux_outs_per_view.append(aux_outs)

        if verbose_timing:
            torch.cuda.synchronize()
            t_teacher0 = time.time()
        with torch.no_grad():
            teacher_outs = [model.teacher_forward(v) for v in views[:n_global]]
        if verbose_timing:
            torch.cuda.synchronize()
            print(f"    [timing] teacher forward ({n_global} views): {time.time() - t_teacher0:.2f}s")

        dino_loss = dino_criterion(student_deep_outs, teacher_outs, teacher_temp=teacher_temp)

        byot_losses = [byot_criterion(aux_outs, deep_out, aux_weights)
                       for deep_out, aux_outs in zip(student_deep_outs, student_aux_outs_per_view)]
        byot_loss = sum(byot_losses) / len(byot_losses)

        total_loss = dino_loss + byot_weight * byot_loss

    if verbose_timing:
        torch.cuda.synchronize()
        t_bwd0 = time.time()

    if scaler is not None:
        scaler.scale(total_loss).backward()
        if clip_grad is not None:
            scaler.unscale_(optimizer)  # must unscale before clipping, or the norm is wrong
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
    else:
        total_loss.backward()
        if clip_grad is not None:
            torch.nn.utils.clip_grad_norm_(model.student.parameters(), clip_grad)
        optimizer.step()

    if verbose_timing:
        torch.cuda.synchronize()
        print(f"    [timing] backward+step: {time.time() - t_bwd0:.2f}s")

    model.update_teacher(momentum)
    return total_loss, dino_loss, byot_loss


def run_training(args, device) -> Path:
    output_dir = Path(args.output_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    transform = make_multicrop_transform(
        global_size=args.global_crop_size, local_size=args.local_crop_size,
        n_global=args.n_global_crops, n_local=args.n_local_crops,
    )
    dataset = PreExtractedCellDataset(args.train_data_root, transform=transform,
                                       dataset_name="train")

    real_steps_per_epoch = len(dataset) // args.batch_size
    if args.steps_per_epoch is None:
        args.steps_per_epoch = real_steps_per_epoch
        print(f"--steps-per-epoch not set -- defaulting to {real_steps_per_epoch} "
              f"(len(dataset) // batch_size), so one 'epoch' here means one real pass "
              f"over the dataset.")
    elif args.steps_per_epoch != real_steps_per_epoch:
        total_passes = (args.epochs * args.steps_per_epoch) / real_steps_per_epoch
        print(f"NOTE: --steps-per-epoch={args.steps_per_epoch} does not match a real "
              f"dataset pass ({real_steps_per_epoch} steps for batch_size={args.batch_size}). "
              f"'Epoch' here is just a checkpointing/logging cadence, not a literal single "
              f"pass over the data -- the DataLoader's own shuffle-and-restart (inside "
              f"infinite_loader) still guarantees every sample is seen once per real pass, "
              f"independent of how 'epoch' is defined for logging. Over the full "
              f"{args.epochs}-epoch run, the dataset will be traversed roughly "
              f"{total_passes:.1f} times.")

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
        collate_fn=multicrop_collate, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    aux_weights = dict(zip(args.aux_stages, args.aux_weight))
    if len(aux_weights) != len(args.aux_stages):
        raise ValueError("--aux-stages and --aux-weight must have matching, non-duplicate lengths")

    model = StudentTeacherMultiStage(
        pretrained=args.pretrained, hf_token=args.hf_token,
        reduce_stages=tuple(args.reduce_stages), aux_stages=tuple(args.aux_stages),
        out_dim=args.out_dim,
    ).to(device)
    dino_criterion = DINOLoss(out_dim=args.out_dim, teacher_temp=args.teacher_temp).to(device)
    byot_criterion = BYOTLoss(temp=args.byot_temp).to(device)

    backbone_params = list(model.student.backbone.parameters())
    head_params = (list(model.student.deep_head.parameters())
                   + list(model.student.aux_heads.parameters()))
    optimizer = torch.optim.AdamW([
        {"params": backbone_params, "lr_multiplier": args.backbone_lr_multiplier},
        {"params": head_params, "lr_multiplier": 1.0},
    ], lr=args.lr, weight_decay=args.weight_decay)

    # --- AMP setup ---
    # bf16 needs no GradScaler (same exponent range as fp32 -- no under/overflow
    # risk from scaling). fp16 does, since its narrow exponent range means small
    # gradients can flush to zero without loss scaling. "none" disables AMP
    # entirely, matching the original full-fp32 behavior.
    amp_dtype = None
    scaler = None
    if device.type == "cuda" and args.amp_dtype != "none":
        amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[args.amp_dtype]
        if args.amp_dtype == "fp16":
            scaler = torch.cuda.amp.GradScaler()
        print(f"AMP enabled: dtype={args.amp_dtype}"
              f"{' (with GradScaler)' if scaler is not None else ' (no GradScaler needed)'}")
    else:
        print("AMP disabled -- training in full fp32.")

    max_iter = args.epochs * args.steps_per_epoch
    warmup_iters = args.warmup_epochs * args.steps_per_epoch
    lr_schedule = cosine_schedule(args.lr, args.min_lr, max_iter, warmup_iters)
    mom_schedule = cosine_schedule(args.momentum_teacher, args.final_momentum_teacher, max_iter)
    tt_warmup_iters = args.warmup_teacher_temp_epochs * args.steps_per_epoch
    teacher_temp_schedule = np.concatenate([
        np.linspace(args.warmup_teacher_temp, args.teacher_temp, tt_warmup_iters),
        np.full(max(0, max_iter - tt_warmup_iters), args.teacher_temp),
    ])[:max_iter]

    if tt_warmup_iters > max_iter:
        final_temp_reached = teacher_temp_schedule[-1]
        print(f"WARNING: --warmup-teacher-temp-epochs {args.warmup_teacher_temp_epochs} "
              f"({tt_warmup_iters} steps) exceeds total training length ({max_iter} steps). "
              f"Teacher temperature will only reach {final_temp_reached:.4f} by the end of this "
              f"run, never its target {args.teacher_temp}. Lower --warmup-teacher-temp-epochs "
              f"or extend the run.")

    # --- Checkpoint resume ---
    start_iteration = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.student.load_state_dict(ckpt["student"])
        model.teacher_backbone.load_state_dict(ckpt["teacher_backbone"])
        model.teacher_deep_head.load_state_dict(ckpt["teacher_deep_head"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler is not None and ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_iteration = ckpt["iteration"] + 1
        print(f"Resumed at iteration {start_iteration} (checkpoint was saved after "
              f"epoch {ckpt.get('epoch')}, iteration {ckpt['iteration']}).")
        if start_iteration >= max_iter:
            raise RuntimeError(
                f"Checkpoint's next iteration ({start_iteration}) is already >= "
                f"--epochs x --steps-per-epoch ({max_iter}). Increase --epochs or "
                f"--steps-per-epoch to continue training past this checkpoint."
            )

    # Backbone freeze: resume-aware. If we're resuming past the point where
    # the original run would have unfrozen the backbone, don't re-freeze it.
    backbone_currently_frozen = False
    if args.backbone_freeze_steps > 0:
        if start_iteration < args.backbone_freeze_steps:
            for p_ in model.student.backbone.parameters():
                p_.requires_grad_(False)
            backbone_currently_frozen = True
            print(f"Backbone frozen for the first {args.backbone_freeze_steps} steps "
                  f"-- only the DINO/BYOT heads train until then.")
        else:
            print(f"Resuming at iteration {start_iteration}, past --backbone-freeze-steps="
                  f"{args.backbone_freeze_steps} -- backbone starts unfrozen.")

    if 0 < args.backbone_freeze_steps < tt_warmup_iters:
        print(f"NOTE: backbone unfreezes at step {args.backbone_freeze_steps}, while teacher "
              f"temperature is still warming up (reaches target at step {tt_warmup_iters}).")

    print(f"\nStarting training: {max_iter} total steps "
          f"({args.epochs} epochs x {args.steps_per_epoch} steps), "
          f"starting from iteration {start_iteration}\n")
    print(f"DataLoader: num_workers={args.num_workers}, batch_size={args.batch_size}, "
          f"views_per_sample={args.n_global_crops + args.n_local_crops} "
          f"-- first batch may take a little while to warm up.\n")

    def infinite_loader(dl):
        while True:
            for batch in dl:
                yield batch

    data_iter = infinite_loader(loader)
    running_total, running_dino, running_byot, running_count = 0.0, 0.0, 0.0, 0
    t0 = time.time()
    last_checkpoint = None

    model.train()
    for iteration in range(start_iteration, max_iter):
        if backbone_currently_frozen and iteration == args.backbone_freeze_steps:
            for p_ in model.student.backbone.parameters():
                p_.requires_grad_(True)
            print(f"[iter {iteration}] Backbone unfrozen -- full fine-tuning begins now.")

        if iteration == 0:
            print("[iter 0] Fetching first batch from DataLoader (this includes spawn worker "
                  "warmup)...", flush=True)
            t_fetch0 = time.time()
        views = next(data_iter)
        if iteration == 0:
            print(f"[iter 0] First batch fetched in {time.time() - t_fetch0:.2f}s", flush=True)
        views = [v.to(device, non_blocking=True) for v in views]

        for g in optimizer.param_groups:
            g["lr"] = lr_schedule[iteration] * g["lr_multiplier"]

        verbose_timing = iteration < 3
        if verbose_timing:
            print(f"[iter {iteration}] Starting forward/backward with per-stage timing...", flush=True)

        total_loss, dino_loss, byot_loss = training_step(
            model, dino_criterion, byot_criterion, views, args.n_global_crops,
            momentum=mom_schedule[iteration],
            teacher_temp=teacher_temp_schedule[iteration],
            aux_weights=aux_weights, byot_weight=args.byot_weight,
            optimizer=optimizer, clip_grad=args.clip_grad,
            verbose_timing=verbose_timing,
            amp_dtype=amp_dtype, scaler=scaler,
        )

        if math.isnan(total_loss.item()):
            raise RuntimeError(f"NaN loss at iteration {iteration}")

        running_total += total_loss.item()
        running_dino += dino_loss.item()
        running_byot += byot_loss.item()
        running_count += 1
        epoch = iteration // args.steps_per_epoch

        if iteration % 20 == 0:
            avg_total = running_total / max(running_count, 1)
            avg_dino = running_dino / max(running_count, 1)
            avg_byot = running_byot / max(running_count, 1)
            elapsed = time.time() - t0
            print(f"[epoch {epoch} iter {iteration}/{max_iter}] "
                  f"loss={avg_total:.4f} (dino={avg_dino:.4f} byot={avg_byot:.4f}) "
                  f"lr_head={lr_schedule[iteration]:.2e} "
                  f"lr_backbone={lr_schedule[iteration] * args.backbone_lr_multiplier:.2e} "
                  f"mom={mom_schedule[iteration]:.5f} ({elapsed:.1f}s elapsed)")
            running_total, running_dino, running_byot, running_count = 0.0, 0.0, 0.0, 0

        is_epoch_boundary = (iteration + 1) % args.steps_per_epoch == 0
        is_last_step = iteration == max_iter - 1
        if is_epoch_boundary and ((epoch + 1) % args.checkpoint_every_epochs == 0 or is_last_step):
            ckpt_path = output_dir / f"checkpoint_epoch{epoch + 1}.pth"
            torch.save({
                "student": model.student.state_dict(),
                "teacher_backbone": model.teacher_backbone.state_dict(),
                "teacher_deep_head": model.teacher_deep_head.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler is not None else None,
                "iteration": iteration,
                "epoch": epoch,
                "args": vars(args),
            }, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")
            last_checkpoint = ckpt_path

    print("Training complete.")
    return last_checkpoint


# ============================================================================
# SECTION 8: Linear probe evaluation.
# ============================================================================

class LabeledCellDataset(Dataset):
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        self.classes = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        if len(self.classes) == 0:
            raise RuntimeError(f"No class subfolders found in {root}")
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        for cls in self.classes:
            images_dir = self.root / cls / "Images"
            search_dir = images_dir if images_dir.exists() else self.root / cls
            for img_path in sorted(search_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append((img_path, self.class_to_idx[cls]))

        if len(self.samples) == 0:
            raise RuntimeError(f"No labeled crops found in {root}")

        counts = Counter(c for _, c in self.samples)
        summary = ", ".join(f"{self.classes[i]}: {counts[i]}" for i in sorted(counts))
        print(f"[LabeledCellDataset] {len(self.samples)} crops, "
              f"{len(self.classes)} classes: {{{summary}}}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def make_eval_transform(size=64):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


@torch.no_grad()
def extract_features(backbone, loader, device):
    backbone.eval()
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        out = backbone(imgs)
        all_feats.append(out.pooler_output.cpu().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


def load_teacher_backbone(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    reduce_stages = tuple(ckpt.get("args", {}).get("reduce_stages", [0, 1]))
    backbone = load_dinov3_convnext_atrous(pretrained=False, reduce_stages=reduce_stages)
    backbone.load_state_dict(ckpt["teacher_backbone"], strict=True)
    backbone = backbone.to(device)
    print(f"Loaded teacher backbone from {checkpoint_path} "
          f"(epoch {ckpt.get('epoch')}, iteration {ckpt.get('iteration')}, "
          f"reduce_stages={reduce_stages})")
    return backbone


def run_linear_probe(args, checkpoint_path, device):
    dataset = LabeledCellDataset(args.eval_data_root, transform=make_eval_transform())
    loader = DataLoader(dataset, batch_size=args.probe_batch_size, shuffle=False,
                         num_workers=args.num_workers)

    backbone = load_teacher_backbone(checkpoint_path, device)

    print("Extracting frozen features...")
    feats, labels = extract_features(backbone, loader, device)
    print(f"Feature matrix: {feats.shape}, labels: {labels.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        feats, labels, test_size=args.probe_test_size, random_state=args.seed, stratify=labels,
    )
    print(f"Probe train: {X_train.shape[0]}  Probe test: {X_test.shape[0]}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    clf = LogisticRegression(max_iter=args.probe_max_iter)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    overall_acc = accuracy_score(y_test, y_pred)
    print(f"\nOverall accuracy: {overall_acc:.4f}\n")

    labels_present = sorted(set(y_test) | set(y_pred))
    target_names = [dataset.classes[i] for i in labels_present]

    report = classification_report(
        y_test, y_pred, labels=labels_present, target_names=target_names,
        digits=3, zero_division=0,
    )
    print("Per-class report:")
    print(report)

    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    print("Confusion matrix (rows=true, cols=pred):")
    header = "          " + " ".join(f"{c[:8]:>9}" for c in target_names)
    print(header)
    for i, row in enumerate(cm):
        print(f"{target_names[i][:9]:>9} " + " ".join(f"{v:>9}" for v in row))

    report_dict = classification_report(
        y_test, y_pred, labels=labels_present, target_names=target_names,
        digits=3, zero_division=0, output_dict=True,
    )
    out = {
        "overall_accuracy": float(overall_acc),
        "per_class": report_dict,
        "confusion_matrix": cm.tolist(),
        "classes": target_names,
        "checkpoint": str(checkpoint_path),
    }
    probe_json_path = Path(args.output_root) / "linear_probe_results.json"
    with open(probe_json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved metrics to {probe_json_path}")
    return out


# ============================================================================
# SECTION 9: CLI orchestration.
# ============================================================================

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--output-root", required=True,
                    help="All crops, checkpoints, and results go under here.")
    p.add_argument("--hf-token", default=None,
                    help="Needed for --pretrained unless already logged in via `huggingface-cli login`.")

    # Stage toggles
    p.add_argument("--skip-extraction", action="store_true",
                    help="Skip PanNuke download/extraction (crops must already exist under --output-root).")
    p.add_argument("--skip-training", action="store_true",
                    help="Skip training (use --checkpoint to specify which checkpoint to probe).")
    p.add_argument("--skip-probe", action="store_true", help="Skip the linear probe evaluation.")
    p.add_argument("--checkpoint", default=None,
                    help="Checkpoint to probe when --skip-training is set. Ignored otherwise "
                         "(the checkpoint training just produced is used automatically).")
    p.add_argument("--resume", default=None,
                    help="Path to a checkpoint to resume TRAINING from (student/teacher/optimizer "
                         "state + iteration count are restored). Separate from --checkpoint, which "
                         "is for probing only. Not compatible with --skip-training.")

    # Data
    p.add_argument("--extraction-log-every", type=int, default=200)

    # Model / training (mirrors train_cellconvnext.py)
    p.add_argument("--pretrained", action="store_true", default=True)
    p.add_argument("--amp-dtype", choices=["bf16", "fp16", "none"], default="bf16",
                    help="Mixed-precision dtype for training. bf16 (default) needs no "
                         "GradScaler and is the recommended choice on Ampere+ GPUs (A100 and "
                         "newer). fp16 uses a GradScaler for numerical stability. 'none' "
                         "disables AMP, training in full fp32 (original behavior).")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false",
                    help="Random-init instead of real DINOv3 weights (pipeline testing only).")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--steps-per-epoch", type=int, default=None,
                    help="Steps per logged/checkpointed 'epoch'. If unset, defaults to "
                         "len(dataset)//batch_size so one 'epoch' means one real pass over "
                         "the data. Set explicitly if you want a different checkpointing "
                         "cadence -- see the printed NOTE at startup explaining what this "
                         "does and doesn't guarantee about dataset coverage.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--global-crop-size", type=int, default=64,
                    help="Global crop size in pixels. Was hardcoded to 64 -- now configurable.")
    p.add_argument("--local-crop-size", type=int, default=48,
                    help="Local crop size in pixels. Was hardcoded to 32, which on a 64px base "
                         "image risks landing on background or only part of a nucleus (32px at "
                         "0.3-0.6 scale of a 64px image is a small, aggressive crop). Bumped to "
                         "48 by default -- override if you want the original 32 or something else.")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--backbone-lr-multiplier", type=float, default=0.1)
    p.add_argument("--min-lr", type=float, default=1e-6)
    p.add_argument("--weight-decay", type=float, default=0.04)
    p.add_argument("--warmup-epochs", type=int, default=5)
    p.add_argument("--out-dim", type=int, default=4096)
    p.add_argument("--reduce-stages", type=int, nargs="*", default=[0, 1])
    p.add_argument("--aux-stages", type=int, nargs="*", default=[1, 2])
    p.add_argument("--aux-weight", type=float, nargs="*", default=[0.2, 0.3])
    p.add_argument("--byot-weight", type=float, default=1.0)
    p.add_argument("--byot-temp", type=float, default=0.1)
    p.add_argument("--n-global-crops", type=int, default=2)
    p.add_argument("--n-local-crops", type=int, default=1)
    p.add_argument("--momentum-teacher", type=float, default=0.996)
    p.add_argument("--final-momentum-teacher", type=float, default=1.0)
    p.add_argument("--teacher-temp", type=float, default=0.04)
    p.add_argument("--warmup-teacher-temp", type=float, default=0.02)
    p.add_argument("--warmup-teacher-temp-epochs", type=int, default=10)
    p.add_argument("--clip-grad", type=float, default=3.0)
    p.add_argument("--backbone-freeze-steps", type=int, default=500)
    p.add_argument("--checkpoint-every-epochs", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)

    # Probe
    p.add_argument("--probe-batch-size", type=int, default=128)
    p.add_argument("--probe-test-size", type=float, default=0.2)
    p.add_argument("--probe-max-iter", type=int, default=2000)

    args = p.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    args.train_data_root = str(output_root / "pannuke_crops_train")
    args.eval_data_root = str(output_root / "pannuke_crops_eval")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # if not args.skip_extraction:
    #     download_and_extract_pannuke(Path(args.train_data_root), Path(args.eval_data_root))
    # else:
    #     print(f"Skipping extraction -- expecting crops already under {args.train_data_root} "
    #           f"and {args.eval_data_root}.")

    checkpoint_path = None
    if not args.skip_training:
        checkpoint_path = run_training(args, device)
        if checkpoint_path is None:
            raise RuntimeError("Training finished but no checkpoint was saved -- check "
                                "--checkpoint-every-epochs vs --epochs.")
    else:
        if args.resume is not None:
            raise ValueError("--resume continues training and is incompatible with "
                              "--skip-training. Use --checkpoint instead if you only want to probe.")
        if args.checkpoint is None:
            raise ValueError("--skip-training requires --checkpoint to specify what to probe.")
        checkpoint_path = Path(args.checkpoint)
        print(f"Skipping training -- using provided checkpoint: {checkpoint_path}")

    if not args.skip_probe:
        run_linear_probe(args, checkpoint_path, device)
    else:
        print("Skipping linear probe evaluation.")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()