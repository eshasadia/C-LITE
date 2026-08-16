# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# viz_utils.py -- lightweight, wandb-logged visualizations for tracking what
# DST-DINOv2 is actually learning during self-supervised training.
#
# IMPORTANT (FSDP): model.teacher["backbone"] / model.student["backbone"] are
# FSDP-wrapped. Calling their PUBLIC forward (i.e. backbone(x, ...) via
# __call__) is always safe -- FSDP's forward hooks unshard parameters for
# you. Calling a custom method directly (get_last_selfattention) bypasses
# those hooks, so that specific call is wrapped in
# FSDP.summon_full_params(..., writeback=False) to force full, correctly
# shaped parameters for its duration. Skipping this is what produced the
# "weight should have at least three dimensions" and CPU/Half-precision
# errors seen in practice.

import contextlib
import io
import logging

import numpy as np
import torch
import torch.nn.functional as F
import wandb
from PIL import Image

logger = logging.getLogger("dinov2")

try:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    _HAS_FSDP = True
except ImportError:
    _HAS_FSDP = False

try:
    import umap
    _HAS_UMAP = True
except ImportError:
    _HAS_UMAP = False
    logger.warning("umap-learn not installed -- embedding_umap visualization will be skipped. "
                    "Install with: pip install umap-learn")

try:
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False
    logger.warning("scikit-learn not installed -- patch PCA / linear probe will be skipped. "
                    "Install with: pip install scikit-learn")


def _full_params(module):
    """Context manager: temporarily materializes full, correctly-shaped
    parameters for an FSDP-wrapped module so a direct (non-__call__) method
    call sees real tensors instead of sharded/flat ones. No-op if the module
    isn't FSDP-wrapped or FSDP isn't available."""
    if _HAS_FSDP and isinstance(module, FSDP):
        return FSDP.summon_full_params(module, writeback=False)
    return contextlib.nullcontext()


def _resolve_device(model, device=None):
    """Never trust next(module.parameters()).device under FSDP sharding --
    it can report the wrong thing depending on what's currently materialized.
    Always pass device explicitly from train_dst.py (torch.device(f"cuda:{local_rank}"))."""
    if device is not None:
        return device
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _get_backbone(model, key):
    return model.teacher["backbone"] if key == "teacher" else model.student["backbone"]


def _model_param_dtype(backbone):
    """Match input dtype to the backbone's param dtype (fp16 per your
    compute_precision config) so we never trigger a CPU/Half fallback from a
    dtype mismatch."""
    try:
        return next(backbone.parameters()).dtype
    except StopIteration:
        return torch.float32


# =====================================================================
# 1. Self-attention maps (CLS token -> Micro tokens, per head)
# =====================================================================

@torch.no_grad()
def log_attention_maps(model, sample_imgs, iteration, key="teacher", prefix="attn", device=None):
    """
    sample_imgs: [N, 3, H, W] tensor of a handful of FIXED crops (same ones every
                 call) so you can watch attention evolve on the same cells.
    key: "teacher" or "student" -- which backbone to pull attention from.
    device: pass the training device explicitly (e.g. torch.device(f"cuda:{local_rank}")).
    """
    backbone = _get_backbone(model, key)
    backbone.eval()
    device = _resolve_device(model, device)
    param_dtype = _model_param_dtype(backbone)

    if not hasattr(backbone, "get_last_selfattention"):
        logger.warning("Backbone has no get_last_selfattention(); skipping attention viz.")
        return

    sample_imgs = sample_imgs.to(device=device, dtype=param_dtype)

    images_logged = []

    for i in range(sample_imgs.shape[0]):
        img = sample_imgs[i : i + 1]
        with _full_params(backbone):
            attn = backbone.get_last_selfattention(img)  # (1, nh, grid, grid) for this DST backbone
        attn = attn[0].float()  # (nh, grid, grid)
        img_size = img.shape[-1]
        attn = F.interpolate(attn.unsqueeze(0), size=(img_size, img_size), mode="nearest")[0]
        attn = attn.cpu().numpy()

        orig = _tensor_to_uint8(img[0])
        panel = _make_attention_panel(orig, attn)
        images_logged.append(wandb.Image(panel, caption=f"sample_{i}_{key}"))

    wandb.log({f"{prefix}/{key}_attention_grids": images_logged}, step=iteration)


def _tensor_to_uint8(img_tensor):
    """De-normalize-agnostic conversion: just min-max stretch to 0-255 for display."""
    img = img_tensor.detach().float().cpu()
    img = img - img.min()
    denom = img.max().clamp(min=1e-6)
    img = (img / denom * 255).clamp(0, 255).byte()
    return img.permute(1, 2, 0).numpy()  # HWC


def _make_attention_panel(orig_hwc, attn_maps):
    """Stack [original | head0 | head1 | ...] into one wide image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nh = attn_maps.shape[0]
    fig, axes = plt.subplots(1, nh + 1, figsize=(2 * (nh + 1), 2))
    axes[0].imshow(orig_hwc)
    axes[0].set_title("crop", fontsize=8)
    axes[0].axis("off")

    for h in range(nh):
        axes[h + 1].imshow(orig_hwc)
        axes[h + 1].imshow(attn_maps[h], cmap="jet", alpha=0.55)
        axes[h + 1].set_title(f"head {h}", fontsize=8)
        axes[h + 1].axis("off")

    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# =====================================================================
# 2. Embedding space structure (PCA/UMAP scatter of CLS tokens)
# =====================================================================

@torch.no_grad()
def log_embedding_projection(embeddings, labels, iteration, prefix="embed", method="umap"):
    """
    embeddings: [N, D] tensor/array of CLS tokens (e.g. reuse the same buffer
                you already build for smooth_rank_measure).
    labels: list[str] or list[int] of length N -- dataset name or class name
            per embedding, used only for coloring the plot, not for training.
    """
    emb = embeddings.detach().float().cpu().numpy() if torch.is_tensor(embeddings) else np.asarray(embeddings)

    if method == "umap" and _HAS_UMAP:
        reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1)
        proj = reducer.fit_transform(emb)
    elif _HAS_SKLEARN:
        proj = PCA(n_components=2).fit_transform(emb)
        method = "pca"
    else:
        logger.warning("Neither umap-learn nor scikit-learn available; skipping embedding projection.")
        return

    table = wandb.Table(
        data=[[float(x), float(y), str(l)] for (x, y), l in zip(proj, labels)],
        columns=["x", "y", "label"],
    )
    wandb.log(
        {f"{prefix}/{method}_projection": wandb.plot.scatter(table, "x", "y", title=f"{method} of CLS embeddings")},
        step=iteration,
    )


# =====================================================================
# 3. Nearest-neighbor retrieval sanity check
# =====================================================================

@torch.no_grad()
def log_nn_retrieval(model, query_imgs, gallery_imgs, gallery_labels, iteration, key="teacher",
                      top_k=5, prefix="nn_retrieval", device=None):
    """
    query_imgs:   [Nq, 3, H, W] -- small fixed set of query crops.
    gallery_imgs: [Ng, 3, H, W] -- reference pool to search against.
    gallery_labels: list[str], len Ng -- for captioning neighbors (class/dataset name).
    """
    device = _resolve_device(model, device)

    q_emb = _extract_embeddings(model, query_imgs, key=key, device=device)
    g_emb = _extract_embeddings(model, gallery_imgs, key=key, device=device)
    q_emb = F.normalize(torch.from_numpy(q_emb), dim=-1)
    g_emb = F.normalize(torch.from_numpy(g_emb), dim=-1)
    sims = q_emb @ g_emb.T  # [Nq, Ng]

    panels = []
    for i in range(q_emb.shape[0]):
        topk_idx = torch.topk(sims[i], k=min(top_k, g_emb.shape[0])).indices.tolist()
        panel = _make_nn_panel(
            _tensor_to_uint8(query_imgs[i]),
            [_tensor_to_uint8(gallery_imgs[j]) for j in topk_idx],
            [gallery_labels[j] for j in topk_idx],
        )
        panels.append(wandb.Image(panel, caption=f"query_{i}_{key}"))

    wandb.log({f"{prefix}/{key}_top{top_k}": panels}, step=iteration)


def _make_nn_panel(query_hwc, neighbor_hwcs, neighbor_labels):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(neighbor_hwcs)
    fig, axes = plt.subplots(1, n + 1, figsize=(2 * (n + 1), 2.3))
    axes[0].imshow(query_hwc)
    axes[0].set_title("query", fontsize=8)
    axes[0].axis("off")
    for i, (img, lbl) in enumerate(zip(neighbor_hwcs, neighbor_labels)):
        axes[i + 1].imshow(img)
        axes[i + 1].set_title(str(lbl), fontsize=7)
        axes[i + 1].axis("off")
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# =====================================================================
# 4. Patch-token PCA within a single image (foreground/object heatmap)
# =====================================================================

@torch.no_grad()
def log_patch_pca(model, sample_imgs, iteration, key="teacher", prefix="patch_pca", device=None):
    """
    Runs PCA across all Micro-token patches of each sample image and
    visualizes the top principal component as a heatmap -- classic DINO
    'object discovery for free' visualization, at cell-crop scale.
    """
    if not _HAS_SKLEARN:
        return

    backbone = _get_backbone(model, key)
    backbone.eval()
    device = _resolve_device(model, device)
    param_dtype = _model_param_dtype(backbone)
    sample_imgs = sample_imgs.to(device=device, dtype=param_dtype)

    micro_grid = getattr(backbone, "micro_grid", None)

    panels = []
    for i in range(sample_imgs.shape[0]):
        img = sample_imgs[i : i + 1]
        # Go through __call__ (not .forward_features directly) so FSDP's
        # forward hooks unshard parameters properly; is_training=True makes
        # the DST backbone's forward() return the full feature dict.
        out = backbone(img, is_training=True)
        patch_tokens = out["x_norm_patchtokens"] if isinstance(out, dict) else out
        patch_tokens = patch_tokens[0].float().cpu().numpy()  # [n_micro, D]

        if micro_grid is None:
            grid = int(round(patch_tokens.shape[0] ** 0.5))
        else:
            grid = micro_grid

        pc1 = PCA(n_components=1).fit_transform(patch_tokens).reshape(grid, grid)
        pc1 = (pc1 - pc1.min()) / (pc1.max() - pc1.min() + 1e-6)

        orig = _tensor_to_uint8(img[0])
        panel = _make_pca_panel(orig, pc1)
        panels.append(wandb.Image(panel, caption=f"sample_{i}_{key}"))

    wandb.log({f"{prefix}/{key}_pc1": panels}, step=iteration)


def _make_pca_panel(orig_hwc, pc1_grid):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))
    axes[0].imshow(orig_hwc)
    axes[0].set_title("crop", fontsize=8)
    axes[0].axis("off")
    axes[1].imshow(pc1_grid, cmap="viridis")
    axes[1].set_title("patch PC1", fontsize=8)
    axes[1].axis("off")
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


# =====================================================================
# 5. Lightweight linear probe (representation quality over time)
# =====================================================================

@torch.no_grad()
def _extract_embeddings(model, imgs, key="teacher", bs=64, device=None):
    backbone = _get_backbone(model, key)
    backbone.eval()
    device = _resolve_device(model, device)
    param_dtype = _model_param_dtype(backbone)

    outs = []
    for i in range(0, imgs.shape[0], bs):
        chunk = imgs[i : i + bs].to(device=device, dtype=param_dtype)
        # via __call__ -- FSDP-safe. Default is_training=False -> DST
        # backbone's forward() already returns just the CLS token tensor.
        out = backbone(chunk)
        cls = out["x_norm_clstoken"] if isinstance(out, dict) else out
        outs.append(cls.float().cpu().numpy())
    return np.concatenate(outs, axis=0)


def log_linear_probe(model, train_imgs, train_labels, val_imgs, val_labels, iteration,
                      key="teacher", prefix="linear_probe", max_iter=1000, device=None):
    """
    train/val_imgs: tensors of pre-cropped cell images (no augmentation needed).
    train/val_labels: int class labels (e.g. from CONIC or PanNuke, which you
                       already have labels for -- used ONLY to evaluate the
                       frozen representation, never to train the backbone).
    """
    if not _HAS_SKLEARN:
        return

    device = _resolve_device(model, device)
    train_emb = _extract_embeddings(model, train_imgs, key=key, device=device)
    val_emb = _extract_embeddings(model, val_imgs, key=key, device=device)

    clf = LogisticRegression(max_iter=max_iter, multi_class="auto")
    clf.fit(train_emb, train_labels)
    train_acc = clf.score(train_emb, train_labels)
    val_acc = clf.score(val_emb, val_labels)

    wandb.log({f"{prefix}/{key}_train_acc": train_acc, f"{prefix}/{key}_val_acc": val_acc}, step=iteration)
    logger.info(f"[linear probe @ iter {iteration}] {key} train_acc={train_acc:.4f} val_acc={val_acc:.4f}")
    return val_acc


# =====================================================================
# 6. Augmentation sanity check
# =====================================================================

def log_augmented_views(global_crops, iteration, prefix="aug_check", max_pairs=8):
    """
    global_crops: [2*B, 3, H, W] as produced by collate_data_and_cast
                  ("collated_global_crops"). Logs a handful of the two-view
                  pairs so you can eyeball whether augmentation is too
                  destructive for 64x64 crops.
    """
    B2 = global_crops.shape[0]
    B = B2 // 2
    n = min(max_pairs, B)

    panels = []
    for i in range(n):
        v1 = _tensor_to_uint8(global_crops[i])
        v2 = _tensor_to_uint8(global_crops[B + i])
        panels.append(wandb.Image(np.concatenate([v1, v2], axis=1), caption=f"pair_{i}"))

    wandb.log({f"{prefix}/global_view_pairs": panels}, step=iteration)