# <img src="clite_logo.png" width="40" valign="middle"> C-LITE — A Cell-Level Representation Learning Model

<img src="clite_logo.png" alt="C-LITE — A Cell-Level Representation Learning Model" width="312">

C-LITE learns representations of individual cells from H&E histology. The backbone is a **ViT-5** vision transformer (`DinoViT5`) trained with DINO-style self-supervision at **patch size 4** on **64×64 cell-centred crops**. Each crop becomes a 16×16 grid of patch tokens, plus a CLS token and 4 register tokens.

---

## Downstream Tasks

- Cell Phenotyping
- Cell Type Classification
- Mitosis Classification
- Nucleus Classification — PanNuke, see [`vit5_p4_pannuke_hf_classification.ipynb`](vit5_p4_pannuke_hf_classification.ipynb)
- Hierarchical Subtype Classification

All tasks use the same protocol. The backbone is frozen, features are extracted once, and a **logistic-regression linear probe** (`C = 1/l2`, `l2 = 100 / (M·C)`, lbfgs) and a **cosine kNN** classifier are trained on top.

Metrics reported: Accuracy, Balanced Accuracy, Macro/Weighted F1, Macro Precision/Recall, Cohen's κ, macro AUROC, mAP, Log-loss, and per-class F1.

---

## Quick Start: PanNuke Nucleus Classification

The notebook is self-contained. The ViT-5 architecture is defined inline, so it needs no `timm`, `einops`, `flash-attn` or `xformers`. The data is downloaded directly from the Hugging Face Hub.

```bash
pip install torch torchvision datasets numpy pillow scikit-learn matplotlib tqdm
jupyter lab vit5_p4_pannuke_hf_classification.ipynb
```

Edit the configuration cell (§1) before running:

| Setting | What it is |
|---|---|
| `CHECKPOINT` | Path to the teacher checkpoint (`teacher_checkpoint.pth`, full `model_*.pth`, KD `student` checkpoint, or flat release) |
| `EXPECTED_PATCH_SIZE` | Sanity check against the checkpoint (`4`) |
| `ROPE_PT_SEQ_LEN` | RoPE reference grid. **Not stored in the checkpoint** and must match training. It is `224 // 16 = 14` when the model was built with `get_vit5_backbone(patch_size=4)` |
| `HF_CACHE_DIR` | Where the ~1.7 GB PanNuke download is cached |
| `CELL_CROP_ROOT` | Where the 64×64 cell crops are written |
| `FEATURE_MODE` | `"cls"` (default) or `"cls+patchmean"` |
| `NORM_MEAN` / `NORM_STD` | Must match training normalisation (ImageNet by default) |

### Data: PanNuke from Hugging Face

The notebook uses [`RationAI/PanNuke`](https://huggingface.co/datasets/RationAI/PanNuke): 7,901 tiles of 256×256 px (40×, 0.25 µm/px), 19 tissue types, 5 nucleus classes, and three official folds.

| Split | Fold |
|---|---|
| Train | Fold 1 |
| Val | Fold 2 |
| Test | Fold 3 |

Classes (label order used throughout): `Connective, Dead, Epithelial, Inflammatory, Neoplastic`.

### Cell-centric 64×64 crops

For every nucleus instance mask in every tile, the notebook:

1. Computes the mask's centroid (centre of mass).
2. Reflect-pads the tile by 32 px so nuclei at the tile edge still get a full crop.
3. Cuts a 64×64 crop with the centroid at pixel (32, 32).
4. Drops masks smaller than `MIN_NUCLEUS_AREA` (10 px), which are slivers of nuclei cut off at the tile border.

The crops are written as:

```
<CELL_CROP_ROOT>/
├── Fold1_cell_patches/<Class>/fold1_<tile>_<nucleus>_<Class>_<Tissue>.png
├── Fold1_manifest.csv      # tile, nucleus, class, tissue, centroid, area, border/padding flags, path
├── Fold2_cell_patches/...
└── Fold3_cell_patches/...
```

Cropping runs once. A fold is skipped on later runs if its manifest already exists; set `OVERWRITE_CROPS = True` to redo it.

---

## Feature Extraction

The backbone's hyper-parameters (patch size, width, depth, heads, registers, pos-embed grid, gated attention) are **read from the checkpoint's tensor shapes**. The weights are loaded strictly, so any missing or unexpected key raises an error.

Token layout for a 64×64 input at patch 4:

```
[CLS] + 16×16 patch tokens + 4 register tokens
```

Registers come **last**, unlike DINOv2, where they follow the CLS token. The absolute position embedding (56×56 in the checkpoint) is bicubically resized to 16×16. RoPE is applied to patch tokens, and a separate RoPE is applied to the registers.

Features are saved per split for reuse:

```
<OUTPUT_DIR>/<MODEL_TAG>_<FEATURE_MODE>_{train,val,test}_features.npz   # keys: features, labels, ids
<OUTPUT_DIR>/<MODEL_TAG>_<FEATURE_MODE>_pannuke_metrics.csv
<OUTPUT_DIR>/confusion_matrices.png
```

---

## Attention Visualization

The notebook's §10 draws CLS→patch attention maps for each head, plus the head mean, overlaid on the crop with a jet colormap.

- `visualizations/attention/attn_sample_<i>_<name>.png` shows an `image | mean | head 0 … head H-1` strip for each image.
- `visualizations/attention/attention_grid_block<k>.png` is an inline grid that also shows how much of the CLS token's attention goes to the register tokens.
- `visualizations/attention/attention_per_block.png` shows the head-averaged CLS attention at every depth.

> **ViT-5 note:** QK-RMSNorm and RoPE are applied to q and k *after* the `qkv` projection. Hooking `attn.qkv`, as the DINOv2 visualization scripts do, therefore gives incorrect maps. The notebook stores the true softmax attention from inside the chosen block instead.

---

## PCA Visualization

The notebook's §11 takes the final-norm patch tokens (CLS and registers dropped) and fits a 3-component whitened PCA. It maps the components to RGB with `sigmoid(2·x)`, bicubically upsamples to 64×64, and overlays the result on the image.

- `visualizations/pca/pca_<name>.png` is an `Image | PCA | Overlay` figure fitted per image.
- `visualizations/pca/pca_joint.png` uses one PCA basis fitted on all images, so a colour means the same thing across crops.

---

## t-SNE / UMAP Plots

The notebook doesn't make these plots yet. They can be drawn from the saved feature files:

```python
import numpy as np, matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap  # pip install umap-learn

CLASSES = ["Connective", "Dead", "Epithelial", "Inflammatory", "Neoplastic"]
d = np.load("vit5_small_p4_pannuke_results/vit5_small_p4_cls_test_features.npz")
X, y = d["features"], d["labels"]
X = X / np.linalg.norm(X, axis=1, keepdims=True)

emb = {
    "t-SNE": TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X),
    "UMAP":  umap.UMAP(n_neighbors=30, min_dist=0.1, metric="cosine", random_state=42).fit_transform(X),
}
fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
for ax, (name, Z) in zip(axes, emb.items()):
    for c, cls in enumerate(CLASSES):
        m = y == c
        ax.scatter(Z[m, 0], Z[m, 1], s=2, alpha=0.6, label=cls)
    ax.set_title(f"{name} — PanNuke Fold 3 (test)"); ax.set_xticks([]); ax.set_yticks([])
axes[1].legend(markerscale=6, fontsize=8)
plt.tight_layout(); plt.savefig("tsne_umap_test.png", dpi=200)
```

Fold 3 has roughly 60k+ nuclei, and t-SNE on all of them is slow. For a quick look, subsample to about 20k with stratified sampling, and keep all the *Dead* cells because that class is rare.

---

## Acknowledgements

PanNuke dataset: Gamper *et al.*, *PanNuke: an open pan-cancer histology dataset for nuclei instance segmentation and classification* (ECDP 2019) and *PanNuke Dataset Extension, Insights and Baselines* (arXiv:2003.10778). The Hugging Face release is by RationAI, licensed CC BY-NC-SA 4.0.
  
