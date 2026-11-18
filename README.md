# <img src="clite_logo.png" width="40" valign="middle"> C-LITE — A Cell-Level Representation Learning Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GITHUB_USER>/<REPO>/blob/main/vit5_p4_pannuke_hf_classification.ipynb)

<img src="clite_logo.png" alt="C-LITE — A Cell-Level Representation Learning Model" width="312">

C-LITE learns representations of individual cells from H&E histology. 
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



