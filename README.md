# <img src="clite_logo.png" width="40" valign="middle"> C-LITE: A Lightweight Foundation Model for Cellular feature learning in Histopathology

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/<GITHUB_USER>/<REPO>/blob/main/vit5_p4_pannuke_hf_classification.ipynb)

<img src="clite_logo.png" alt="C-LITE — A Lightweight Foundation Model for Cellular
feature learning in Histopathology" width="312">

C-LITE learns representations of individual cells from H&E histology. 

## Downstream Tasks

- Immune Cell Phenotyping
- Cell Type Classification
- Mitosis Classification
- Nucleus Classification 
- Hierarchical Subtype Classification


## Quick Start: 

The notebook is self-contained. The ViT-5 architecture is defined inline, so it needs no `timm`, `einops`, `flash-attn` or `xformers`. The data is downloaded directly from the Hugging Face Hub.

```bash
pip install torch torchvision datasets numpy pillow scikit-learn matplotlib tqdm
jupyter lab clite_cell_feature_learning.ipynb
```



