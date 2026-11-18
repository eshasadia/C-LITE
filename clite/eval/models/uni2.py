import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked


class UNI2(nn.Module):
    """
    MahmoodLab UNI2-h model wrapper.

    UNI2-h is a ViT-H/14 (1536-dim, 24-head, 24-layer) histology foundation
    model with SwiGLU MLP, layer-scale, 8 register tokens, and no
    class-token positional embedding (no_embed_class=True). Loaded via
    timm's hf-hub integration with the exact architecture kwargs the
    checkpoint was trained with (per the model card on Hugging Face).

    Requires a Hugging Face account with access approved for
    "MahmoodLab/UNI2-h", and either `huggingface-cli login` or the
    HF_TOKEN environment variable set before loading.
    """
    def __init__(self):
        super(UNI2, self).__init__()
        timm_kwargs = {
            "img_size": 224,
            "patch_size": 14,
            "depth": 24,
            "num_heads": 24,
            "init_values": 1e-5,
            "embed_dim": 1536,
            "mlp_ratio": 2.66667 * 2,
            "num_classes": 0,
            "no_embed_class": True,
            "mlp_layer": SwiGLUPacked,
            "act_layer": torch.nn.SiLU,
            "reg_tokens": 8,
            "dynamic_img_size": True,
        }
        self.model = timm.create_model(
            "hf-hub:MahmoodLab/UNI2-h",
            pretrained=True,
            **timm_kwargs
        )

    def forward(self, x):
        # CLS token
        features = self.model(x)
        return features