import torch
import torch.nn as nn
import timm
from timm.layers import SwiGLUPacked


class Virchow(nn.Module):
    """Virchow2 model wrapper"""
    def __init__(self):
        super(Virchow, self).__init__()
        self.model = timm.create_model(
            "hf-hub:paige-ai/Virchow2",
            pretrained=True,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU
        )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs[:, 0]
        return features