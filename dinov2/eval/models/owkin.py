import torch
import torch.nn as nn
import timm
from transformers import ViTModel


class Phikon(nn.Module):
    """Owkin Phikon model wrapper"""
    def __init__(self):
        super(Phikon, self).__init__()
        self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs.last_hidden_state[:, 0, :]
        return features


class H0mini(nn.Module):
    """h0 mini model wrapper"""
    def __init__(self):
        super(H0mini, self).__init__()
        self.model = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=True,
            mlp_layer=timm.layers.SwiGLUPacked,
            act_layer=torch.nn.SiLU,
        )

    def forward(self, x):
        outputs = self.model(x)
        features = outputs[:, 0]
        return features