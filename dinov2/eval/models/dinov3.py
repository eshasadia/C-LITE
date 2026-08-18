import os
import torch.nn as nn
from transformers import AutoModel


class DINOv3ConvNeXt(nn.Module):
    """
    Vanilla DINOv3-ConvNeXt-Tiny straight from Hugging Face -- no atrous
    stride/dilation surgery, no PanNuke fine-tuning. Useful as a clean
    baseline against AtrousConvNeXtCell to see what the low-res adaptation
    + histology fine-tuning actually buys you.

    Needs the license accepted at
    https://huggingface.co/facebook/dinov3-convnext-tiny-pretrain-lvd1689m
    and either `huggingface-cli login` beforehand, or an HF token passed
    via hf_token / the HF_TOKEN environment variable.
    """
    MODEL_NAME = "facebook/dinov3-convnext-tiny-pretrain-lvd1689m"

    def __init__(self, hf_token=None):
        super().__init__()
        token = hf_token or os.environ.get("HF_TOKEN")
        self.model = AutoModel.from_pretrained(self.MODEL_NAME, token=token)

    def forward(self, x):
        # Has .pooler_output, same convention as the other wrappers in this
        # file -- save_features_combined()/save_features_hest() already
        # check for it via hasattr(batch_features, "pooler_output").
        return self.model(x)