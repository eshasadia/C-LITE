"""
LEMON ViT Model Loader

This module handles loading LEMON (LEarning from cell MOtioN) ViT models
with MoCo-style pre-trained checkpoints.

LEMON uses Vision Transformers (ViT) trained with MoCo v3 on cell images,
typically with 40x40 pixel patches.
"""

import torch
import torch.nn as nn
from pathlib import Path

try:
    from . import vits  # ✅ correct relative import
except ImportError:
    vits = None
    print("Warning: vits module not found. LEMON models will not be available.")


def _clean_moco_state_dict(state_dict: dict, linear_keyword: str) -> dict:
    """
    Filters and renames keys from a MoCo state_dict.
    
    MoCo checkpoints have keys like 'module.base_encoder.xxx'.
    This function:
    1. Keeps only keys from the base_encoder (not momentum encoder)
    2. Removes the linear head (specified by linear_keyword)
    3. Strips the 'module.base_encoder.' prefix
    
    Args:
        state_dict: MoCo checkpoint state dict
        linear_keyword: Name of the head layer to exclude (e.g., 'fc' or 'head')
    
    Returns:
        Cleaned state dict with renamed keys
    """
    cleaned = {}
    for key in list(state_dict.keys()):
        # Check if the key belongs to the base encoder's backbone
        if key.startswith('module.base_encoder') and not key.startswith(f'module.base_encoder.{linear_keyword}'):
            # Create a new key by stripping the prefix
            new_key = key[len("module.base_encoder."):]
            cleaned[new_key] = state_dict[key]
    
    return cleaned


def load_moco_encoder(model: nn.Module, weight_path: str, linear_keyword: str) -> nn.Module:

    weight_path = Path(weight_path)
    assert weight_path.exists(), f"Checkpoint not found at '{weight_path}'"
    print(f"=> Loading MoCo checkpoint from '{weight_path}'")

    checkpoint = torch.load(weight_path, map_location="cpu")

    # Handle different checkpoint formats
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "module" in checkpoint:
        state_dict = checkpoint["module"]
    else:
        # assume it's already a raw state dict
        state_dict = checkpoint


    # Clean the state_dict to match the model's architecture
    cleaned_state_dict = _clean_moco_state_dict(state_dict, linear_keyword)

    # Load the cleaned weights into the model
    msg = model.load_state_dict(cleaned_state_dict, strict=False)
    print("Load result:", msg)
    print("=> Successfully loaded pre-trained model backbone.")

    # Replace the model's head to turn it into a feature extractor
    if hasattr(model, linear_keyword):
        setattr(model, linear_keyword, nn.Identity())
        print(f"=> Model's '{linear_keyword}' layer replaced with nn.Identity for feature extraction.")

    return model


def get_lemon_vit(weight_path: str, model_name: str = "vits8", img_size: int = 40) -> nn.Module:
    """
    Creates a LEMON ViT feature extractor.
    
    This is the main entry point for loading LEMON models. It:
    1. Creates the ViT architecture
    2. Loads MoCo pre-trained weights
    3. Returns a feature extractor (no classification head)
    
    Args:
        weight_path: Path to LEMON checkpoint (.pth.tar file)
        model_name: ViT model variant, one of:
            - 'vits4': ViT-Small with 4x4 patches
            - 'vits8': ViT-Small with 8x8 patches (default, recommended)
            - 'vitb4': ViT-Base with 4x4 patches
            - 'vitb8': ViT-Base with 8x8 patches
        img_size: Input image size in pixels (typically 40 for cell patches)
    
    Returns:
        PyTorch model ready for feature extraction
        - ViT-Small outputs 384-dimensional features
        - ViT-Base outputs 768-dimensional features
    
    Raises:
        ImportError: If vits module is not available
        KeyError: If model_name is not recognized
    
    Example:
        >>> model = get_lemon_vit('/path/to/lemon.pth.tar', 'vits8', 40)
        >>> model.eval()
        >>> features = model(images)  # (batch, 384)
    """
    if vits is None:
        raise ImportError(
            "vits module not found. Please ensure vits.py is in the models directory.\n"
            "The vits.py file should contain VisionTransformerMoCo definitions."
        )
    
    if model_name not in vits.__dict__:
        available = [k for k in vits.__dict__.keys() if k.startswith('vit')]
        raise KeyError(
            f"Model '{model_name}' not found in vits module.\n"
            f"Available models: {available}"
        )
    
    # Create the model architecture shell
    # num_classes=0 means no classification head
    vit_model = vits.__dict__[model_name](img_size=img_size, num_classes=0)

    # Load MoCo weights and prepare for feature extraction
    feature_extractor = load_moco_encoder(
        model=vit_model,
        weight_path=weight_path,
        linear_keyword='head'  # ViT models use 'head' instead of 'fc'
    )
    
    return feature_extractor


def get_lemon_resnet(weight_path: str, arch: str = 'resnet50') -> nn.Module:
    """
    Creates a LEMON ResNet feature extractor (if LEMON has ResNet checkpoints).
    
    This function is provided for completeness but may not be used if LEMON
    only provides ViT checkpoints.
    
    Args:
        weight_path: Path to LEMON checkpoint
        arch: ResNet architecture ('resnet50', 'resnet101', etc.)
    
    Returns:
        ResNet model ready for feature extraction
    """
    from torchvision.models import resnet
    
    # Create ResNet architecture
    if arch == 'resnet50':
        model = resnet.resnet50(pretrained=False)
    elif arch == 'resnet101':
        model = resnet.resnet101(pretrained=False)
    else:
        raise ValueError(f"Unsupported ResNet architecture: {arch}")
    
    # Load MoCo weights
    model = load_moco_encoder(
        model=model,
        weight_path=weight_path,
        linear_keyword='fc'  # ResNet models use 'fc'
    )
    
    return model