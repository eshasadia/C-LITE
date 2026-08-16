import re
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet
from transformers import AutoImageProcessor, Data2VecVisionModel, ViTModel
from transformers import AutoImageProcessor, AutoModel
import timm
from timm.layers import SwiGLUPacked
import math
import torch
from torch import nn
import sys
import os
try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None

# Loader for the atrous-adapted DINOv3-ConvNeXt-Tiny backbone trained by
# run_pannuke_pipeline.py. Optional import -- only needed if you actually
# load an "atrous_convnext_cell" checkpoint; everything else in this file
# works fine without run_pannuke_pipeline.py being importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from convnext_pannuke import load_dinov3_convnext_atrous
    _atrous_import_error = None
except Exception as e:
    load_dinov3_convnext_atrous = None
    _atrous_import_error = e

    
def _interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))
    patch_pos_embed = nn.functional.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)

def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    tensor_corr_shape = _interpolate_pos_encoding(model.pos_embed, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    return model
# Try to import local modules
try:
    from models.resnet_retccl import resnet50 as retccl_res50
except ImportError:
    retccl_res50 = None

try:
    from models.lemon_loader import get_lemon_vit
except ImportError:
    get_lemon_vit = None

try:
    from dinov2.models.vision_transformer import DinoVisionTransformer
except ImportError:
    DinoVisionTransformer = None



def get_models(modelname, saved_model_path=None, hf_token=None):
    """
    Main function to load various pre-trained models for feature extraction.
    
    Args:
        modelname: Name of the model to load
        saved_model_path: Path to checkpoint (required for some models)
        hf_token: HF token for gated Hugging Face weights (e.g. DINOv3
            ConvNeXt). Falls back to the HF_TOKEN environment variable if
            not passed. Ignored by models that don't need it.
    
    Returns:
        PyTorch model ready for feature extraction
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- histology-pretrained models
    if modelname.lower() == "retccl":
        model = get_retCCL(saved_model_path)
    elif modelname.lower() == "lemon":
        if get_lemon_vit is None:
            raise ImportError("LEMON loader not available. Ensure models/lemon_loader.py exists.")
        model = get_lemon_vit(saved_model_path, model_name="vits8", img_size=40)
    elif modelname.lower() == "owkin":
        model = Phikon()
    elif modelname.lower() == "virchow":
        model = Virchow()
    elif modelname.lower() == "uni2":
        model = UNI2()
    elif modelname.lower() == "cellvitpp":
        model = CellViTPlusPlus(saved_model_path, input_size=1024)
    elif modelname.lower() == "atrous_convnext_cell":
        model = AtrousConvNeXtCell(saved_model_path)
    elif modelname.lower() == "dinov3_convnext":
        model = DINOv3ConvNeXt(hf_token=hf_token)
    # --- vision foundation models
    elif modelname.lower() == "resnet50":
        model = get_res50()
    elif modelname.lower() == "resnet50_full":
        model = get_full_res50()
    elif modelname.lower() == "h0_mini":
        model = H0mini(device)
    elif modelname.lower() == "distilled_vit_small":
        model = get_distilled_vit_small(saved_model_path)
    # --- our finetuned models
    elif modelname.lower() in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_ssl"]:
        model = get_dino_finetuned_downloaded(saved_model_path, modelname)
    elif modelname.lower() == "celldino":
        if DinoVisionTransformer is None:
            raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
        model = get_celldino_finetuned_downloaded(saved_model_path, modelname)
    elif modelname.lower() in ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vitg16"]:
        model = get_dinov3_finetuned_downloaded(saved_model_path, modelname)
    else:
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model


# ============================================================================
# DISTILLED VIT-SMALL LOADER (NEW)
# ============================================================================
def get_distilled_vit_small(model_path):
        """
        Load the distilled ViT-Small/14 model (multi-teacher KD: UNI2, Virchow2,
        H-optimus-0 -> ViT-S/14, from train_knowledgedistill.py).

        Accepts checkpoints in these formats:
        1. {"student": {"key": value, ...}}            -- do_test() eval snapshot,
                                                            RECOMMENDED: backbone only,
                                                            no distillation-head or
                                                            optimizer state to strip.
        2. {"model": {"student_backbone.key": value,    -- full training
                        "distill_head.key": value, ...},     checkpoint_*.pth
            "optimizer": {...}, "iteration": N}             (save_checkpoint()).
                                                            distill_head.* keys are
                                                            discarded; only the
                                                            backbone is loaded.
        3. {"teacher": {"backbone.key": value, ...}}    -- older single-teacher
        4. {"backbone": {"key": value, ...}}               KD experiment formats,
        5. Plain state_dict {"key": value, ...}            kept for backward compat.

        Returns a timm ViT-Small/14 (DINOv2-style) model ready for feature
        extraction (CLS token).
        """
        ARCH = "vit_small_patch14_dinov2.lvd142m"
        IMG_SIZE = 224

        if model_path is None:
            print(f"No checkpoint provided, loading ImageNet-pretrained {ARCH}")
            model = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=IMG_SIZE)
            return model

        pretrained = torch.load(model_path, map_location="cpu", weights_only=False)

        # Extract state_dict from various checkpoint formats
        if isinstance(pretrained, dict):
            if "student" in pretrained:
                # do_test() eval snapshot: {"student": student_backbone.state_dict()}
                # -- already backbone-only, no head/optimizer keys to strip.
                state_dict = pretrained["student"]
                print(f"Loaded from 'student' key, {len(state_dict)} params")

            elif "teacher" in pretrained:
                # Format: {"teacher": {"backbone.cls_token": ..., ...}}
                raw_state_dict = pretrained["teacher"]
                state_dict = {}
                for key, value in raw_state_dict.items():
                    if "dino_head" in key or "ibot_head" in key:
                        continue
                    new_key = key.replace("backbone.", "")
                    state_dict[new_key] = value
                print(f"Loaded from 'teacher' key, {len(state_dict)} params")

            elif "backbone" in pretrained:
                # Format: {"backbone": {"cls_token": ..., ...}, "classifier": ...}
                state_dict = pretrained["backbone"]
                print(f"Loaded from 'backbone' key, {len(state_dict)} params")

            elif "state_dict" in pretrained:
                state_dict = pretrained["state_dict"]
                print(f"Loaded from 'state_dict' key, {len(state_dict)} params")

            elif "model" in pretrained:
                # save_checkpoint() full training checkpoint:
                # {"model": DistillationModel.state_dict(), "optimizer": ..., "iteration": ...}
                # DistillationModel.state_dict() keys are prefixed "student_backbone."
                # and "distill_head." -- keep only the backbone, drop the head, and
                # strip the prefix so keys match the plain timm model.
                raw_state_dict = pretrained["model"]
                state_dict = {}
                for key, value in raw_state_dict.items():
                    if not key.startswith("student_backbone."):
                        continue  # discards every distill_head.* key
                    state_dict[key[len("student_backbone."):]] = value
                print(f"Loaded from 'model' key (student_backbone.* only), {len(state_dict)} params")

            else:
                # Assume it's a plain state_dict
                state_dict = pretrained
                print(f"Loaded as plain state_dict, {len(state_dict)} params")
        else:
            state_dict = pretrained
            print(f"Loaded as plain state_dict, {len(state_dict)} params")

        # Remove any remaining prefixes (module.)
        clean_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "")
            clean_state_dict[new_key] = value

        # Infer embed_dim from checkpoint (sanity check, not architecture selection --
        # ARCH is fixed above to match training exactly, patch14 not patch16)
        if "patch_embed.proj.weight" in clean_state_dict:
            patch_weight = clean_state_dict["patch_embed.proj.weight"]
            embed_dim, in_chans, patch_h, patch_w = patch_weight.shape
            print(f"Checkpoint patch_embed: embed_dim={embed_dim}, patch_size={patch_h}, in_chans={in_chans}")
            if patch_h != 14:
                print(f"[warn] checkpoint patch_size={patch_h} != expected 14 for {ARCH} -- "
                    f"this checkpoint may not match the distilled ViT-S/14 architecture, "
                    f"expect a shape-mismatch RuntimeError below if so")

        # Build timm model matching training exactly
        model = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=IMG_SIZE)

        # Load weights
        msg = model.load_state_dict(clean_state_dict, strict=False)

        print(f"\n==== LOAD REPORT (distilled_vit_small) ====")
        if msg.missing_keys:
            print(f"Missing keys ({len(msg.missing_keys)}):")
            for k in msg.missing_keys:
                print(f"    {k}")
        else:
            print("Missing keys: none")

        if msg.unexpected_keys:
            print(f"Unexpected keys ({len(msg.unexpected_keys)}):")
            for k in msg.unexpected_keys:
                print(f"    {k}")
        else:
            print("Unexpected keys: none")

        return model


def get_retCCL(model_path):
    """Load RetCCL ResNet50 model"""
    if retccl_res50 is None:
        raise ImportError("RetCCL model not available")
    
    model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model

def get_dino_finetuned_downloaded(saved_model_path, modelname="dinov2_vits14"):
    model = get_downloaded_dino_vit_interpolated(modelname)
    pretrained = torch.load(saved_model_path, map_location="cpu")

    # Support multiple checkpoint formats:
    # 1. old EMA-teacher format:  {"teacher": {"backbone.xxx": ...}}
    # 2. new virchow-distill format: {"backbone": {"xxx": ...}}
    #    (optionally with FSDP "_fsdp_wrapped_module." prefix)
    if "teacher" in pretrained:
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in pretrained["teacher"].items()
            if k.startswith("backbone.")
        }
    elif "student" in pretrained:
        state_dict = pretrained["student"]
        # state_dict = {
        #     k.replace("backbone.", ""): v
        #     for k, v in pretrained["student"].items()
        #     if k.startswith("backbone.")
        # }
    elif "backbone" in pretrained:
        state_dict = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in pretrained["backbone"].items()
        }
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'backbone' key. "
            f"Available keys: {list(pretrained.keys())}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    return model

def get_celldino_finetuned_downloaded(saved_model_path, modelname):
    """Load CellDINO model with automatic architecture inference"""
    pretrained = torch.load(saved_model_path, map_location="cpu")

    # Extract state_dict safely
    if "teacher" in pretrained:
        state_dict = pretrained["teacher"]
    elif "student" in pretrained:
        state_dict = pretrained["student"]
    elif "state_dict" in pretrained:
        state_dict = pretrained["state_dict"]
    elif "model" in pretrained:
        state_dict = pretrained["model"]
    else:
        state_dict = pretrained

    # Remove DDP/module prefixes
    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_key = new_key.replace("backbone.", "")
        clean_state_dict[new_key] = value

    # Infer architecture from checkpoint
    patch_weight = clean_state_dict["patch_embed.proj.weight"]
    embed_dim, in_chans, patch_h, patch_w = patch_weight.shape

    assert patch_h == patch_w, f"Non-square patch embedding found: {patch_h}x{patch_w}"
    patch_size = patch_h

    print(f"Inferred from checkpoint:")
    print(f"  embed_dim  = {embed_dim}")
    print(f"  in_chans   = {in_chans}")
    print(f"  patch_size = {patch_size}")

    # Infer model depth/heads from embed_dim
    if embed_dim == 384:
        depth = 12
        num_heads = 6
    elif embed_dim == 768:
        depth = 12
        num_heads = 12
    elif embed_dim == 1024:
        depth = 24
        num_heads = 16
    elif embed_dim == 1536:
        depth = 40
        num_heads = 24
    else:
        raise ValueError(f"Unsupported embed_dim={embed_dim}. Please specify architecture manually.")

    # Infer image size from pos_embed
    img_size = 128  # fallback default

    if "pos_embed" in clean_state_dict:
        num_tokens = clean_state_dict["pos_embed"].shape[1]
        num_patches = num_tokens - 1
        grid_size = int(num_patches ** 0.5)

        if grid_size * grid_size == num_patches:
            img_size = grid_size * patch_size
            print(f"Inferred img_size = {img_size} from pos_embed")
        else:
            print("Could not infer square image size from pos_embed; using fallback img_size=128")

    # Build compatible DinoVisionTransformer
    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
    )

    # Remove classifier/incompatible heads
    keys_to_remove = []
    for k in clean_state_dict.keys():
        if any(x in k for x in ["head", "classifier", "fc", "dino_head", "ibot_head"]):
            keys_to_remove.append(k)

    for k in keys_to_remove:
        clean_state_dict.pop(k, None)

    # Load weights safely
    if "pos_embed" in clean_state_dict:
        if clean_state_dict["pos_embed"].shape != model.pos_embed.shape:
            print(
                f"Skipping pos_embed due to shape mismatch: "
                f"{clean_state_dict['pos_embed'].shape} vs {model.pos_embed.shape}"
            )
            clean_state_dict.pop("pos_embed")

    msg = model.load_state_dict(clean_state_dict, strict=False)
    print("Load result:", msg)

    return model


def get_dinov3_finetuned_downloaded(model_path, modelname):
    """Load DINOv3 model"""
    pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
    processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
    model = AutoModel.from_pretrained(
        pretrained_model_name, 
        device_map="auto"
    )
    return model


def get_res50():
    """Load ResNet50 with early stopping (remove last residual block)"""
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    # Delete last res block as this has been shown to work better
    model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

    return model


def get_full_res50():
    """Load full ResNet50"""
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")
    return model


# ============================================================================
# TRANSFORMS
# ============================================================================

def get_transforms(model_name):
    """
    Get appropriate image transforms for each model.
    """
    # Default ImageNet normalization
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ["ctranspath", "resnet50", "resnet50_full"]:
        size = 224
    elif model_name.lower() == "lemon":
        size = 40
    elif model_name.lower() == "owkin":
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        mean, std = image_processor.image_mean, image_processor.image_std
        size = image_processor.size["height"]
    elif model_name.lower() == "retccl":
        size = 256
    elif model_name.lower() == "kimianet":
        size = 1000
    elif model_name.lower() == "cellvitpp":
        size = 1024
    elif model_name.lower() == "atrous_convnext_cell":
        # Matches run_pannuke_pipeline.py's make_eval_transform() -- trained
        # and evaluated on 64x64 cell-centered crops. Same ImageNet
        # mean/std as the default above, kept explicit here for clarity
        # and so this line is the one place to change if that ever drifts.
        size = 64
    elif model_name.lower() == "dinov3_convnext":
        # Native DINOv3-ConvNeXt pretrain resolution -- this is the
        # vanilla, un-atrous-adapted backbone (see DINOv3ConvNeXt), so it
        # expects standard-size crops, not the 64px cell crops used for
        # atrous_convnext_cell above.
        size = 64
    elif model_name.lower() == "imagebind":
        size = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif model_name.lower() in [
        "dinov2_vits14",
        "dinov2_ssl",
        "dinov3_vits16",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "dinov2_finetuned",
        "dinov2_vits14_interpolated",
        "dinov2_finetuned_downloaded",
        "remedis",
        "vim_finetuned",
        "virchow",
        "uni2",            # <-- ADDED
        "h0_mini",
        "celldino",
        "distilled_vit_small"
    ]:
        size = 224
    else:
        raise ValueError(f"Model name '{model_name}' not found")

    size = (size, size)

    transforms_list = [
        transforms.Resize(size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ]

    preprocess_transforms = transforms.Compose(transforms_list)
    return preprocess_transforms

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
        self.model = timm.create_model("hf-hub:bioptimus/H0-mini",pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,)

    def forward(self, x):
        outputs = self.model(x)
        features = outputs[:, 0]
        return features

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


class CellViTPlusPlus(nn.Module):
    """
    CellViT-SAM-H encoder feature extractor.

    Loads the (CellViT-finetuned) SAM-H ViT-H/16 image encoder from an
    official CellViT-SAM-H-x40-AMP.pth checkpoint and produces one
    embedding per input crop by mean-pooling the encoder's patch tokens.

    CAVEAT: this pools over ALL tokens in the crop. The official CellViT++
    classification pipeline instead pools only the tokens that fall inside
    a cell's segmentation mask within a larger WSI tile. This is a
    reasonable proxy for single, centered cell crops -- but it is an
    approximation, not a literal reproduction of their reported numbers.
    Label results from this accordingly (e.g. "CellViT-SAM-H (encoder
    tokens, approx.)"), not as an exact CellViT++ benchmark figure.

    Requires:
      pip install segment-anything
      CellViT-SAM-H-x40-AMP.pth from https://github.com/TIO-IKIM/CellViT-plus-plus
      passed via --checkpoint
    """
    def __init__(self, checkpoint_path, input_size=1024):
        super().__init__()
        if sam_model_registry is None:
            raise ImportError(
                "segment-anything not installed. Run: pip install segment-anything"
            )
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint is required for cellvitpp: path to CellViT-SAM-H-x40-AMP.pth"
            )

        # Bare SAM-H encoder shell, then overwrite with CellViT-finetuned weights
        sam = sam_model_registry["vit_h"](checkpoint=None)
        self.encoder = sam.image_encoder  # ViT-H/16, native input 1024x1024

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

        # CellViT checkpoint releases have used different key prefixes for
        # the encoder submodule; try the known ones.
        encoder_state = {}
        for key, value in state_dict.items():
            for prefix in ("encoder.", "model.encoder.", "image_encoder."):
                if key.startswith(prefix):
                    encoder_state[key[len(prefix):]] = value
                    break

        if len(encoder_state) == 0:
            raise KeyError(
                "Could not find encoder weights using known prefixes "
                "('encoder.', 'model.encoder.', 'image_encoder.'). "
                f"Top-level checkpoint keys: {list(state_dict.keys())[:10]}. "
                "Inspect the checkpoint structure and adjust the prefix list "
                "in CellViTPlusPlus.__init__."
            )

        msg = self.encoder.load_state_dict(encoder_state, strict=False)
        print(f"[cellvitpp] Loaded SAM-H encoder: "
              f"{len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys")
        if msg.missing_keys:
            print(f"  missing (first 5): {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")

        # Only interpolate pos_embed if deviating from SAM-H's native 1024 input
        if input_size != 1024:
            old_pos_embed = self.encoder.pos_embed.data           # (1, 64, 64, C)
            old_pos_embed = old_pos_embed.permute(0, 3, 1, 2)      # (1, C, 64, 64)
            new_grid = input_size // 16
            new_pos_embed = nn.functional.interpolate(
                old_pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False
            )
            new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
            self.encoder.pos_embed = nn.Parameter(new_pos_embed)

    def forward(self, x):
        feats = self.encoder(x)          # (B, 256, H', W') SAM neck output
        feats = feats.mean(dim=[2, 3])   # mean-pool tokens -> (B, 256)
        return feats


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


class AtrousConvNeXtCell(nn.Module):
    """
    Wrapper around the atrous-adapted DINOv3-ConvNeXt-Tiny backbone trained
    by run_pannuke_pipeline.py (StudentTeacherMultiStage). Loads the EMA
    teacher_backbone weights only -- no DINO/BYOT heads -- since those are
    only needed for pretraining, not for downstream feature extraction.

    Checkpoint is expected to be a run_pannuke_pipeline.py output:
        {"teacher_backbone": <state_dict>, "args": {..., "reduce_stages": [...]}, ...}
    (i.e. exactly what run_training() saves, and what
    load_teacher_backbone() in that script already knows how to read --
    this wrapper mirrors that logic so it's usable from get_models()).

    Requires convnext_pannuke.py to be importable (same directory as this
    file, or on PYTHONPATH) -- that's where load_dinov3_convnext_atrous
    and the atrous-surgery logic actually live; this class does not
    duplicate that logic.
    """
    def __init__(self, checkpoint_path):
        super().__init__()
        if load_dinov3_convnext_atrous is None:
            raise ImportError(
                "Could not import load_dinov3_convnext_atrous from "
                "convnext_pannuke.py -- make sure that file is on the "
                f"Python path (same directory or installed as a module). "
                f"Underlying error: {_atrous_import_error!r}"
            )
        if checkpoint_path is None:
            raise ValueError(
                "--checkpoint is required for atrous_convnext_cell: path to "
                "a run_pannuke_pipeline.py checkpoint_epochN.pth file."
            )

        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if "teacher_backbone" not in ckpt:
            raise KeyError(
                f"Checkpoint at {checkpoint_path} has no 'teacher_backbone' key "
                f"-- this loader only accepts run_pannuke_pipeline.py checkpoints. "
                f"Available keys: {list(ckpt.keys())}"
            )

        reduce_stages = tuple(ckpt.get("args", {}).get("reduce_stages", [0, 1]))
        self.model = load_dinov3_convnext_atrous(
            pretrained=False, reduce_stages=reduce_stages
        )
        missing, unexpected = self.model.load_state_dict(
            ckpt["teacher_backbone"], strict=True
        )
        print(f"[atrous_convnext_cell] Loaded teacher_backbone from {checkpoint_path} "
              f"(epoch {ckpt.get('epoch')}, iteration {ckpt.get('iteration')}, "
              f"reduce_stages={reduce_stages})")

    def forward(self, x):
        # Returns the underlying DINOv3ConvNextModel output, which already
        # exposes .pooler_output -- matching the convention
        # save_features_combined()/save_features_hest() already rely on
        # via `hasattr(batch_features, "pooler_output")`.
        return self.model(x)