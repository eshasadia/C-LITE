import os
import math
import torch
import torch.nn as nn
from torch import nn  # noqa: F811 (kept for readability parity with original imports)
from torchvision import transforms
from torchvision.models import resnet
from transformers import AutoImageProcessor, AutoModel, ViTModel
import timm
from timm.layers import SwiGLUPacked

try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None

# ----------------------------------------------------------------------------
# Wrapper classes live in their own files -- imported here, never redefined.
# ----------------------------------------------------------------------------
from models.uni2 import UNI2
from models.virchow2 import Virchow
from models.owkin import Phikon, H0mini
from models.dinov3 import DINOv3ConvNeXt
from models.cellvitpp import CellViTPlusPlus
from models.hovernext import HoverNeXt
from models.histoplus import HistoPLUS

try:
    from models.cytoformer import CytoFormer
except ImportError:
    CytoFormer = None

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

try:
    from models.convnext_pannuke import load_dinov3_convnext_atrous
except ImportError:
    load_dinov3_convnext_atrous = None

try:
    # vit5_dino_wrapper.py lives outside models/ -- adjust to your actual
    # layout if this import fails. Kept optional (like the other wrapper
    # imports above) so return_model.py still loads if the ViT-5 wrapper
    # isn't on the path yet.
    from vit5_dino_wrapper import load_vit5_teacher_for_eval
except ImportError:
    load_vit5_teacher_for_eval = None


# ----------------------------------------------------------------------------
# Positional-embedding interpolation helper for torch.hub DINOv2 backbones
# ----------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------
# Single dispatcher -- union of every model name previously split across
# return_model.py and return_model_dinov3.py
# ----------------------------------------------------------------------------
def get_models(modelname, saved_model_path=None, hf_token=None, cellvitpp_input_size=1024,
               hovernext_backbone="convnextv2_tiny", vit5_patch_size=16):
    """
    Main function to load various pre-trained models for feature extraction.

    Args:
        modelname: Name of the model to load
        saved_model_path: Path to checkpoint (required for some models)
        hf_token: Optional Hugging Face token, used by gated models
                  (dinov3_convnext). Falls back to HF_TOKEN env var if omitted.
        cellvitpp_input_size: Input resolution for cellvitpp only. SAM-H's
                  native resolution is 1024, but that means a small crop
                  (e.g. 64x64) gets massively upsampled before hitting a
                  632M-param ViT-H encoder -- extremely expensive relative
                  to the crop's actual information content, and the main
                  cause of slow throughput on small-crop datasets. Lower
                  this (e.g. 128 or 256) to trade fidelity to SAM-H's
                  original resolution for a large speed/memory win. Must
                  match the `size` used by get_transforms() below.
        vit5_patch_size: Patch size for vit5_* models only. MUST match
                  whatever the checkpoint at saved_model_path was actually
                  trained with -- getting this wrong does not raise an
                  error, it silently drops patch_embed/pos_embed into
                  "missing keys" and you end up evaluating a partially
                  randomly initialized backbone. See
                  vit5_dino_wrapper.load_vit5_teacher_for_eval's docstring.
                  Also make sure get_transforms() is called with a matching
                  vit5_patch_size so the crop size stays a valid multiple
                  of it.

    Returns:
        PyTorch model ready for feature extraction
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    name = modelname.lower()

    # --- histology-pretrained models
    if name == "retccl":
        model = get_retCCL(saved_model_path)
    elif name == "lemon":
        if get_lemon_vit is None:
            raise ImportError("LEMON loader not available. Ensure models/lemon_loader.py exists.")
        model = get_lemon_vit(saved_model_path, model_name="vits8", img_size=40)
    elif name == "owkin":
        model = Phikon()
    elif name == "virchow":
        model = Virchow()
    elif name == "uni2":
        model = UNI2()
    elif name == "cellvitpp":
        print(f"[get_models] cellvitpp input_size={cellvitpp_input_size} "
              f"-> token grid {cellvitpp_input_size // 16}x{cellvitpp_input_size // 16} "
              f"= {(cellvitpp_input_size // 16) ** 2} tokens")
        model = CellViTPlusPlus(saved_model_path, input_size=cellvitpp_input_size)
    elif name == "dinov3_convnext":
        model = DINOv3ConvNeXt(hf_token=hf_token)
    elif name == "hovernext":
        model = HoverNeXt(saved_model_path, backbone_name=hovernext_backbone)
    elif name == "histoplus":
        model = HistoPLUS(saved_model_path)
    elif name == "cytoformer":
        if CytoFormer is None:
            raise ImportError(
                "CytoFormer wrapper not available. Ensure models/cytoformer.py exists "
                "and the `cytoformer` package (github.com/zhihuanglab/CytoFormer) is "
                "installed -- pip install git+https://github.com/zhihuanglab/CytoFormer"
            )
        model = CytoFormer(saved_model_path)
    elif name == "atrous_convnext_cell":
        raise NotImplementedError(
            "atrous_convnext_cell: no AtrousConvNeXtCell class exists yet. "
            "See models/convnext_pannuke.py:load_dinov3_convnext_atrous for the "
            "underlying loader -- wire it up here once its usage is confirmed."
        )
    # --- vision foundation models
    elif name == "resnet50":
        model = get_res50()
    elif name == "resnet50_full":
        model = get_full_res50()
    elif name == "h0_mini":
        model = H0mini()
    elif name == "distilled_vit_small":
        model = get_distilled_vit_small(saved_model_path)
    # --- our finetuned models
    elif name in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_ssl"]:
        # model = get_dino_finetuned_downloaded(saved_model_path, modelname)
        if saved_model_path is None:
            print(f"[get_models] No checkpoint given for {modelname} -- "
                  f"loading plain torch.hub DINOv2 weights (no fine-tuning).")
            model = get_downloaded_dino_vit_interpolated(modelname)
        else:
            model = get_dino_finetuned_downloaded(saved_model_path, modelname)
    elif name == "dinov2_ssl":
        if saved_model_path is None:
            raise ValueError(
                "--checkpoint is required for dinov2_ssl -- there's no official "
                "'dinov2_ssl' pretrained variant on torch.hub."
            )
        model = get_dino_finetuned_downloaded(saved_model_path, modelname)
    elif name == "celldino":
        if DinoVisionTransformer is None:
            raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
        model = get_celldino_finetuned_downloaded(saved_model_path, modelname)
    elif name == "vit_small_patch7":
        if DinoVisionTransformer is None:
            raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
        model = get_vit_small_patch7_finetuned(saved_model_path, modelname)
    elif name == "dinov2_p4":
        # ImmunoDINO continued-pretraining checkpoint from
        # train_v1_balancedsmapling.py (dinov2_p4 config): dinov2_vits14
        # base architecture, patch_embed PI-resized 14x14 -> 4x4,
        # global_crops_size=64, 4 register tokens. See get_dino_p4_finetuned
        # docstring for accepted checkpoint formats.
        model = get_dino_p4_finetuned(saved_model_path, "dinov2_vits14")
    elif name in ["microformer_small", "microformer_tiny", "microformer_base"]:
        model = get_microformer_finetuned(saved_model_path, modelname)
    elif name in ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vitg16"]:
        model = get_dinov3_finetuned_downloaded(saved_model_path, modelname)
    elif name in ["vit5_small", "vit5_base", "vit5_large"]:
        # ViT-5 DINO/iBOT teacher backbone. Returns an eval-mode wrapper
        # (ViT5FeatureExtractor) whose forward(x) already returns the
        # pooled CLS-token feature -- unlike the branches above, this is
        # NOT a bare backbone, so downstream code that expects
        # .pooler_output / tuple outputs will just get a plain tensor,
        # which feature_extraction_vit5.py already handles.
        if load_vit5_teacher_for_eval is None:
            raise ImportError(
                "vit5_dino_wrapper.load_vit5_teacher_for_eval could not be "
                "imported -- check that vit5_dino_wrapper.py is on sys.path."
            )
        if saved_model_path is None:
            raise ValueError(
                "--checkpoint is required for vit5_* models -- pass the path "
                "to your fine-tuned DINO/iBOT teacher checkpoint, not the "
                "ImageNet-pretrained weights."
            )
        model = load_vit5_teacher_for_eval(
            saved_model_path,
            modelname=name,
            patch_size=vit5_patch_size,
        )
        # load_vit5_teacher_for_eval already returns an eval()-mode wrapper;
        # skip the generic model.to(device); model.eval() below since the
        # wrapper's internal backbone handles device placement the same way.
        model = model.to(device)
        model.eval()
        return model
    else:
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model


# ----------------------------------------------------------------------------
# Individual loaders
# ----------------------------------------------------------------------------
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

    if "teacher" in pretrained:
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in pretrained["teacher"].items()
            if k.startswith("backbone.")
        }
    elif "student" in pretrained:
        state_dict = pretrained["student"]
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
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    model.eval()
    return model


def get_dino_p4_finetuned(saved_model_path, modelname="dinov2_vits14"):
    """
    Load the from-scratch-continued ImmunoDINO backbone trained by
    train_v1_balancedsmapling.py (the dinov2_p4 config): dinov2_vits14 base
    architecture, PI-resized patch_embed kernel 14x14 -> 4x4, pos_embed
    interpolated to (64/4)=16 -> 16x16=256 patches, 4 register tokens,
    global_crops_size=64. Mirrors the exact PI-resize + pos_embed routine
    used at train time in dinov2/train/ssl_meta_arch_v0.py's
    get_downloaded_dino_vit_interpolated -- NOT the generic
    get_downloaded_dino_vit_interpolated() above in this file, which
    hardcodes a patch14 -> 16x16 pos_embed interpolation with no kernel
    resize and is wrong for this checkpoint.

    Accepts the checkpoint formats produced by FSDPCheckpointer / do_test()
    in that training script:
      {"teacher": {"backbone.xxx": ..., "dino_head.xxx": ...}}   -- do_test() snapshot
      {"model": {"student.backbone.xxx": ...}}                   -- full FSDP checkpoint
      {"student": {...}}                                         -- student-only snapshot
      {"backbone": {...}}                                        -- backbone-only snapshot

    GRAM (`gram_teacher.*`) and contrastive-head keys, if present in the
    checkpoint, are ignored -- only the plain backbone is loaded here.

    NOTE: register tokens. Training used num_register_tokens=4. The stock
    torch.hub "dinov2_vits14" architecture loaded below has
    num_register_tokens=0 by default. Check the printed [warn] Unexpected
    keys list the first time you run this -- if register-token parameters
    show up there, they are being silently dropped and this loader needs
    to build the backbone via dinov2's build_model_from_cfg (like
    get_vit_small_patch7_finetuned does) with num_register_tokens=4 set
    explicitly, instead of torch.hub.load, so the architecture actually
    has slots for them.
    """
    if saved_model_path is None:
        raise ValueError("--checkpoint is required for dinov2_p4")

    PATCH_SIZE = 4
    IMG_SIZE = 64

    model = torch.hub.load("facebookresearch/dinov2", modelname)

    # PI-resize patch_embed kernel 14x14 -> 4x4 (same routine used at train time)
    old_conv = model.patch_embed.proj
    old_patch_size = old_conv.kernel_size[0]
    if PATCH_SIZE != old_patch_size:
        src = torch.eye(old_patch_size * old_patch_size).reshape(
            old_patch_size * old_patch_size, 1, old_patch_size, old_patch_size)
        resized = nn.functional.interpolate(
            src, size=(PATCH_SIZE, PATCH_SIZE), mode="bicubic",
            align_corners=False, antialias=True,
        )
        B = resized.reshape(old_patch_size * old_patch_size, PATCH_SIZE * PATCH_SIZE).T
        P = torch.linalg.pinv(B.T, rcond=1e-4)
        flat = old_conv.weight.data.reshape(
            old_conv.out_channels * old_conv.in_channels, old_patch_size * old_patch_size)
        new_weight = (flat @ P.T).reshape(
            old_conv.out_channels, old_conv.in_channels, PATCH_SIZE, PATCH_SIZE)

        new_conv = nn.Conv2d(
            old_conv.in_channels, old_conv.out_channels,
            kernel_size=PATCH_SIZE, stride=PATCH_SIZE,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        model.patch_embed.proj = new_conv
        if hasattr(model.patch_embed, "patch_size"):
            model.patch_embed.patch_size = (PATCH_SIZE, PATCH_SIZE)
        if hasattr(model.patch_embed, "img_size"):
            model.patch_embed.img_size = (IMG_SIZE, IMG_SIZE)
        if hasattr(model, "patch_size"):
            model.patch_size = PATCH_SIZE

    grid_side = IMG_SIZE // PATCH_SIZE  # 16
    tensor_corr_shape = _interpolate_pos_encoding(model.pos_embed, grid_side, grid_side)
    pos_embed = nn.Parameter(torch.zeros_like(tensor_corr_shape))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed

    pretrained = torch.load(saved_model_path, map_location="cpu", weights_only=False)

    if "teacher" in pretrained:
        raw_sd = pretrained["teacher"]
        prefix_strip = ("backbone.",)
    elif "model" in pretrained:
        raw_sd = pretrained["model"]
        prefix_strip = ("student.backbone.", "teacher.backbone.")
    elif "student" in pretrained:
        raw_sd = pretrained["student"]
        prefix_strip = ("backbone.",)
    elif "backbone" in pretrained:
        raw_sd = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in pretrained["backbone"].items()}
        prefix_strip = ()
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has none of 'teacher'/'model'/'student'/'backbone' keys. "
            f"Available keys: {list(pretrained.keys())}"
        )

    state_dict = {}
    for k, v in raw_sd.items():
        if "dino_head" in k or "ibot_head" in k or "gram_teacher" in k or "contrastive_head" in k:
            continue
        matched = False
        for p in prefix_strip:
            if k.startswith(p):
                state_dict[k[len(p):]] = v
                matched = True
                break
        if not matched and not prefix_strip:
            state_dict[k] = v
        elif not matched and prefix_strip:
            continue  # skip keys under a different top-level prefix (e.g. dino_head only under "student.")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("\n==== LOAD REPORT (dinov2_p4) ====")
    print(f"Loaded {len(state_dict) - len(missing)}/{len(state_dict)} tensors")
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")
    if len(state_dict) == 0:
        print("[warn] 0 tensors found in checkpoint -- check prefix stripping above.")

    model.eval()
    return model


def get_distilled_vit_small(model_path):
    """
    Load the distilled ViT-Small/14 model (multi-teacher KD: UNI2, Virchow2,
    H-optimus-0 -> ViT-S/14, from train_knowledgedistill.py).

    Accepts checkpoints in these formats:
    1. {"student": {"key": value, ...}}              -- do_test() eval snapshot,
                                                          backbone only.
    2. {"model": {"student_backbone.key": value,      -- full training
                    "distill_head.key": value, ...},       checkpoint_*.pth
        "optimizer": {...}, "iteration": N}                (save_checkpoint()).
                                                          distill_head.* keys
                                                          discarded.
    3. {"teacher": {"backbone.key": value, ...}}      -- older single-teacher
    4. {"backbone": {"key": value, ...}}                 KD experiment formats
    5. Plain state_dict {"key": value, ...}              kept for backward compat.
    """
    ARCH = "vit_small_patch14_dinov2.lvd142m"
    IMG_SIZE = 224

    if model_path is None:
        print(f"No checkpoint provided, loading ImageNet-pretrained {ARCH}")
        return timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=IMG_SIZE)

    pretrained = torch.load(model_path, map_location="cpu", weights_only=False)

    if isinstance(pretrained, dict):
        if "student" in pretrained:
            state_dict = pretrained["student"]
        elif "teacher" in pretrained:
            raw_state_dict = pretrained["teacher"]
            state_dict = {
                key.replace("backbone.", ""): value
                for key, value in raw_state_dict.items()
                if "dino_head" not in key and "ibot_head" not in key
            }
        elif "backbone" in pretrained:
            state_dict = pretrained["backbone"]
        elif "state_dict" in pretrained:
            state_dict = pretrained["state_dict"]
        elif "model" in pretrained:
            raw_state_dict = pretrained["model"]
            state_dict = {
                key[len("student_backbone."):]: value
                for key, value in raw_state_dict.items()
                if key.startswith("student_backbone.")
            }
        else:
            state_dict = pretrained
    else:
        state_dict = pretrained

    clean_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    if "patch_embed.proj.weight" in clean_state_dict:
        patch_weight = clean_state_dict["patch_embed.proj.weight"]
        embed_dim, in_chans, patch_h, patch_w = patch_weight.shape
        print(f"Checkpoint patch_embed: embed_dim={embed_dim}, patch_size={patch_h}, in_chans={in_chans}")
        if patch_h != 14:
            print(f"[warn] checkpoint patch_size={patch_h} != expected 14 for {ARCH}")

    model = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=IMG_SIZE)
    msg = model.load_state_dict(clean_state_dict, strict=False)

    print("\n==== LOAD REPORT (distilled_vit_small) ====")
    print(f"Missing keys: {msg.missing_keys if msg.missing_keys else 'none'}")
    print(f"Unexpected keys: {msg.unexpected_keys if msg.unexpected_keys else 'none'}")

    return model


def get_vit_small_patch7_finetuned(saved_model_path, modelname="vit_small_patch7"):
    """
    Load the from-scratch ViT-Small/patch7 ImmunoDINO backbone.

    Built via build_model_from_cfg (arch="vit_small", patch_size=7,
    global_crops_size=63), NOT via the torch.hub patch14 pretrained path.
    Do not route through get_downloaded_dino_vit_interpolated.
    """
    from omegaconf import OmegaConf
    from dinov2.models import build_model_from_cfg

    if saved_model_path is None:
        raise ValueError("--checkpoint is required for vit_small_patch7")

    cfg = OmegaConf.create({
        "student": {
            "arch": "vit_small",
            "patch_size": 7,
            "layerscale": 1.0e-05,
            "drop_path_rate": 0.4,
            "drop_path_uniform": True,
            "ffn_layer": "mlp",
            "block_chunks": 0,
            "qkv_bias": False,
            "proj_bias": True,
            "ffn_bias": True,
            "num_register_tokens": 4,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
            "gating_type": "headwise",
            "gate_bias_init": 4.0,
            "window_size": 64,
        },
        "crops": {"global_crops_size": 63},
    })

    _, teacher_backbone, _ = build_model_from_cfg(cfg, only_teacher=False)
    model = teacher_backbone

    pretrained = torch.load(saved_model_path, map_location="cpu")

    if "teacher" in pretrained:
        raw_sd = pretrained["teacher"]
    elif "student" in pretrained:
        raw_sd = pretrained["student"]
    elif "backbone" in pretrained:
        raw_sd = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in pretrained["backbone"].items()}
    elif isinstance(pretrained, dict):
        raw_sd = pretrained
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'student' "
            f"nor 'backbone' key. Available keys: {list(pretrained.keys())}"
        )

    state_dict = {}
    for k, v in raw_sd.items():
        if "dino_head" in k or "ibot_head" in k:
            continue
        if k.startswith("teacher.backbone."):
            state_dict[k[len("teacher.backbone."):]] = v
        elif k.startswith("student.backbone."):
            state_dict[k[len("student.backbone."):]] = v
        elif k.startswith("backbone."):
            state_dict[k[len("backbone."):]] = v
        else:
            state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if len(state_dict) == 0:
        print("[warn] 0 tensors found in checkpoint -- check checkpoint format / prefixes above.")
    elif len(missing) == len(list(model.state_dict().keys())):
        print("[warn] 0 tensors were actually loaded (all keys mismatched).")
    else:
        print(f"[ok] Loaded {len(state_dict) - len(missing)}/{len(state_dict)} backbone tensors.")

    model.eval()
    return model


def get_microformer_finetuned(saved_model_path, modelname):
    raise NotImplementedError(
        f"get_microformer_finetuned for '{modelname}' was referenced in get_models() "
        "but never implemented in the original return_model.py -- fill this in with "
        "the actual microformer construction + checkpoint loading logic."
    )


def get_celldino_finetuned_downloaded(saved_model_path, modelname):
    """Load CellDINO model with automatic architecture inference"""
    pretrained = torch.load(saved_model_path, map_location="cpu")

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

    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "").replace("backbone.", "")
        clean_state_dict[new_key] = value

    patch_weight = clean_state_dict["patch_embed.proj.weight"]
    embed_dim, in_chans, patch_h, patch_w = patch_weight.shape
    assert patch_h == patch_w, f"Non-square patch embedding found: {patch_h}x{patch_w}"
    patch_size = patch_h

    print(f"Inferred from checkpoint: embed_dim={embed_dim}, in_chans={in_chans}, patch_size={patch_size}")

    if embed_dim == 384:
        depth, num_heads = 12, 6
    elif embed_dim == 768:
        depth, num_heads = 12, 12
    elif embed_dim == 1024:
        depth, num_heads = 24, 16
    elif embed_dim == 1536:
        depth, num_heads = 40, 24
    else:
        raise ValueError(f"Unsupported embed_dim={embed_dim}. Please specify architecture manually.")

    img_size = 128
    if "pos_embed" in clean_state_dict:
        num_tokens = clean_state_dict["pos_embed"].shape[1]
        num_patches = num_tokens - 1
        grid_size = int(num_patches ** 0.5)
        if grid_size * grid_size == num_patches:
            img_size = grid_size * patch_size
            print(f"Inferred img_size = {img_size} from pos_embed")
        else:
            print("Could not infer square image size from pos_embed; using fallback img_size=128")

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

    keys_to_remove = [
        k for k in clean_state_dict.keys()
        if any(x in k for x in ["head", "classifier", "fc", "dino_head", "ibot_head"])
    ]
    for k in keys_to_remove:
        clean_state_dict.pop(k, None)

    if "pos_embed" in clean_state_dict and clean_state_dict["pos_embed"].shape != model.pos_embed.shape:
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
    AutoImageProcessor.from_pretrained(pretrained_model_name)  # validates access; processor unused downstream
    model = AutoModel.from_pretrained(pretrained_model_name, device_map="auto")
    return model


def get_res50():
    """Load ResNet50 with early stopping (remove last residual block)"""
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())
    return model


def get_full_res50():
    """Load full ResNet50"""
    return resnet.resnet50(weights="ResNet50_Weights.DEFAULT")


# ----------------------------------------------------------------------------
# Transforms -- union of both files' branches
# ----------------------------------------------------------------------------
def get_transforms(model_name, cellvitpp_input_size=1024, hovernext_input_size=64,
                    vit5_crop_size=224, vit5_patch_size=16):
    """
    Get appropriate image transforms for each model.

    cellvitpp_input_size: must match the value passed to get_models() for
    cellvitpp, since the transform's resize target and the model's
    pos_embed interpolation target have to agree. See get_models()
    docstring for why you'd want to lower this from SAM-H's native 1024.

    hovernext_input_size: crop size fed to HoVer-NeXt's ConvNeXt-V2
    encoder. Unlike cellvitpp, there's no architectural constraint forcing
    a specific value -- default 64 assumes you're running at your actual
    crop resolution with no upsampling.

    vit5_crop_size / vit5_patch_size: for vit5_* models only. crop_size
    must be a multiple of patch_size (raises ValueError otherwise) and
    patch_size must match get_models(..., vit5_patch_size=...) for the
    same checkpoint -- see that docstring.
    """
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    name = model_name.lower()

    if name in ["ctranspath", "resnet50", "resnet50_full"]:
        size = 224
    elif name == "lemon":
        size = 40
    elif name == "owkin":
        image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
        mean, std = image_processor.image_mean, image_processor.image_std
        size = image_processor.size["height"]
    elif name == "retccl":
        size = 256
    elif name == "kimianet":
        size = 1000
    elif name == "cellvitpp":
        size = cellvitpp_input_size
    elif name == "hovernext":
        # No fixed native resolution requirement (no absolute pos_embed in
        # ConvNeXt) -- default to running at your actual crop size. Override
        # by passing hovernext_input_size explicitly if you want to test a
        # larger context window instead.
        size = hovernext_input_size
    elif name == "vit_small_patch7":
        # Must match training: global_crops_size=63, patch_size=7 -> 9x9=81 tokens.
        size = 63
    elif name == "dinov2_p4":
        # Must match training: global_crops_size=64, patch_size=4 -> 16x16=256 tokens.
        # NOTE: training used data_transform="hema" (DataAugmentationHEMA),
        # which may apply stain-specific normalization beyond plain
        # ImageNet mean/std. If DataAugmentationHEMA has a deterministic
        # eval-mode transform, mirror its normalization stats here instead
        # of assuming the generic ImageNet mean/std below are correct.
        size = 64
    elif name in ["microformer_small", "microformer_tiny", "microformer_base"]:
        size = 64
    elif name in ["atrous_convnext_cell", "dinov3_convnext"]:
        size = 64
    elif name == "imagebind":
        size = 224
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    elif name in ["vit5_small", "vit5_base", "vit5_large"]:
        if vit5_crop_size % vit5_patch_size != 0:
            raise ValueError(
                f"vit5_crop_size must be a multiple of vit5_patch_size "
                f"({vit5_patch_size}), got crop_size={vit5_crop_size}"
            )
        size = vit5_crop_size
    elif name in [
        "dinov2_vits14", "dinov2_ssl", "dinov3_vits16", "dinov2_vitb14",
        "dinov2_vitl14", "dinov2_vitg14", "dinov2_finetuned",
        "dinov2_vits14_interpolated", "dinov2_finetuned_downloaded",
        "remedis", "vim_finetuned", "virchow", "uni2", "h0_mini",
        "celldino", "distilled_vit_small", "histoplus", "cytoformer",
    ]:
        size = 224
    else:
        raise ValueError(f"Model name '{model_name}' not found")

    size = (size, size)
    transforms_list = [
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    return transforms.Compose(transforms_list)