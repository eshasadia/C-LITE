import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
from vit5_arch import vit_models, RMSNorm, Block  # see vit5_arch.py
from functools import partial

# --- NEW: detect flash-attn availability once, at import time, instead of
# assuming it's installed. `vit5_arch.py`'s Attention/Block classes should
# themselves also guard their flash_attn_qkvpacked_func call on whatever
# `flash` flag they receive -- this only fixes the caller-side bug where
# `flash=True` was hardcoded regardless of whether the package was actually
# importable. If Attention.forward() unconditionally calls
# flash_attn_qkvpacked_func without checking self.flash, this alone won't
# be enough -- that call site also needs a guard (see note at bottom).
try:
    import flash_attn  # noqa: F401
    _FLASH_ATTN_AVAILABLE = True
except ImportError:
    _FLASH_ATTN_AVAILABLE = False

try:
    import xformers  # noqa: F401
    _XFORMERS_AVAILABLE = True
except ImportError:
    _XFORMERS_AVAILABLE = False

if not _FLASH_ATTN_AVAILABLE:
    # --- CHANGED: this used to unconditionally recommend installing
    # flash-attn whenever it wasn't found, even though vit5_arch.py's
    # Attention class will actually fall back to xFormers first (if
    # available) before ever touching plain SDPA -- xFormers gives
    # comparable performance to flash-attn and, unlike it, ships prebuilt
    # wheels that don't need a local CUDA toolkit to install. The message
    # now reflects what will actually happen, instead of always pointing
    # at the harder-to-install option.
    if _XFORMERS_AVAILABLE:
        print(
            "[ViT-5] flash-attn is not installed/importable, but xFormers "
            "is -- vit5_arch.py's Attention will use xformers.ops."
            "memory_efficient_attention as the fast path (comparable "
            "performance to flash-attn, no local build needed)."
        )
    else:
        print(
            "[ViT-5] Neither flash-attn nor xFormers is importable in this "
            "environment -- falling back to torch.nn.functional."
            "scaled_dot_product_attention (fused eager attention). This "
            "still avoids the old manual softmax-attention path, but is "
            "typically slower than flash-attn/xFormers at long sequence "
            "lengths. Install one of: `pip install flash-attn "
            "--no-build-isolation` (needs CUDA_HOME/nvcc) or "
            "`pip install xformers` (usually installs a prebuilt wheel, "
            "no compiler needed) for full speed."
        )


def _bilinear_resize_matrix(src_size, dst_size, device, dtype, mode="bicubic", antialias=True):
    src = torch.eye(src_size * src_size, device=device, dtype=dtype).reshape(
        src_size * src_size, 1, src_size, src_size)
    resized = nn.functional.interpolate(src, size=(dst_size, dst_size), mode=mode,
                                         align_corners=False, antialias=antialias)
    return resized.reshape(src_size * src_size, dst_size * dst_size).T  # (dst^2, src^2)


def pi_resize_patch_kernel(kernel, target_size):
    out_ch, in_ch, kh, kw = kernel.shape
    assert kh == kw, "PI-resize as implemented here assumes a square kernel"
    if kh == target_size:
        return kernel
    B = _bilinear_resize_matrix(kh, target_size, kernel.device, kernel.dtype)
    P = torch.linalg.pinv(B.T, rcond=1e-4)
    flat = kernel.reshape(out_ch * in_ch, kh * kh)
    resized_flat = flat @ P.T
    return resized_flat.reshape(out_ch, in_ch, target_size, target_size)

def _detect_vit5_gating(state_dict: dict, num_heads: int):
    """Returns (gating_type, gate_bias_init) or (None, None) if the
    checkpoint wasn't trained with Gated Attention.

    state_dict here should be whatever dict you're about to load_state_dict
    from -- i.e. AFTER stripping the 'backbone.'/'student.'/etc prefix
    (the same dict your existing prefix-stripping code already produces,
    the one that gets you to "[ViT-5] stripped prefix 'backbone.' -> N
    tensors" in your logs).
    """
    gate_weight_keys = [k for k in state_dict if k.endswith("attn.gate_proj.weight")]
    if not gate_weight_keys:
        return None, None
    out_dim = state_dict[gate_weight_keys[0]].shape[0]
    gating_type = "headwise" if out_dim == num_heads else "elementwise"
    return gating_type, 4.0  # gate_bias_init doesn't matter once real weights load




class DinoViT5(vit_models):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Set explicitly rather than trusting vit_models to have stored it --
        # _interpolate_pos_encoding below depends on this attribute existing,
        # and we don't have visibility into whether the base class sets it.
        self.patch_size = kwargs.get("patch_size", getattr(self, "patch_size", 16))
        self.mask_token = nn.Parameter(torch.zeros(1, self.embed_dim))
        trunc_normal_(self.mask_token, std=0.02)

    # ---- position-embedding interpolation for arbitrary crop sizes ----
    def _interpolate_pos_encoding(self, npatch, dim, w, h):
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        src_grid = int(math.sqrt(N))
        assert src_grid * src_grid == N, "pretrained pos_embed grid must be square"
        pos_embed = self.pos_embed.reshape(1, src_grid, src_grid, dim).permute(0, 3, 1, 2)
        pos_embed = nn.functional.interpolate(
            pos_embed.float(), size=(h0, w0), mode="bicubic", align_corners=False
        ).to(pos_embed.dtype)
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed

    def prepare_tokens_with_masks(self, x, masks=None):
        # (B, C, H, W) is the real tensor layout -- named to match (previously
        # W and H were swapped here; harmless for square crops, wrong for
        # non-square ones).
        B, _, H, W = x.shape
        x = self.patch_embed(x)  # (B, N, C)

        if masks is not None:
            x = torch.where(
                masks.unsqueeze(-1).to(x.device),
                self.mask_token.to(x.dtype).unsqueeze(0),
                x,
            )
        pos_embed = self._interpolate_pos_encoding(x.shape[1], x.shape[-1], W, H)
        x = x + pos_embed
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        if self.reg_token is not None:
            registers = self.reg_token.expand(B, -1, -1)
            x = torch.cat((x, registers), dim=1)
        return x

    def forward_features(self, x, masks=None):
        x = self.prepare_tokens_with_masks(x, masks)
        for blk in self.blocks:
            x = blk(x)
        x_norm = self.norm(x)
        n_reg = self.num_registers
        if n_reg > 0:
            patch_tokens = x_norm[:, 1:-n_reg]
            reg_tokens = x_norm[:, -n_reg:]
        else:
            patch_tokens = x_norm[:, 1:]
            reg_tokens = x_norm[:, :0]

        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": reg_tokens,
            "x_norm_patchtokens": patch_tokens,
            "x_prenorm": x,
        }

    def forward_features_list(self, x_list, masks_list):
        return [self.forward_features(x, m) for x, m in zip(x_list, masks_list)]

    def forward(self, x, masks=None, is_training=False):
        if isinstance(x, list):
            outputs = self.forward_features_list(x, masks)
        else:
            outputs = self.forward_features(x, masks)

        if is_training:
            return outputs
        if isinstance(x, list):
            return [o["x_norm_clstoken"] for o in outputs]
        return outputs["x_norm_clstoken"]


_VIT5_CONFIGS = {
    "vit5_small": dict(embed_dim=384, depth=12, num_heads=6),
    "vit5_base": dict(embed_dim=768, depth=12, num_heads=12),
    "vit5_large": dict(embed_dim=1024, depth=24, num_heads=16),
}

_VIT5_CHECKPOINT_URLS = {
    "vit5_small": "https://huggingface.co/FengWang3211/ViT-5/resolve/main/vit5_small_patch16_224.pth",
    "vit5_base": "https://huggingface.co/FengWang3211/ViT-5/resolve/main/vit5_base_patch16_224.pth",
    "vit5_large": "https://huggingface.co/FengWang3211/ViT-5/resolve/main/vit5_large_patch16_224.pth",
}

_VIT5_NATIVE_IMG_SIZE = 224     # resolution the released checkpoints' pos_embed was trained at
_VIT5_NATIVE_PATCH_SIZE = 16    # patch size the released checkpoints were trained with

def _normalize_state_dict(sd, verbose=True):
    # unwrap common single-key containers
    for key in ("model", "state_dict"):
        if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
            sd = sd[key]
    # KD training checkpoints (step/student/projector/optimizer/scheduler)
    # save the trainable backbone under 'student', not 'teacher' -- e.g.
    # uni2h_to_vit5_tcga_unsupervised checkpoints, where UNI2-h is the
    # (unsaved, externally-loaded) teacher and ViT-5 is the student being
    # trained. Prefer 'teacher' when present (eval convention for
    # released/EMA weights), otherwise fall back to 'student'.
    if isinstance(sd, dict) and "teacher" in sd and isinstance(sd["teacher"], dict):
        sd = sd["teacher"]
    elif isinstance(sd, dict) and "student" in sd and isinstance(sd["student"], dict):
        if verbose:
            print("[ViT-5] no 'teacher' key found -- unwrapping 'student' "
                  "sub-dict instead (KD training checkpoint).")
        sd = sd["student"]

    # strip branch / ModuleDict / FSDP prefixes, first match wins
    for pref in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
        hits = {
            k[len(pref):].replace("_fsdp_wrapped_module.", ""): v
            for k, v in sd.items()
            if k.startswith(pref)
        }
        if hits:
            if verbose:
                print(f"[ViT-5] stripped prefix '{pref}' -> {len(hits)} tensors")
            return hits

    # already flat (e.g. the official release)
    return sd
# def _normalize_state_dict(sd, verbose=True):
#     # unwrap common single-key containers
#     for key in ("model", "state_dict"):
#         if isinstance(sd, dict) and key in sd and isinstance(sd[key], dict):
#             sd = sd[key]
#     if isinstance(sd, dict) and "teacher" in sd and isinstance(sd["teacher"], dict):
#         sd = sd["teacher"]

#     # strip branch / ModuleDict / FSDP prefixes, first match wins
#     for pref in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
#         hits = {
#             k[len(pref):].replace("_fsdp_wrapped_module.", ""): v
#             for k, v in sd.items()
#             if k.startswith(pref)
#         }
#         if hits:
#             if verbose:
#                 print(f"[ViT-5] stripped prefix '{pref}' -> {len(hits)} tensors")
#             return hits

#     # already flat (e.g. the official release)
#     return sd


def _resize_patch_size_inplace(model, target_patch_size):
    """PI-resizes patch_embed.proj's kernel and rebuilds pos_embed to match
    the new patch size's native token grid. Called AFTER the checkpoint
    (always at _VIT5_NATIVE_PATCH_SIZE) is already loaded, so
    load_state_dict never sees a shape mismatch for a present key."""
    old_conv = model.patch_embed.proj
    old_patch_size = old_conv.kernel_size[0]
    if target_patch_size == old_patch_size:
        return

    print(f"[ViT-5] PI-resizing patch_embed kernel {old_patch_size}x{old_patch_size} "
          f"-> {target_patch_size}x{target_patch_size}")
    new_weight = pi_resize_patch_kernel(old_conv.weight.data, target_patch_size)
    new_conv = nn.Conv2d(
        old_conv.in_channels, old_conv.out_channels,
        kernel_size=target_patch_size, stride=target_patch_size,
        bias=old_conv.bias is not None,
    )
    new_conv.weight.data = new_weight
    if old_conv.bias is not None:
        new_conv.bias.data = old_conv.bias.data.clone()
    model.patch_embed.proj = new_conv
    if hasattr(model.patch_embed, "patch_size"):
        model.patch_embed.patch_size = (target_patch_size, target_patch_size)
    model.patch_size = target_patch_size

    # pos_embed's *base* grid also needs to move to the new patch size's
    # native resolution (_VIT5_NATIVE_IMG_SIZE // target_patch_size) --
    # DinoViT5._interpolate_pos_encoding() only handles *runtime* crop-size
    # changes away from this base; it has no idea the checkpoint's patch
    # size itself just changed, so that base grid has to be fixed up here.
    old_grid = int(math.sqrt(model.pos_embed.shape[1]))
    new_grid = _VIT5_NATIVE_IMG_SIZE // target_patch_size
    pe = model.pos_embed.reshape(1, old_grid, old_grid, -1).permute(0, 3, 1, 2)
    pe = nn.functional.interpolate(pe.float(), size=(new_grid, new_grid),
                                    mode="bicubic", align_corners=False).to(pe.dtype)
    pe = pe.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1)
    model.pos_embed = nn.Parameter(pe)


def get_vit5_backbone(modelname="vit5_small", img_size=224, patch_size=16,
                       pretrained_weights=None, pretrained=True, flash=None):
    """
    flash: whether to use flash-attn inside the backbone's attention blocks.
        - None (default): auto-detect -- True if `flash-attn` is importable
          in this environment, False otherwise. This replaces the old
          hardcoded `flash=True`, which crashed with
          `TypeError: 'NoneType' object is not callable` whenever
          flash-attn wasn't installed, because flash_attn_qkvpacked_func
          would be None but was still being called unconditionally.
        - True: force flash attention. Will raise/crash at forward time if
          flash-attn isn't actually importable -- only pass this if you've
          confirmed `python -c "from flash_attn import flash_attn_qkvpacked_func"`
          succeeds in your env.
        - False: force eager/SDPA attention regardless of what's installed.
    """
    assert modelname in _VIT5_CONFIGS, f"unknown ViT-5 variant: {modelname}"
    cfg = _VIT5_CONFIGS[modelname]

    if flash is None:
        flash = _FLASH_ATTN_AVAILABLE or _XFORMERS_AVAILABLE
    elif flash and not (_FLASH_ATTN_AVAILABLE or _XFORMERS_AVAILABLE):
        print(
            "[ViT-5] WARNING: flash=True was explicitly requested but "
            "neither flash-attn nor xFormers is importable in this "
            "environment. vit5_arch.py's Attention will fall back to "
            "scaled_dot_product_attention automatically (it resolves its "
            "own backend internally), so this will NOT crash -- but it "
            "will run slower than either fast path."
        )

    # NOTE: the actual per-block backend choice (flash / xformers / sdpa)
    # is now resolved inside vit5_arch.py's Attention class itself (see
    # _select_attn_backend there), which is printed once at import time as
    # "[vit5_arch] attention backend auto-selected: ...". The `flash` bool
    # here only controls whether Attention is allowed to use a fast path
    # at all (flash=False forces plain SDPA regardless of what's
    # installed) -- it is NOT itself the name of the backend that gets
    # used, so we no longer print a separate (and easily stale-looking)
    # "attention backend: eager/SDPA"-style message here.
    print(f"[ViT-5] fast attention path requested: {flash} "
          f"(flash-attn available: {_FLASH_ATTN_AVAILABLE}, "
          f"xformers available: {_XFORMERS_AVAILABLE})")

    if pretrained and img_size != _VIT5_NATIVE_IMG_SIZE:
        print(f"[ViT-5] note: building backbone at native {_VIT5_NATIVE_IMG_SIZE}px "
              f"(pos_embed grid) even though crops are {img_size}px; resolution is "
              f"reconciled per-forward-pass via bicubic pos-embed interpolation.")

    # Always construct at the checkpoint's native patch size so the raw
    # state_dict loads with no shape mismatch, then PI-resize afterward if
    # the caller wants a different patch_size (see _resize_patch_size_inplace).
    model = DinoViT5(
        img_size=_VIT5_NATIVE_IMG_SIZE,
        patch_size=_VIT5_NATIVE_PATCH_SIZE,
        embed_dim=cfg["embed_dim"],
        depth=cfg["depth"],
        num_heads=cfg["num_heads"],
        mlp_ratio=4,
        qkv_bias=False,
        num_registers=4,
        flash=flash,
        norm_layer=partial(RMSNorm, eps=1e-6),
        block_layers=Block,
        rope=True,
        reg_theta=100,
        qk_norm=True,
        num_classes=0,  # no classification head needed for SSL
    )
    if hasattr(model.patch_embed, "strict_img_size"):
        model.patch_embed.strict_img_size = False

    if not pretrained:
        if patch_size != _VIT5_NATIVE_PATCH_SIZE:
            _resize_patch_size_inplace(model, patch_size)
        return model, cfg["embed_dim"]

    if pretrained_weights:
        state = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
    else:
        url = _VIT5_CHECKPOINT_URLS[modelname]
        state = torch.hub.load_state_dict_from_url(url, map_location="cpu")

    state_dict = _normalize_state_dict(state)

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[ViT-5] loaded pretrained weights from "
          f"{pretrained_weights or _VIT5_CHECKPOINT_URLS[modelname]}")
    print(f"[ViT-5] missing keys ({len(missing)}): {missing}")
    print(f"[ViT-5] unexpected keys ({len(unexpected)}): {unexpected}")

    if patch_size != _VIT5_NATIVE_PATCH_SIZE:
        _resize_patch_size_inplace(model, patch_size)

    return model, cfg["embed_dim"]


try:
    from torch.nn.attention import SDPBackend, sdpa_kernel
    _SDPA_KERNEL_AVAILABLE = True
except ImportError:
    _SDPA_KERNEL_AVAILABLE = False


class ViT5FeatureExtractor(nn.Module):
    """
    Thin eval-mode wrapper: forward(x) -> pooled (B, embed_dim) CLS-token
    feature, matching what extract_features_unified.py expects (a plain
    tensor, or something with no .pooler_output/tuple wrapping).

    force_sdpa_backend: when flash-attn isn't installed, Attention.forward
        falls back to F.scaled_dot_product_attention, which auto-selects a
        kernel (flash / memory-efficient / math) based on input shapes and
        dtype. That auto-selection can silently pick the slow `math`
        kernel if the fast kernels don't support the current shapes/dtype
        -- there's no error, just worse performance. Setting this True
        forces PyTorch to only consider the fast kernels and raise a clear
        RuntimeError if neither is usable, so a silent slow fallback can't
        hide. Set to False to let PyTorch auto-select (default behavior).
    """
    def __init__(self, backbone, force_sdpa_backend=True):
        super().__init__()
        self.backbone = backbone
        self.force_sdpa_backend = force_sdpa_backend and _SDPA_KERNEL_AVAILABLE
        if force_sdpa_backend and not _SDPA_KERNEL_AVAILABLE:
            print(
                "[ViT-5] torch.nn.attention.sdpa_kernel not available in "
                "this torch version -- skipping explicit backend forcing, "
                "PyTorch will auto-select the SDPA kernel as usual."
            )

    def forward(self, x):
        if self.force_sdpa_backend:
            with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
                return self.backbone(x, is_training=False)
        return self.backbone(x, is_training=False)


def compile_vit5_feature_extractor(feature_extractor, mode="reduce-overhead"):
    """
    Wraps feature_extractor.backbone in torch.compile for reduced per-call
    Python/kernel-launch overhead across the model's sequential blocks.

    IMPORTANT: only call this if your DataLoader uses drop_last=True (or
    your dataset size happens to divide evenly by batch_size). torch.compile
    specializes on input shapes by default -- a differently-shaped final
    partial batch will trigger a recompilation, costing you the time this
    was supposed to save. First batch after compiling will also be slow
    (compilation itself); this is expected and only happens once.

    Call this AFTER load_vit5_teacher_for_eval(), e.g.:
        feature_extractor = load_vit5_teacher_for_eval(...)
        feature_extractor = compile_vit5_feature_extractor(feature_extractor)
    """
    feature_extractor.backbone = torch.compile(feature_extractor.backbone, mode=mode)
    return feature_extractor


def load_vit5_teacher_for_eval(
    checkpoint_path,
    modelname="vit5_small",
    patch_size=4,
    flash=None,
    force_sdpa_backend=True,
):
    """
    NOTE: `patch_size` must match whatever the checkpoint at
    `checkpoint_path` was actually trained with -- this rebuilds the
    backbone at that patch size (via the same native-then-PI-resize path
    as get_vit5_backbone) *before* loading, so the checkpoint's tensor
    shapes line up. Passing the wrong patch_size here will silently drop
    patch_embed.proj / pos_embed into "missing keys" (shape mismatch,
    strict=False) and you'll be evaluating a partially randomly
    initialized backbone with no error raised.

    flash: see get_vit5_backbone -- defaults to None (auto-detect based on
    whether flash-attn is importable in the current environment).
    """
    model, embed_dim = get_vit5_backbone(
        modelname,
        patch_size=patch_size,
        pretrained=False,
        flash=flash,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone_state = _normalize_state_dict(ckpt)

    if not backbone_state:
        raise RuntimeError(
            f"Empty state dict after normalization for {checkpoint_path}. "
            f"Check that this is a teacher checkpoint saved by do_test()."
        )
    if any(k.startswith(("student.", "teacher.", "dino_loss.", "ibot_patch_loss."))
           or k in ("student", "teacher")
           for k in backbone_state):
        raise RuntimeError(
            f"Could not isolate backbone weights in {checkpoint_path}. "
            f"Top-level keys were: {list(backbone_state.keys())[:10]}..."
        )

    missing, unexpected = model.load_state_dict(backbone_state, strict=False)
    print(f"[ViT-5 eval] loaded fine-tuned teacher backbone from {checkpoint_path}")
    print(f"[ViT-5 eval] missing keys ({len(missing)}): {missing}")
    print(f"[ViT-5 eval] unexpected keys ({len(unexpected)}): {unexpected}")
    model.eval()
    return ViT5FeatureExtractor(model, force_sdpa_backend=force_sdpa_backend)
