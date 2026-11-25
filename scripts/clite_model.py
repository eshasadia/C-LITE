"""C-LiTE encoder for inference (ViT-5-S, 4x4 patches, 64x64 px cell crops).

The architecture and loading code are copied from the TIAToolbox example notebook
(notebooks/14_clite_cell_feature_learning.ipynb) so the viewer scripts need no timm/flash-attn.

    from clite_model import load_clite
    model = load_clite("c-lite.pth", device)        # frozen, eval mode
    feats = model.forward_features(x)["cls"]         # x: (B,3,64,64) normalised, -> (B,384)
"""
from __future__ import annotations

import logging
import math

import torch
import torch.nn.functional as F  # noqa: N812
from torch import nn

logger = logging.getLogger("clite")

ROPE_PT_SEQ_LEN = 224 // 16
REG_THETA = 100
IMG_SIZE = 64
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


class RMSNorm(nn.Module):
    """Root-mean-square layer normalisation."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        """Initialise the learnable scale."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.variance_epsilon = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise the last dimension."""
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
        return self.weight * x.to(dt)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Map (x0, x1, x2, x3, ...) to (-x1, x0, -x3, x2, ...)."""
    x1, x2 = x.unflatten(-1, (-1, 2)).unbind(-1)
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def rope_freqs(dim: int, theta: float) -> torch.Tensor:
    """Frequencies of the rotary embedding."""
    return 1.0 / (theta ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))


class VisionRotaryEmbedding(nn.Module):
    """2D rotary position embedding for a square grid of tokens."""

    def __init__(self, dim: int, pt_seq_len: int, theta: float = 10000) -> None:
        """Initialise the frequencies and an empty cache."""
        super().__init__()
        self.pt_seq_len = pt_seq_len
        self.register_buffer("freqs", rope_freqs(dim, theta))
        self._cache = {}

    def _cos_sin(
        self, n: int, device: torch.device, dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (n, device, dtype)
        if key not in self._cache:
            t = torch.arange(n, device=device) / n * self.pt_seq_len
            f = torch.outer(t, self.freqs.to(device)).repeat_interleave(2, dim=-1)
            f = torch.cat(
                [f[:, None, :].expand(n, n, -1), f[None, :, :].expand(n, n, -1)],
                dim=-1,
            )
            f = f.reshape(n * n, 1, -1)  # (n*n, 1, head_dim)
            self._cache[key] = (f.cos().to(dtype), f.sin().to(dtype))
        return self._cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate x of shape (B, n*n, heads, head_dim)."""
        n = math.isqrt(x.shape[1])
        cos, sin = self._cos_sin(n, x.device, x.dtype)
        return x * cos + rotate_half(x) * sin


class Attention(nn.Module):
    """Multi-head self-attention with QK-norm, RoPE and optional gating."""

    def __init__(
        self,
        dim: int,
        num_heads: int,
        qkv_bias: bool,  # noqa: FBT001
        rope_pt_seq_len: int,
        rope_reg_size: int,
        num_registers: int,
        reg_theta: float,
        use_rope: bool = True,  # noqa: FBT001, FBT002
        gating_type: str | None = None,
    ) -> None:
        """Initialise the projections, norms and rotary embeddings."""
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.num_registers = num_registers
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.rope = (
            VisionRotaryEmbedding(head_dim // 2, rope_pt_seq_len) if use_rope else None
        )
        self.rope_reg = (
            VisionRotaryEmbedding(head_dim // 2, rope_reg_size, theta=reg_theta)
            if rope_reg_size > 0
            else None
        )
        self.gating_type = gating_type
        if gating_type is not None:
            self.gate_proj = nn.Linear(
                dim, num_heads if gating_type == "headwise" else dim
            )
        self.store_attn, self.last_attn = False, None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to the token sequence."""
        x_in = x
        b, n, c = x.shape
        r = n - self.num_registers
        q, k, v = (
            self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).unbind(2)
        )
        dt = q.dtype
        q, k = self.q_norm(q).to(dt), self.k_norm(k).to(dt)
        if self.rope is not None:  # CLS token is not rotated
            q = torch.cat((q[:, :1], self.rope(q[:, 1:r]), q[:, r:]), dim=1)
            k = torch.cat((k[:, :1], self.rope(k[:, 1:r]), k[:, r:]), dim=1)
        if self.rope_reg is not None:
            q = torch.cat((q[:, :r], self.rope_reg(q[:, r:])), dim=1)
            k = torch.cat((k[:, :r], self.rope_reg(k[:, r:])), dim=1)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if self.store_attn:
            attn = ((q.float() @ k.float().transpose(-2, -1)) * self.scale).softmax(-1)
            self.last_attn = attn.detach()
            x = attn.to(v.dtype) @ v
        else:
            x = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        x = x.transpose(1, 2).reshape(b, n, c)
        if self.gating_type is not None:  # gated attention
            g = torch.sigmoid(self.gate_proj(x_in))
            if self.gating_type == "headwise":
                g = (
                    g.unsqueeze(-1)
                    .expand(b, n, self.num_heads, c // self.num_heads)
                    .reshape(b, n, c)
                )
            x = x * g
        return self.proj(x)


class Mlp(nn.Module):
    """Two-layer feed-forward network."""

    def __init__(self, dim: int, hidden: int) -> None:
        """Initialise the two linear layers."""
        super().__init__()
        self.fc1, self.act, self.fc2 = (
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the MLP."""
        return self.fc2(self.act(self.fc1(x)))


class Block(nn.Module):
    """Transformer block with RMSNorm and LayerScale."""

    def __init__(self, dim: int, mlp_hidden: int, **attn_kw: dict) -> None:
        """Initialise attention, MLP and LayerScale parameters."""
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, **attn_kw)
        self.norm2 = RMSNorm(dim)
        self.mlp = Mlp(dim, mlp_hidden)
        self.gamma_1 = nn.Parameter(1e-4 * torch.ones(dim))
        self.gamma_2 = nn.Parameter(1e-4 * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the block."""
        x = x + self.gamma_1 * self.attn(self.norm1(x))
        return x + self.gamma_2 * self.mlp(self.norm2(x))


class PatchEmbed(nn.Module):
    """Split the image into patches and project each patch to a token."""

    def __init__(self, patch_size: int, embed_dim: int, in_chans: int = 3) -> None:
        """Initialise the patch projection."""
        super().__init__()
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return tokens of shape (B, N, embed_dim)."""
        return self.proj(x).flatten(2).transpose(1, 2)


class ViT5(nn.Module):
    """Inference-only C-LiTE encoder. forward() returns the final-norm CLS token."""

    def __init__(  # noqa: PLR0913, PLR0917
        self,
        patch_size: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        pos_grid: int,
        num_registers: int,
        mlp_hidden: int,
        qkv_bias: bool = False,  # noqa: FBT001, FBT002
        rope_pt_seq_len: int = 14,
        use_rope: bool = True,  # noqa: FBT001, FBT002
        reg_theta: float = 100,
        gating_type: str | None = None,
    ) -> None:
        """Initialise the encoder."""
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_registers = num_registers
        self.patch_embed = PatchEmbed(patch_size, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.reg_token = (
            nn.Parameter(torch.zeros(1, num_registers, embed_dim))
            if num_registers
            else None
        )
        self.pos_embed = nn.Parameter(torch.zeros(1, pos_grid * pos_grid, embed_dim))
        # Unused at inference, kept so that the checkpoint loads without errors.
        self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        rope_reg_size = math.isqrt(num_registers)
        if rope_reg_size**2 != num_registers:
            msg = "num_registers must be a square number"
            raise ValueError(msg)
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    mlp_hidden,
                    num_heads=num_heads,
                    qkv_bias=qkv_bias,
                    rope_pt_seq_len=rope_pt_seq_len,
                    rope_reg_size=rope_reg_size,
                    num_registers=num_registers,
                    reg_theta=reg_theta,
                    use_rope=use_rope,
                    gating_type=gating_type,
                )
                for _ in range(depth)
            ]
        )
        self.norm = RMSNorm(embed_dim)

    def interpolate_pos_encoding(self, n_patch: int, h: int, w: int) -> torch.Tensor:
        """Resize the learned position embeddings to the input grid."""
        n, dim = self.pos_embed.shape[1:]
        if n_patch == n and w == h:
            return self.pos_embed
        g = math.isqrt(n)
        pe = self.pos_embed.reshape(1, g, g, dim).permute(0, 3, 1, 2)
        pe = F.interpolate(
            pe.float(),
            size=(h // self.patch_size, w // self.patch_size),
            mode="bicubic",
            align_corners=False,
        ).to(self.pos_embed.dtype)
        return pe.permute(0, 2, 3, 1).reshape(1, -1, dim)

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return the final CLS, patch and register tokens."""
        b, _, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x.shape[1], h, w)
        x = torch.cat((self.cls_token.expand(b, -1, -1), x), dim=1)
        if self.reg_token is not None:
            x = torch.cat((x, self.reg_token.expand(b, -1, -1)), dim=1)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        r = self.num_registers
        return {
            "cls": x[:, 0],
            "patch": x[:, 1 : x.shape[1] - r],
            "reg": x[:, x.shape[1] - r :],
        }

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the CLS token."""
        return self.forward_features(x)["cls"]

def normalize_state_dict(sd: dict) -> dict:
    """Extract the backbone weights from a C-LiTE training checkpoint."""
    for key in ("model", "state_dict"):
        if isinstance(sd, dict) and isinstance(sd.get(key), dict):
            sd = sd[key]
    if isinstance(sd.get("teacher"), dict):
        sd = sd["teacher"]
    elif isinstance(sd.get("student"), dict):
        logger.info("No 'teacher' key -- using the 'student' weights.")
        sd = sd["student"]
    for pref in ("teacher.backbone.", "student.backbone.", "backbone.", "module."):
        hits = {
            k[len(pref) :].replace("_fsdp_wrapped_module.", ""): v
            for k, v in sd.items()
            if k.startswith(pref)
        }
        if hits:
            logger.info("Stripped prefix '%s' -> %d tensors", pref, len(hits))
            sd = hits
            break
    heads = ("dino_head", "ibot_head", "gram_teacher", "contrastive_head", "head.")
    return {k: v for k, v in sd.items() if not k.startswith(heads)}


def build_vit5_from_state_dict(sd: dict) -> tuple[ViT5, dict]:
    """Infer the encoder configuration from the weights and build the model."""
    embed_dim = sd["cls_token"].shape[-1]
    depth = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("blocks."))
    head_dim = sd["blocks.0.attn.q_norm.weight"].shape[0]
    num_heads = embed_dim // head_dim
    gate_key = "blocks.0.attn.gate_proj.weight"
    if gate_key not in sd:
        gating_type = None
    elif sd[gate_key].shape[0] == num_heads:
        gating_type = "headwise"
    else:
        gating_type = "elementwise"
    cfg = {
        "patch_size": sd["patch_embed.proj.weight"].shape[-1],
        "embed_dim": embed_dim,
        "depth": depth,
        "num_heads": num_heads,
        "pos_grid": math.isqrt(sd["pos_embed"].shape[1]),
        "num_registers": sd["reg_token"].shape[1] if "reg_token" in sd else 0,
        "mlp_hidden": sd["blocks.0.mlp.fc1.weight"].shape[0],
        "qkv_bias": "blocks.0.attn.qkv.bias" in sd,
        "rope_pt_seq_len": ROPE_PT_SEQ_LEN,
        "use_rope": True,
        "reg_theta": REG_THETA,
        "gating_type": gating_type,
    }
    if cfg["pos_grid"] ** 2 != sd["pos_embed"].shape[1]:
        msg = "pos_embed grid must be square"
        raise ValueError(msg)
    return ViT5(**cfg), cfg


def load_clite(checkpoint_path, device="cpu") -> ViT5:
    """Build the encoder from a C-LiTE training checkpoint and load its weights."""
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = normalize_state_dict(ckpt)
    model, arch = build_vit5_from_state_dict(sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [k for k in missing if not k.endswith(("attn.rope.freqs", "attn.rope_reg.freqs"))]
    if missing or unexpected:
        raise RuntimeError(f"Checkpoint mismatch: missing={missing} unexpected={unexpected}")
    logger.info("C-LiTE loaded: %s", arch)
    return model.to(device).eval()
