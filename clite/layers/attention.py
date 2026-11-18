import logging
import os
import warnings

from torch import Tensor, nn
import torch

logger = logging.getLogger("dinov2")

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        from xformers.ops import memory_efficient_attention, unbind

        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Attention)")
    else:
        warnings.warn("xFormers is disabled (Attention)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Attention)")


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        gating_type: str = None,       # <-- NEW: None (default, unchanged behavior), "headwise", or "elementwise"
        gate_bias_init: float = 4.0,   # <-- NEW: only used if gating_type is not None
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.return_attention = False
        self.last_attn = None

        # --- NEW: Gated Attention (Qiu et al. 2025, NeurIPS Best Paper) ---
        # Same mechanism as vit5_arch.py's Attention: a sigmoid gate,
        # conditioned on the pre-attention input, applied to the attention
        # output right before the output projection.
        #     out = proj( sigmoid(gate_proj(x_in)) * Attn(Q, K, V) )
        # gate_bias_init=4.0 with zero weight => sigmoid(4.0)~=0.98 at init,
        # so a freshly-constructed gated model behaves ~identically to the
        # ungated one and learns the gate from there.
        assert gating_type in (None, "headwise", "elementwise")
        self.gating_type = gating_type
        if gating_type is not None:
            gate_out_dim = num_heads if gating_type == "headwise" else dim
            self.gate_proj = nn.Linear(dim, gate_out_dim)
            nn.init.zeros_(self.gate_proj.weight)
            nn.init.constant_(self.gate_proj.bias, gate_bias_init)

    def _apply_gate(self, x_in: Tensor, attn_out: Tensor) -> Tensor:
        if self.gating_type is None:
            return attn_out
        B, N, C = x_in.shape
        gate = torch.sigmoid(self.gate_proj(x_in))
        if self.gating_type == "headwise":
            gate = gate.unsqueeze(-1).expand(B, N, self.num_heads, C // self.num_heads).reshape(B, N, C)
        return attn_out * gate

    def forward(self, x: Tensor) -> Tensor:
        x_in = x  # NEW: keep pre-attention input around for the gate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        if self.return_attention:
            self.last_attn = attn.detach()

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self._apply_gate(x_in, x)  # <-- NEW: gate applied before proj, matching vit5_arch.py
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MemEffAttention(Attention):
    def forward(self, x: Tensor, attn_bias=None) -> Tensor:
        x_in = x  # NEW

        if not XFORMERS_AVAILABLE:
            if attn_bias is not None:
                raise AssertionError("xFormers is required for using nested tensors")
            return super().forward(x)

        if self.return_attention:
            logger.warning(
                "return_attention=True has no effect on MemEffAttention -- xFormers' "
                "memory_efficient_attention never materializes a dense attention tensor "
                "to capture. Set XFORMERS_DISABLED=1 to fall back to plain Attention if "
                "you need real attention maps out of this run."
            )

        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)

        q, k, v = unbind(qkv, 2)

        x = memory_efficient_attention(q, k, v, attn_bias=attn_bias)
        x = x.reshape([B, N, C])
        x = self._apply_gate(x_in, x)  # <-- NEW: same gate, applied on the xformers fast path too

        x = self.proj(x)
        x = self.proj_drop(x)
        return x
