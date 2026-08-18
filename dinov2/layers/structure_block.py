# dinov2/layers/structure_block.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class MorphologyAffinity(nn.Module):
    def __init__(self, in_dim=3, proj_dim=16):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, proj_dim), nn.GELU(), nn.Linear(proj_dim, proj_dim))

    def forward(self, morph_feat):
        z = F.normalize(self.proj(morph_feat), dim=-1)
        return torch.bmm(z, z.transpose(1, 2))


class StructureAttentionBlock(nn.Module):
    """Adds one struct token that cross-attends to patch tokens, with morphology-similarity bias added to the attention logits."""

    def __init__(self, hidden_dim, n_heads=4, lam_init=1.0, use_morphology_bias=True):
        super().__init__()
        self.use_morphology_bias = use_morphology_bias
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.morph_affinity = MorphologyAffinity()
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(lam_init)))
        self.struct_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, patch_tokens, morph_feat):
        B, N, D = patch_tokens.shape
        struct_tok = self.struct_token.expand(B, -1, -1)
        x = torch.cat([struct_tok, patch_tokens], dim=1)
        qkv = self.qkv(self.norm(x)).reshape(B, N + 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        M = self.morph_affinity(morph_feat)
        if self.use_morphology_bias:
            lam = torch.exp(self.log_lambda)
            bias = torch.zeros(B, N + 1, N + 1, device=x.device, dtype=x.dtype)
            bias[:, 1:, 1:] = lam * M
            attn_scores = attn_scores + bias.unsqueeze(1)
        attn = attn_scores.softmax(dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, N + 1, D))
        x = x + out
        struct_tok_out, patch_tokens_out = x[:, :1], x[:, 1:]
        struct_attn_map = attn[:, :, 1:, 0].mean(1)
        return patch_tokens_out, struct_tok_out.squeeze(1), struct_attn_map, M