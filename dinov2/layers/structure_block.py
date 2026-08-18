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
    """Adds one struct token that cross-attends to patch tokens, with an
    optional morphology-similarity bias added to the attention logits.

    The morphology bias reaches THREE interactions, not just one:
      - patch-query x patch-key   (original design: morphologically similar
        patches attend to each other more)
      - struct-query x patch-key  (struct token's own attention is pulled
        toward patches with high average morphological centrality)
      - patch-query x struct-key  (symmetric complement, so patches'
        attention TO the struct token is influenced the same way)
    Without the latter two, the struct token's attention weights are never
    influenced by morphology at all -- only patch-to-patch interactions
    would be biased, leaving the struct token (and therefore the fused
    cls+struct embedding used everywhere downstream) with no actual
    connection to the morphology signal.
    """

    def __init__(self, hidden_dim, n_heads=4, lam_init=1.0, use_morphology_bias=True):
        super().__init__()
        self.use_morphology_bias = use_morphology_bias
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.qkv = nn.Linear(hidden_dim, hidden_dim * 3)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.morph_affinity = MorphologyAffinity()
        # 1D shape (numel=1), NOT a 0-d scalar -- FSDP explicitly rejects
        # 0-dimensional parameters ("FSDP doesn't support scalar parameters").
        self.log_lambda = nn.Parameter(torch.log(torch.tensor([lam_init])))
        self.struct_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

    def forward(self, patch_tokens, morph_feat):
        B, N, D = patch_tokens.shape
        struct_tok = self.struct_token.expand(B, -1, -1)
        x = torch.cat([struct_tok, patch_tokens], dim=1)
        qkv = self.qkv(self.norm(x)).reshape(B, N + 1, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        M = self.morph_affinity(morph_feat)  # (B, N, N) patch-patch morphology similarity
        if self.use_morphology_bias:
            lam = torch.exp(self.log_lambda)  # shape (1,)
            bias = torch.zeros(B, N + 1, N + 1, device=x.device, dtype=x.dtype)
            bias[:, 1:, 1:] = lam * M

            # Extend the bias to the struct token, using each patch's mean
            # affinity to all other patches as a proxy for "how central /
            # typical this patch's morphology is" within the crop.
            struct_patch_bias = lam * M.mean(dim=-1)  # (B, N)
            bias[:, 0, 1:] = struct_patch_bias  # struct-as-query, patch-as-key
            bias[:, 1:, 0] = struct_patch_bias  # patch-as-query, struct-as-key

            attn_scores = attn_scores + bias.unsqueeze(1)

        attn = attn_scores.softmax(dim=-1)
        out = self.proj((attn @ v).transpose(1, 2).reshape(B, N + 1, D))
        x = x + out
        struct_tok_out, patch_tokens_out = x[:, :1], x[:, 1:]

        # struct token's OWN attention distribution over patches
        # (query=struct, key=patches) -- "what is the struct token looking
        # at". Previously this indexed attn[:, :, 1:, 0], which is patches
        # attending TO the struct token -- the opposite direction, and not
        # what a "what did the model learn to focus on" visualization needs.
        struct_attn_map = attn[:, :, 0, 1:].mean(1)  # mean over heads -> (B, N)

        return patch_tokens_out, struct_tok_out.squeeze(1), struct_attn_map, M