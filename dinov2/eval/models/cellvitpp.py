import torch
import torch.nn as nn

try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None


class CellViTPlusPlus(nn.Module):
    """
    """
    def __init__(self, checkpoint_path, input_size=1024):
        super().__init__()
        if sam_model_registry is None:
            raise ImportError(
                "segment-anything not installed. Run: pip install segment-anything"
            )
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for cellvitpp: path to CellViT-SAM-H-x40-AMP.pth"
            )

        # Bare SAM-H encoder shell, then overwrite with CellViT-finetuned weights
        sam = sam_model_registry["vit_h"](checkpoint=None)
        self.encoder = sam.image_encoder  # ViT-H/16, native input 1024x1024

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and any(
            k in ckpt for k in ("model", "model_state_dict", "state_dict")
        ):
            state_dict = ckpt.get(
                "model", ckpt.get("model_state_dict", ckpt.get("state_dict"))
            )
        else:
            state_dict = ckpt

        # CellViT checkpoint releases have used different key prefixes for
        # the encoder submodule; try the known ones.
        encoder_state = {}
        for key, value in state_dict.items():
            for prefix in ("encoder.", "model.encoder.", "image_encoder."):
                if key.startswith(prefix):
                    encoder_state[key[len(prefix):]] = value
                    break

        if len(encoder_state) == 0:
            sample_keys = list(state_dict.keys())[:20]
            raise KeyError(
                "Could not find encoder weights using known prefixes "
                "('encoder.', 'model.encoder.', 'image_encoder.'). "
                f"First 20 keys actually present in the state_dict: {sample_keys}. "
                "Inspect these key names and add the correct prefix to the "
                "list in CellViTPlusPlus.__init__."
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