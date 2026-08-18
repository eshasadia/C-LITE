import re

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    timm = None


def _remap_underscore_index_segments(key):
    """
    hover_next_train checkpoints name top-level ConvNeXt-V2 submodules like
    "stem_0", "stem_1", "stages_0" (underscore-separated index) instead of
    timm's own "stem.0", "stem.1", "stages.0" (dot-separated, i.e. regular
    nn.Sequential/ModuleList indexing). Everything below that level
    (blocks.N.conv_dw, norm, mlp.fc1, mlp.grn, mlp.fc2) already matches
    timm's naming exactly, so this only needs to fix the top-level segment.
    "conv_dw", "fc1", "fc2", "grn" etc. are untouched since they don't
    match the trailing-underscore-digit pattern.
    """
    parts = key.split(".")
    fixed_parts = []
    for part in parts:
        m = re.fullmatch(r"([a-zA-Z]+)_([0-9]+)", part)
        if m:
            fixed_parts.append(m.group(1))
            fixed_parts.append(m.group(2))
        else:
            fixed_parts.append(part)
    return ".".join(fixed_parts)


class HoverNeXt(nn.Module):
    """
    HoVer-NeXt encoder feature extractor.

    HoVer-NeXt (Baumann et al., "HoVer-NeXt: A Fast Nuclei Segmentation and
    Classification Pipeline for Next Generation Histopathology", MIDL 2024;
    https://github.com/digitalpathologybern/hover_next_train) is an
    encoder-decoder nuclei segmentation+classification model built on a
    ConvNeXt-V2 encoder, with separate instance and class decoder branches.
    This wrapper loads only the encoder and mean-pools its final feature
    map to produce one embedding per input crop -- analogous to how
    CellViTPlusPlus in this codebase uses only SAM-H's encoder and discards
    CellViT's segmentation/classification decoders.

    CAVEAT: this is a feature-extraction proxy, not a reproduction of
    HoVer-NeXt's actual task output. HoVer-NeXt's decoders (which this
    wrapper never loads) are where its instance segmentation and nucleus
    typing actually happen -- what you get here is "whatever spatial
    features its ConvNeXt-V2 encoder learned while training toward that
    objective," pooled into a single vector. Reasonable for a supervised
    representation-quality baseline; not a substitute for running the full
    HoVer-NeXt pipeline if you need its actual segmentation/classification
    predictions.

    CONFIRMED CHECKPOINT FORMAT (from an actual hover_next_train checkpoint
    downloaded from https://zenodo.org/records/10635618): weights are keyed
    as "encoder.model.stem_0...", "encoder.model.stages_0.blocks.N...",
    i.e. prefix "encoder.model." followed by standard timm ConvNeXt-V2
    block internals (conv_dw, norm, mlp.fc1, mlp.grn, mlp.fc2), except the
    top-level "stem"/"stages" indices use underscores ("stem_0",
    "stages_0") instead of timm's own dot-indexed Sequential/ModuleList
    convention ("stem.0", "stages.0"). Both the prefix and this naming
    difference are handled automatically below. `backbone_name` still
    needs to match the actual size (tiny/base/large) your checkpoint uses
    -- this repo's Zenodo release includes a convnextv2_tiny checkpoint,
    so set backbone_name="convnextv2_tiny" if that's what you downloaded.

    PERFORMANCE NOTE (the good news): unlike CellViTPlusPlus/SAM's ViT,
    ConvNeXt has no absolute positional embedding, so there is no
    fixed-native-resolution constraint and no pos_embed-interpolation step
    needed here. You can run this directly at your actual crop resolution
    (e.g. 64x64) with no upsampling and no token-count blowup -- the
    OOM/attention-cost problem CellViTPlusPlus has at small crop sizes
    does not apply to this architecture.

    Requires:
      pip install timm
      A HoVer-NeXt checkpoint (best_model) from
      https://github.com/digitalpathologybern/hover_next_train
      passed via checkpoint_path
    """
    def __init__(self, checkpoint_path, backbone_name="convnextv2_tiny", in_chans=3):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm not installed. Run: pip install timm"
            )
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for hovernext: path to a "
                "hover_next_train best_model checkpoint"
            )

        # Bare ConvNeXt-V2 shell (random init), overwritten with HoVer-NeXt
        # encoder weights below. num_classes=0 strips the classifier head
        # so forward() returns the pooled feature map directly if needed,
        # though we use forward_features() + our own pooling to be explicit
        # about what's happening.
        self.encoder = timm.create_model(
            backbone_name, pretrained=False, num_classes=0, in_chans=in_chans
        )

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(ckpt, dict) and any(
            k in ckpt for k in ("model", "model_state_dict", "state_dict")
        ):
            state_dict = ckpt.get(
                "model", ckpt.get("model_state_dict", ckpt.get("state_dict"))
            )
        else:
            state_dict = ckpt

        # HoVer-NeXt checkpoint releases may prefix the encoder submodule
        # differently depending on the training-code version; try known
        # candidates, longest/most-specific first (a short prefix like
        # "encoder." can spuriously "match" when the real prefix is the
        # longer "encoder.model.", leaving a stray "model." segment behind
        # that then fails to match anything -- this happened in practice,
        # see the "encoder.model.stem_0..." checkpoint format below).
        encoder_state = {}
        matched_prefix = None
        for prefix in ("encoder.model.", "model.encoder.", "encoder.", "backbone.", "model.backbone."):
            candidate = {
                _remap_underscore_index_segments(key[len(prefix):]): value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if len(candidate) > 0:
                encoder_state = candidate
                matched_prefix = prefix
                break

        if len(encoder_state) == 0:
            # No known prefix matched -- fall back to trying the state_dict
            # as-is (bare encoder checkpoint, no prefix at all), still
            # applying the underscore->dot remap in case it's needed.
            encoder_state = {
                _remap_underscore_index_segments(k): v for k, v in state_dict.items()
            }
            matched_prefix = "(none -- used state_dict as-is)"

        target_sd = self.encoder.state_dict()
        filtered_sd = {
            k: v for k, v in encoder_state.items()
            if k in target_sd and target_sd[k].shape == v.shape
        }
        skipped = [k for k in encoder_state.keys() if k not in filtered_sd]

        if len(filtered_sd) == 0:
            sample_keys = list(state_dict.keys())[:20]
            raise KeyError(
                "Could not match any encoder weights. Tried prefix "
                f"'{matched_prefix}' with 0 shape-matching tensors found. "
                f"First 20 keys actually present in the checkpoint: {sample_keys}. "
                "This likely means either the key prefix list in HoverNeXt.__init__ "
                "needs a new entry for your checkpoint version, or `backbone_name` "
                "doesn't match the ConvNeXt-V2 size this checkpoint was trained with "
                "(shapes won't match even with the right prefix). Inspect the keys "
                "above and adjust accordingly."
            )

        msg = self.encoder.load_state_dict(filtered_sd, strict=False)
        print(f"[hovernext] Matched prefix: '{matched_prefix}'")
        print(f"[hovernext] Loaded {len(filtered_sd)}/{len(encoder_state)} encoder tensors "
              f"(skipped {len(skipped)} due to shape mismatch or unknown keys)")
        if skipped:
            print(f"  skipped (first 5): {skipped[:5]}")
        if msg.missing_keys:
            print(f"  missing (first 5): {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")

    def forward(self, x):
        feats = self.encoder.forward_features(x)  # (B, C, H', W') final stage feature map
        if feats.dim() == 4:
            feats = feats.mean(dim=[2, 3])         # global average pool -> (B, C)
        return feats