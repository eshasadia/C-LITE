import torch
import torch.nn as nn

try:
    import timm
    from timm.layers import SwiGLUPacked
except ImportError:
    timm = None
    SwiGLUPacked = None


class HistoPLUS(nn.Module):
    """
    HistoPLUS encoder feature extractor.

    HistoPLUS (Adjadj et al., "Towards Comprehensive Cellular Characterisation
    of H&E slides", 2025; https://huggingface.co/Owkin-Bioptimus/histoplus)
    is a CellViT-style architecture: a ViT encoder (H0-mini, ~86M params,
    distilled from H-Optimus-0 via DINO+iBOT) feeding a task-multiplexed
    HoVerNet-style decoder with NP (nuclei prediction), HV (horizontal-
    vertical distance), and NT (13-class nuclei typing) heads. This wrapper
    loads only the H0-mini encoder and mean-pools its output, discarding
    the decoder heads -- the same "encoder-only, decoders discarded"
    pattern this codebase already uses for CellViTPlusPlus (SAM-H) and
    HoverNeXt (ConvNeXt-V2).

    LICENSE -- READ BEFORE USE: this model is gated on Hugging Face,
    academic/research use only. You must create a HF account, accept the
    license on https://huggingface.co/Owkin-Bioptimus/histoplus, and
    authenticate (huggingface-cli login or HF_TOKEN) before downloading
    weights. Commercial use, redistribution, and publishing copies of the
    model are explicitly prohibited by its terms of use -- check these
    terms yourself before including HistoPLUS-derived results in anything
    beyond internal/academic comparison.

    CAVEAT: this is a feature-extraction proxy, not a reproduction of
    HistoPLUS's actual task output. Its decoders (never loaded here) are
    where its cell detection/segmentation/typing actually happens -- what
    you get is "whatever the fine-tuned H0-mini encoder learned inside
    that pipeline," pooled into one vector. Reasonable for a supervised
    (well: supervised-decoder-trained, SSL-pretrained-encoder) baseline
    under your linear-probe protocol; not a substitute for running
    HistoPLUS's actual inference pipeline if you want its real
    segmentation/13-class typing predictions.

    UNVERIFIED CHECKPOINT FORMAT -- this has not been checked against a
    real downloaded checkpoint (unlike HoverNeXt, where a real traceback
    let this get fixed precisely). Prefix search tries the most likely
    candidates given CellViT-family conventions ("encoder.model.",
    "encoder.", "model.encoder.", "vit_encoder.", "backbone.",
    "model.backbone."), falling back to the bare state_dict. If none
    match, the error message prints the first 20 real keys found -- same
    as it did successfully for HoverNeXt -- so this can be corrected in
    one round once you have an actual checkpoint to test against.

    Requires:
      pip install timm
      huggingface-cli login  (or HF_TOKEN env var), after accepting the
      license at https://huggingface.co/Owkin-Bioptimus/histoplus
      A downloaded HistoPLUS checkpoint file, passed via checkpoint_path
    """
    def __init__(self, checkpoint_path, in_chans=3):
        super().__init__()
        if timm is None:
            raise ImportError(
                "timm not installed. Run: pip install timm"
            )
        if checkpoint_path is None:
            raise ValueError(
                "checkpoint_path is required for histoplus: path to a "
                "downloaded HistoPLUS checkpoint (see license note in "
                "this class's docstring -- gated HF model, academic use "
                "only, requires accepting terms before download)"
            )

        # Bare H0-mini encoder shell (same construction as owkin.py's
        # H0mini class), overwritten with HistoPLUS's fine-tuned encoder
        # weights below rather than the generic SSL-pretrained ones.
        self.encoder = timm.create_model(
            "hf-hub:bioptimus/H0-mini",
            pretrained=False,
            num_classes=0,
            mlp_layer=SwiGLUPacked,
            act_layer=torch.nn.SiLU,
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

        encoder_state = {}
        matched_prefix = None
        for prefix in (
            "encoder.model.", "encoder.", "model.encoder.",
            "vit_encoder.", "backbone.", "model.backbone.",
        ):
            candidate = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if len(candidate) > 0:
                encoder_state = candidate
                matched_prefix = prefix
                break

        if len(encoder_state) == 0:
            encoder_state = state_dict
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
                "Add the correct prefix to the list in HistoPLUS.__init__ once "
                "you can see the real key naming -- this is unverified against "
                "an actual downloaded checkpoint (see UNVERIFIED CHECKPOINT "
                "FORMAT in the class docstring)."
            )

        msg = self.encoder.load_state_dict(filtered_sd, strict=False)
        print(f"[histoplus] Matched prefix: '{matched_prefix}'")
        print(f"[histoplus] Loaded {len(filtered_sd)}/{len(encoder_state)} encoder tensors "
              f"(skipped {len(skipped)} due to shape mismatch or unknown keys)")
        if skipped:
            print(f"  skipped (first 5): {skipped[:5]}")
        if msg.missing_keys:
            print(f"  missing (first 5): {msg.missing_keys[:5]}")
        if msg.unexpected_keys:
            print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")
        if len(filtered_sd) < len(target_sd) * 0.5:
            print(
                "[histoplus] WARNING: fewer than half of the encoder's own "
                "tensors were matched -- this checkpoint's key naming likely "
                "isn't fully covered by the prefix list yet. Treat results "
                "with caution until this is verified/fixed."
            )

    def forward(self, x):
        outputs = self.encoder(x)
        # timm ViT with num_classes=0 returns the pooled CLS-token feature
        # directly as a (B, C) tensor already -- no further pooling needed,
        # unlike the ConvNeXt (spatial map) or SAM (spatial map) cases.
        if outputs.dim() > 2:
            outputs = outputs.mean(dim=list(range(2, outputs.dim())))
        return outputs