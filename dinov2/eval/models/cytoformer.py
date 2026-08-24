import torch
import torch.nn as nn

try:
    from cytoformer import CellClassifier
except ImportError:
    CellClassifier = None


class CytoFormer(nn.Module):
    """
    Wrapper around zhihuanglab/CytoFormer's CellClassifier, used here purely
    as a generic cell-level feature extractor (not for its 16 per-organ
    classification heads).

    Source: https://github.com/zhihuanglab/CytoFormer
    Weights: https://huggingface.co/zhihuanglab/CytoFormer

    Architecture: ViT-giant/14, 1536-d embedding, initialised from UNI2-h
    and fine-tuned end-to-end on 15.4M cells (81 paired Xenium/H&E sections,
    16 organs), with cell-type labels derived from the spatial
    transcriptome rather than manual annotation. A separate linear head per
    organ sits on top of the encoder; the organ id is a routing signal only
    (not fed to the encoder), so CellClassifier.extract_features(x) returns
    the same 1536-d embedding regardless of which organ is declared --
    that's what forward() calls here, so this wrapper needs no organ
    argument and slots into the same `model(images) -> features` interface
    as every other model in return_model.py.

    Checkpoint format: a single checkpoint.pth containing the *entire*
    network (encoder + all 16 heads) as a plain state_dict -- unlike the
    DINO-style checkpoints elsewhere in this file, there's no
    "teacher"/"student"/"backbone" nesting to strip. Download it with:
        hf download zhihuanglab/CytoFormer checkpoint.pth --local-dir <dir>
    CellClassifier() also needs organ_celltype_map.json sitting next to
    wherever the `cytoformer` package looks for it (see the upstream
    README's "Install" section) -- copy it from the cloned repo:
        cp <cytoformer_repo>/cytoformer/organ_celltype_map.json <dir>/
    The UNI2-h foundation weights are NOT needed separately; the checkpoint
    already contains the fine-tuned encoder.

    Native input: 224x224 RGB, ImageNet mean/std, cropped from a 56 um
    field of view centred on the nucleus. This wrapper does not do the
    mpp-aware WSI cropping itself -- that happens upstream in your
    dataset/patch-extraction code (mirroring CytoFormer's own
    scripts/crop_cells.py --mpp/--fov_um); get_transforms(..., "cytoformer")
    in this file only handles the final resize-to-224 + normalize step,
    same as the other 224px models.
    """

    def __init__(self, saved_model_path):
        super().__init__()
        if CellClassifier is None:
            raise ImportError(
                "Could not import CellClassifier from the `cytoformer` package. "
                "Install it with:\n"
                "  pip install git+https://github.com/zhihuanglab/CytoFormer\n"
                "or clone the repo and `pip install -e .` from within it."
            )
        if saved_model_path is None:
            raise ValueError(
                "--checkpoint is required for cytoformer -- pass the path to "
                "checkpoint.pth downloaded from "
                "huggingface.co/zhihuanglab/CytoFormer. Make sure "
                "organ_celltype_map.json is present alongside it as well "
                "(CellClassifier() reads it at construction time)."
            )

        self.net = CellClassifier()

        state_dict = torch.load(saved_model_path, map_location="cpu")
        # The documented format is a plain state_dict loadable directly via
        # net.load_state_dict(...). Unwrap defensively in case a given
        # download wraps it in a dict, without assuming any particular key.
        if isinstance(state_dict, dict) and not any(
            isinstance(v, torch.Tensor) for v in state_dict.values()
        ):
            for key in ("state_dict", "model", "net"):
                if key in state_dict:
                    state_dict = state_dict[key]
                    break

        missing, unexpected = self.net.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"[warn] CytoFormer missing keys ({len(missing)}): "
                  f"{list(missing)[:5]}{'...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"[warn] CytoFormer unexpected keys ({len(unexpected)}): "
                  f"{list(unexpected)[:5]}{'...' if len(unexpected) > 5 else ''}")

    def forward(self, x):
        # Organ-agnostic 1536-d cell embedding -- see class docstring for
        # why extract_features (not the per-organ classification head) is
        # the right thing to call for feature extraction.
        return self.net.extract_features(x)