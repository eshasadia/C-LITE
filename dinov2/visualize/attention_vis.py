import sys, argparse, functools, importlib.util
import torch
import numpy as np
from PIL import Image
import torch.nn.functional as F
torch.load = functools.partial(torch.load, weights_only=False)

sys.path.append(".")
from dinov2.utils.config import setup
_SSL_META_ARCH_PATH = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/cell_representation_learning/dinov2/train/ssl_meta_arch_v0.py"
_spec = importlib.util.spec_from_file_location("ssl_meta_arch_v0", _SSL_META_ARCH_PATH)
ssl_meta_arch_v0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssl_meta_arch_v0)

get_downloaded_dino_vit_interpolated = ssl_meta_arch_v0.get_downloaded_dino_vit_interpolated
load_backbone_checkpoint = ssl_meta_arch_v0.load_backbone_checkpoint
build_model_from_cfg = ssl_meta_arch_v0.build_model_from_cfg

# --- attention-viz helpers, copied inline (self-contained, no train_v0 dependency) ---
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _find_last_block(backbone):
    blocks = backbone.blocks
    last_block = None
    for chunk in blocks:
        sub_blocks = list(chunk) if hasattr(chunk, "__iter__") else [chunk]
        for b in sub_blocks:
            if b.__class__.__name__ == "Identity":
                continue
            if hasattr(b, "attn") and hasattr(b.attn, "qkv"):
                last_block = b
    if last_block is None:
        raise RuntimeError("Could not find a transformer block with attn.qkv on this backbone.")
    return last_block


class _AttentionExtractor:
    def __init__(self, backbone):
        self.backbone = backbone
        self.block = _find_last_block(backbone)
        self.attn_module = self.block.attn
        self.num_heads = self.attn_module.num_heads
        self._qkv_output = None
        self._handle = None

    def _hook(self, module, inp, out):
        self._qkv_output = out.detach()

    def __enter__(self):
        self._qkv_output = None
        self._handle = self.attn_module.qkv.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove()

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def get_attention(self):
        if self._qkv_output is None:
            raise RuntimeError("get_attention() called before a forward pass.")
        B, N, three_c = self._qkv_output.shape
        C = three_c // 3
        head_dim = C // self.num_heads
        qkv = self._qkv_output.reshape(B, N, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k = qkv[0], qkv[1]
        scale = head_dim ** -0.5
        attn = (q.float() @ k.float().transpose(-2, -1)) * scale
        return attn.softmax(dim=-1)  # (B, heads, N, N)


def _get_cls_attention_maps(attn, img_size, num_register_tokens=0):
    B, num_heads, N, _ = attn.shape
    num_patches = N - 1 - num_register_tokens
    grid = int(num_patches ** 0.5)
    if grid * grid != num_patches:
        raise ValueError(f"Token count {N} doesn't form a square patch grid ({num_patches} patches).")
    cls_attn = attn[:, :, 0, 1 + num_register_tokens:]
    cls_attn = cls_attn.reshape(B, num_heads, grid, grid)
    cls_attn = F.interpolate(cls_attn, size=(img_size, img_size), mode="bilinear", align_corners=False)
    return cls_attn.cpu().numpy()


def _unnormalize_image(tensor, mean=_IMAGENET_MEAN, std=_IMAGENET_STD):
    mean_t = torch.tensor(mean, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, device=tensor.device, dtype=tensor.dtype).view(3, 1, 1)
    img = (tensor * std_t + mean_t).clamp(0, 1)
    return (img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)


def _overlay_attention(image_rgb_uint8, attn_map, alpha=0.5):
    import cv2
    attn_norm = attn_map - attn_map.min()
    attn_norm = (attn_norm / (attn_norm.max() + 1e-8) * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(attn_norm, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return (alpha * heatmap + (1 - alpha) * image_rgb_uint8).astype(np.uint8)


def _build_attention_grid(image_rgb_uint8, head_attn_maps, alpha=0.5, max_heads=8):
    panels = [Image.fromarray(image_rgb_uint8)]
    mean_map = head_attn_maps.mean(axis=0)
    panels.append(Image.fromarray(_overlay_attention(image_rgb_uint8, mean_map, alpha)))
    for h in range(min(head_attn_maps.shape[0], max_heads)):
        panels.append(Image.fromarray(_overlay_attention(image_rgb_uint8, head_attn_maps[h], alpha)))
    widths, heights = zip(*(p.size for p in panels))
    grid = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for p in panels:
        grid.paste(p, (x, 0))
        x += p.width
    return grid


# 1. Build config + a bare backbone -- same construction path SSLMetaArch.__init__ uses
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser("Attention visualization", add_help=add_help)
    parser.add_argument("--config-file", default="./dinov2/configs/train/custom.yaml", metavar="FILE")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    parser.add_argument("--output-dir", "--output_dir", default="", type=str)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--eval", type=str, default="")
    parser.add_argument("--name", type=str, default="debug")
    parser.add_argument("--local-rank", type=int, default=0)
    return parser


args = get_args_parser().parse_args()
cfg = setup(args)

if cfg.student.arch in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
    backbone = get_downloaded_dino_vit_interpolated(
        cfg.student.arch,
        patch_size=cfg.student.patch_size,
        img_size=cfg.crops.global_crops_size,
    )
else:
    backbone, _, _ = build_model_from_cfg(cfg)

# 2. Load the teacher checkpoint
ckpt_path = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/imunocto_eval/dino_p4_gram/dinov2_p4_gram/eval/training_999/teacher_checkpoint.pth"
load_backbone_checkpoint(backbone, ckpt_path)
backbone = backbone.cuda().eval()

# 3. Prepare images
def load_and_preprocess(path, img_size):
    img = Image.open(path).convert("RGB").resize((img_size, img_size))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - np.array(_IMAGENET_MEAN)) / np.array(_IMAGENET_STD)
    return torch.from_numpy(arr).permute(2, 0, 1).float()


img_size = cfg.crops.global_crops_size
paths = [
    "/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Connective/fold1_00002_00010_Connective_Breast.png",
    "/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Epithelial/fold1_00045_00003_Epithelial_Breast.png",
]
images = torch.stack([load_and_preprocess(p, img_size) for p in paths]).cuda()

# 4. Extract attention
with _AttentionExtractor(backbone) as extractor:
    with torch.no_grad():
        backbone(images, is_training=True)
    attn = extractor.get_attention()

num_register_tokens = getattr(backbone, "num_register_tokens", 0)
attn_maps = _get_cls_attention_maps(attn, img_size, num_register_tokens=num_register_tokens)

# 5. Save the grids
for i in range(images.shape[0]):
    img_rgb = _unnormalize_image(images[i])
    grid = _build_attention_grid(img_rgb, attn_maps[i], alpha=0.5, max_heads=8)
    grid.save(
        f"/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/cell_representation_learning/dinov2/output_visuals/attention_gram/999/attn_sample_{i}.png"
    )