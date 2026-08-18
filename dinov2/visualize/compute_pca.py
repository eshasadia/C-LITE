import sys, argparse, functools, importlib.util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.decomposition import PCA

torch.load = functools.partial(torch.load, weights_only=False)

sys.path.append(".")
from dinov2.utils.config import setup

# --- load ssl_meta_arch_v0.py directly by path (bypasses dinov2/train/__init__.py) ---
_SSL_META_ARCH_PATH = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/cell_representation_learning/dinov2/train/ssl_meta_arch_v0.py"
_spec = importlib.util.spec_from_file_location("ssl_meta_arch_v0", _SSL_META_ARCH_PATH)
ssl_meta_arch_v0 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ssl_meta_arch_v0)

get_downloaded_dino_vit_interpolated = ssl_meta_arch_v0.get_downloaded_dino_vit_interpolated
load_backbone_checkpoint = ssl_meta_arch_v0.load_backbone_checkpoint
build_model_from_cfg = ssl_meta_arch_v0.build_model_from_cfg

PATCH_SIZE = 4
IMAGE_SIZE = 64
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

OUTPUT_DIR = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/cell_representation_learning/dinov2/output_visuals/pca/9999"

# ---- fill this in ----
image_path = "/tia/phd-data/u5552013/ImmunoDINO_datasets/PanNuke/Fold1_cell_patches/Connective/fold1_00002_00010_Connective_Breast.png"
ckpt_path = "/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/imunocto_eval/dino_p4/dinov2_p4/eval/training_9999/teacher_checkpoint.pth"


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser("PCA visualization", add_help=add_help)
    parser.add_argument(
        "--config-file",
        default="/tia/phd-data/u5552013/ImmunoDINO_baselines_comparison/cell_representation_learning/dinov2/configs/train/custom.yaml",
        metavar="FILE",
    )
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

# --- build backbone + load teacher weights ---
if cfg.student.arch in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
    model = get_downloaded_dino_vit_interpolated(
        cfg.student.arch,
        patch_size=cfg.student.patch_size,
        img_size=cfg.crops.global_crops_size,
    )
else:
    model, _, _ = build_model_from_cfg(cfg)

load_backbone_checkpoint(model, ckpt_path)
model = model.cuda().eval()


def resize_transform(
    mask_image: Image.Image,
    image_size: int = IMAGE_SIZE,
    patch_size: int = PATCH_SIZE,
) -> torch.Tensor:
    w, h = mask_image.size
    h_patches = int(image_size / patch_size)
    w_patches = int((w * image_size) / (h * patch_size))
    return TF.to_tensor(TF.resize(mask_image, (h_patches * patch_size, w_patches * patch_size)))


image = Image.open(image_path).convert("RGB")
image_resized = resize_transform(image)
image_resized_norm = TF.normalize(image_resized, mean=IMAGENET_MEAN, std=IMAGENET_STD)

with torch.inference_mode():
    feats = model.get_intermediate_layers(
        image_resized_norm.unsqueeze(0).cuda(), n=1, reshape=True, norm=True
    )
    x = feats[-1].squeeze().detach().cpu()
    dim = x.shape[0]
    x = x.view(dim, -1).permute(1, 0)

h_patches, w_patches = [int(d / PATCH_SIZE) for d in image_resized.shape[1:]]

# --- PCA on all patches (no foreground masking) ---
pca = PCA(n_components=3, whiten=True)
pca.fit(x)

projected_image = torch.from_numpy(pca.transform(x.numpy())).view(h_patches, w_patches, 3)

# multiply by 2.0 and pass through a sigmoid to get vibrant colors
projected_image = torch.nn.functional.sigmoid(projected_image.mul(2.0)).permute(2, 0, 1)  # (3, h_patches, w_patches)
img_h, img_w = image_resized.shape[1], image_resized.shape[2]  # e.g. 64, 64
projected_image_up = F.interpolate(
    projected_image.unsqueeze(0), size=(img_h, img_w), mode="bicubic", align_corners=False
).squeeze(0)  # (3, img_h, img_w)

plt.rcParams.update({
    "xtick.labelsize": 5,
    "ytick.labelsize": 5,
    "axes.labelsize": 5,
    "axes.titlesize": 4,
})

plt.figure(figsize=(6, 2), dpi=300)

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.axis('off')
plt.title(f"Image, Size {image.size}")

plt.subplot(1, 3, 2)
plt.imshow(projected_image_up.permute(1, 2, 0))
plt.title(f"PCA (3 comp), {img_h}x{img_w}")
plt.axis('off')

plt.subplot(1, 3, 3)
image_resized_np = image_resized.permute(1, 2, 0).numpy()  # (H, W, 3), values in [0, 1]
overlay = 0.5 * image_resized_np + 0.5 * projected_image_up.permute(1, 2, 0).numpy()
plt.imshow(overlay)
plt.title("Overlay")
plt.axis('off')

plt.savefig(f"{OUTPUT_DIR}/pca.png", bbox_inches="tight")
plt.close()