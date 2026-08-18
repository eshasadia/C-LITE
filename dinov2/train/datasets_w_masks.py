# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the Apache License, Version 2.0
# found in the LICENSE file in the root directory of this source tree.
#
# Dataset classes with mask support for morphology-aware DINO training.
# Each dataset's __getitem__ returns:
#   - (image, ())                    when return_masks=False (stock DINO path)
#   - transform(image, mask)         when return_masks=True and a transform is
#                                     set (expects a MorphAwareDataAugmentationDINO
#                                     -- or compatible -- instance, whose
#                                     __call__(image, mask) signature takes both)
#   - (image, mask)                  when return_masks=True and no transform is set
#
# `mask` is always a single-channel PIL image, same H x W as `image`,
# nonzero == nucleus.

from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ==========================================================================
# Shared mask-lookup helper for the common
# "ClassX/Images/*.png + ClassX/Masks/*.png" layout (same filename,
# single-channel mask) used by CONIC and Monkey.
# ==========================================================================

class _ImagesMasksFolderMixin:
    def _mask_path_for(self, img_path: Path) -> Path:
        # .../ClassX/Images/foo.png -> .../ClassX/Masks/foo.png
        return img_path.parent.parent / "Masks" / img_path.name

    def _load_pair(self, img_path: Path):
        img = Image.open(img_path).convert("RGB")
        if not self.return_masks:
            return img, None
        mask_path = self._mask_path_for(img_path)
        if not mask_path.exists():
            print(f"Warning: no mask for {img_path} (expected {mask_path}); using empty mask")
            mask = Image.new("L", img.size, 0)
        else:
            mask_arr = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr)
        return img, mask

    def _finish_getitem(self, img, mask):
        if not self.return_masks:
            if self.transform is not None:
                img = self.transform(img)
            return img, ()
        if self.transform is not None:
            return self.transform(img, mask)  # MorphAwareDataAugmentationDINO.__call__(image, mask)
        return img, mask


# ==========================================================================
# CONIC
# ==========================================================================

class CONICCellDataset(_ImagesMasksFolderMixin, Dataset):
    """
    Loads images directly from the specified CONIC class folders, with
    optional matching nucleus masks from the sibling Masks/ folder, e.g.
    .../CONIC/train/Connective/Images/foo.png ->
    .../CONIC/train/Connective/Masks/foo.png
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, transform=None, return_masks=False):
        self.transform = transform
        self.return_masks = return_masks
        self.samples = []

        image_dirs = [
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Connective/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Eosinophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Epithelial/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Lymphocyte/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Neutrophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/train/Plasma/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Connective/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Eosinophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Epithelial/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Lymphocyte/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Neutrophil/Images"),
            Path("/tia/phd-data/u5552013/ImmunoDINO_datasets/CONIC/test/Plasma/Images"),
        ]

        n_missing_masks = 0
        for img_dir in image_dirs:
            if not img_dir.is_dir():
                print(f"Warning: {img_dir} does not exist, skipping.")
                continue
            if return_masks and not (img_dir.parent / "Masks").is_dir():
                print(f"Warning: {img_dir.parent / 'Masks'} does not exist -- masks for this class will be empty.")
                n_missing_masks += 1
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)

        if len(self.samples) == 0:
            raise RuntimeError("No images found in the provided CONIC folders.")
        if return_masks:
            print(f"[CONICCellDataset] {len(self.samples)} crops, masks enabled "
                  f"({n_missing_masks} class folders missing a Masks/ dir)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self._load_pair(self.samples[idx])
        return self._finish_getitem(img, mask)


# ==========================================================================
# Monkey
# ==========================================================================

class MonkeyCellDataset(_ImagesMasksFolderMixin, Dataset):
    """
    Loads cell crops from the Monkey dataset. Expects:
        root/<ClassName>/Images/*.png
        root/<ClassName>/Masks/*.png   (same filename, single-channel, nonzero=nucleus)
    e.g. root/Lymphocytes/Images, root/Lymphocytes/Masks
    """

    ROOT = "/tia/phd-data/u5552013/ImmunoDINO_datasets/Monkey"
    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root=None, transform=None, return_masks=False):
        self.root = Path(root or self.ROOT)
        self.transform = transform
        self.return_masks = return_masks
        self.samples = []
        self.class_counts = Counter()

        for class_dir in sorted(self.root.iterdir()):
            images_dir = class_dir / "Images"
            if not images_dir.is_dir():
                continue
            if return_masks and not (class_dir / "Masks").is_dir():
                print(f"Warning: {class_dir / 'Masks'} does not exist -- masks for {class_dir.name} will be empty.")
            for img_path in sorted(images_dir.iterdir()):
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)
                    self.class_counts[class_dir.name] += 1

        if len(self.samples) == 0:
            raise RuntimeError(f"No crops found under {self.root}")

        print(f"\n[MonkeyCellDataset] Loaded {len(self.samples)} cell crops"
              f"{' with masks' if return_masks else ''}:")
        for cls, count in sorted(self.class_counts.items()):
            print(f"  {cls}: {count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img, mask = self._load_pair(self.samples[idx])
        return self._finish_getitem(img, mask)


# ==========================================================================
# Immunocto
# ==========================================================================

class ImmunoctoHEDataset(Dataset):
    """
    Loads HE images from:
      root/CRCxx/HE/*.png (or jpg, tif)
    with optional matching nucleus masks from the sibling mask/ folder:
      root/CRCxx/mask/*.png   (same filename, single-channel, nonzero=nucleus)
    """

    IMG_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")

    def __init__(self, root, transform=None, return_masks=False):
        self.root = Path(root)
        self.transform = transform
        self.return_masks = return_masks
        self.samples = []

        n_missing_mask_dirs = 0
        for crc_dir in sorted(self.root.iterdir()):
            he_dir = crc_dir / "HE"
            if not he_dir.is_dir():
                continue
            if return_masks and not (crc_dir / "mask").is_dir():
                print(f"Warning: {crc_dir / 'mask'} does not exist -- masks for {crc_dir.name} will be empty.")
                n_missing_mask_dirs += 1
            for img_path in he_dir.iterdir():
                if img_path.suffix.lower() in self.IMG_EXTENSIONS:
                    self.samples.append(img_path)

        if len(self.samples) == 0:
            raise RuntimeError(f"No HE images found under {root}")
        if return_masks:
            print(f"[ImmunoctoHEDataset] {len(self.samples)} images, masks enabled "
                  f"({n_missing_mask_dirs} CRC folders missing a mask/ dir)")

    def _mask_path_for(self, img_path: Path) -> Path:
        # .../CRCxx/HE/foo.png -> .../CRCxx/mask/foo.png
        return img_path.parent.parent / "mask" / img_path.name

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")

        if not self.return_masks:
            if self.transform is not None:
                img = self.transform(img)
            return img, ()

        mask_path = self._mask_path_for(img_path)
        if not mask_path.exists():
            mask = Image.new("L", img.size, 0)
        else:
            mask_arr = (np.array(Image.open(mask_path).convert("L")) > 0).astype(np.uint8) * 255
            mask = Image.fromarray(mask_arr)

        if self.transform is not None:
            return self.transform(img, mask)
        return img, mask


# ==========================================================================
# PanNuke
# ==========================================================================

class PanNukeRawFoldDataset(Dataset):
    """
    Loads the OFFICIAL raw PanNuke distribution and crops a crop_size x
    crop_size patch around EVERY individual nucleus instance, returning
    (image_crop, mask_crop) pairs.

    Expected layout per fold (official TIA Lab layout):
        <root>/Fold 1/images.npy   (N, 256, 256, 3) uint8
        <root>/Fold 1/masks.npy    (N, 256, 256, 6) -- channels 0-4 = per-class
                                     instance maps (int instance IDs, 0=bg),
                                     channel 5 = tissue mask (unused)
        <root>/Fold 2/... , <root>/Fold 3/...
    """

    CATEGORY_NAMES = ["Neoplastic", "Inflammatory", "Connective", "Dead", "Epithelial"]
    N_CLASSES = 5
    CROP_SIZE = 64

    def __init__(self, root, folds=(1, 2, 3), fold_dir_template="Fold {n}",
                 transform=None, return_masks=True, crop_size=None):
        self.root = Path(root)
        self.transform = transform
        self.return_masks = return_masks
        self.crop_size = crop_size or self.CROP_SIZE

        self.fold_images, self.fold_masks = [], []
        self.samples = []  # (fold_idx, tile_idx, class_idx, instance_id)

        for fold_idx, n in enumerate(folds):
            fold_dir = self.root / fold_dir_template.format(n=n)
            if not fold_dir.is_dir():
                raise FileNotFoundError(
                    f"Fold directory not found: {fold_dir}\n"
                    f"Check root / fold_dir_template, or list what's actually there with: "
                    f"ls \"{self.root}\""
                )
            images_path = self._find_npy(fold_dir, "image")
            masks_path = self._find_npy(fold_dir, "mask")
            images = np.load(images_path, mmap_mode="r")
            masks = np.load(masks_path, mmap_mode="r")
            assert len(images) == len(masks), f"{fold_dir}: images/masks length mismatch"
            self.fold_images.append(images)
            self.fold_masks.append(masks)

            for tile_idx in range(len(masks)):
                mask_stack = np.array(masks[tile_idx])  # (256,256,6)
                for cls in range(self.N_CLASSES):
                    inst_map = mask_stack[..., cls]
                    for inst_id in np.unique(inst_map):
                        if inst_id == 0:
                            continue
                        self.samples.append((fold_idx, tile_idx, cls, int(inst_id)))

        if len(self.samples) == 0:
            raise RuntimeError(f"No nucleus instances found under {root}")

        print(f"[PanNukeRawFoldDataset] folds={folds}: {len(self.samples)} nucleus instances "
              f"across {sum(len(m) for m in self.fold_masks)} tiles"
              f"{' (with masks)' if return_masks else ''}")

    @staticmethod
    def _find_npy(fold_dir, name_hint):
        candidates = sorted(
            (p for p in Path(fold_dir).rglob("*.npy") if name_hint in p.name.lower()),
            key=lambda p: len(str(p)),
        )
        if not candidates:
            raise FileNotFoundError(
                f"Couldn't find a '*{name_hint}*.npy' file under {fold_dir}. "
                f"Check the actual layout with: find \"{fold_dir}\" -iname '*.npy'"
            )
        return candidates[0]

    def __len__(self):
        return len(self.samples)

    def _crop_around(self, arr, center_rc):
        H, W = arr.shape[:2]
        size = self.crop_size
        if H < size or W < size:
            return None
        cy, cx = center_rc
        half = size // 2
        top = max(0, min(int(round(cy - half)), H - size))
        left = max(0, min(int(round(cx - half)), W - size))
        return arr[top:top + size, left:left + size]

    def __getitem__(self, idx):
        fold_idx, tile_idx, cls, inst_id = self.samples[idx]
        img = np.array(self.fold_images[fold_idx][tile_idx]).astype(np.uint8)
        mask_stack = np.array(self.fold_masks[fold_idx][tile_idx])
        mask_bin = (mask_stack[..., cls] == inst_id).astype(np.uint8)

        ys, xs = np.nonzero(mask_bin)
        center = (float(ys.mean()), float(xs.mean())) if len(ys) else (128.0, 128.0)

        img_crop = self._crop_around(img, center)
        mask_crop = self._crop_around(mask_bin, center)
        if img_crop is None or mask_crop is None:  # tile smaller than crop_size -- shouldn't happen for PanNuke's 256px tiles
            img_crop = img[: self.crop_size, : self.crop_size]
            mask_crop = mask_bin[: self.crop_size, : self.crop_size]

        pil_img = Image.fromarray(img_crop)

        if not self.return_masks:
            if self.transform is not None:
                pil_img = self.transform(pil_img)
            return pil_img, ()

        pil_mask = Image.fromarray((mask_crop * 255).astype(np.uint8))
        if self.transform is not None:
            return self.transform(pil_img, pil_mask)  # MorphAwareDataAugmentationDINO.__call__(image, mask)
        return pil_img, pil_mask