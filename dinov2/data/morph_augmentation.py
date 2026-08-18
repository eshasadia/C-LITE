# dinov2/data/morph_augmentations.py
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF

from .augmentations import DataAugmentationDINO


def _sample_resized_crop_params(img, scale, ratio=(3.0 / 4.0, 4.0 / 3.0)):
    return transforms.RandomResizedCrop.get_params(img, scale=scale, ratio=ratio)


class MorphAwareDataAugmentationDINO(DataAugmentationDINO):
    """Same crop/flip/color-jitter recipe as DataAugmentationDINO, but replays
    every geometric transform on the paired instance mask and returns the
    resulting morph_feat (coverage + centroid-offset per patch) alongside
    each crop. Color jitter is NOT applied to the mask (geometry only)."""

    def __init__(self, *args, patch_size=14, **kwargs):
        super().__init__(*args, **kwargs)
        self.patch_size = patch_size

    def _paired_crop(self, image, mask, size, scale, flip_p=0.5):
        i, j, h, w = _sample_resized_crop_params(image, scale=scale)
        img_c = TF.resized_crop(image, i, j, h, w, [size, size], TF.InterpolationMode.BICUBIC)
        mask_c = TF.resized_crop(mask, i, j, h, w, [size, size], TF.InterpolationMode.NEAREST)
        if torch.rand(1).item() < flip_p:
            img_c = TF.hflip(img_c)
            mask_c = TF.hflip(mask_c)
        return img_c, mask_c

    def _morph_feat_from_mask(self, mask_pil, proc_size):
        n_side = proc_size // self.patch_size
        m = (np.array(mask_pil) > 127).astype(np.float32)
        ys, xs = np.nonzero(m)
        cy, cx = (ys.mean(), xs.mean()) if len(ys) else (proc_size / 2, proc_size / 2)
        feats = np.zeros((n_side, n_side, 3), dtype=np.float32)
        for pi in range(n_side):
            for pj in range(n_side):
                r0, r1 = pi * self.patch_size, (pi + 1) * self.patch_size
                c0, c1 = pj * self.patch_size, (pj + 1) * self.patch_size
                coverage = m[r0:r1, c0:c1].mean()
                pcy, pcx = (r0 + r1) / 2, (c0 + c1) / 2
                feats[pi, pj] = [coverage, (pcy - cy) / proc_size, (pcx - cx) / proc_size]
        return torch.from_numpy(feats.reshape(n_side * n_side, 3))

    def __call__(self, image, mask):
        # mask: single-channel PIL image, same H×W as `image`, nonzero == nucleus
        output = {}

        im1_base, m1_base = self._paired_crop(image, mask, self.global_crops_size, self.global_crops_scale)
        im2_base, m2_base = self._paired_crop(image, mask, self.global_crops_size, self.global_crops_scale)
        output["global_crops"] = [self.global_transfo1(im1_base), self.global_transfo2(im2_base)]
        output["global_crops_teacher"] = output["global_crops"]
        output["global_morph_feats"] = [
            self._morph_feat_from_mask(m1_base, self.global_crops_size),
            self._morph_feat_from_mask(m2_base, self.global_crops_size),
        ]

        local_crops, local_morph_feats = [], []
        for _ in range(self.local_crops_number):
            im_c, m_c = self._paired_crop(image, mask, self.local_crops_size, self.local_crops_scale)
            local_crops.append(self.local_transfo(im_c))
            local_morph_feats.append(self._morph_feat_from_mask(m_c, self.local_crops_size))
        output["local_crops"] = local_crops
        output["local_morph_feats"] = local_morph_feats
        output["offsets"] = ()
        return output