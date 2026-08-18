# dinov2/layers/feature_sharpener.py
import torch.nn as nn
import torch.nn.functional as F


class FeatureSharpener(nn.Module):
    def __init__(self, dim, upsample_factor=4, guidance_chans=3, guidance_dim=32):
        super().__init__()
        self.factor = upsample_factor
        self.guidance = nn.Sequential(
            nn.Conv2d(guidance_chans, guidance_dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(guidance_dim, guidance_dim, 3, padding=1),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(dim + guidance_dim, dim, 3, padding=1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, feat_grid, image):
        B, D, H, W = feat_grid.shape
        th, tw = H * self.factor, W * self.factor
        coarse_up = F.interpolate(feat_grid, size=(th, tw), mode="bilinear", align_corners=False)
        guidance_full = self.guidance(F.interpolate(image, size=(th, tw), mode="bilinear", align_corners=False))
        return self.refine(torch.cat([coarse_up, guidance_full], dim=1)) + coarse_up

    @staticmethod
    def down_consistency_loss(sharpened, feat_grid_orig):
        pooled = F.adaptive_avg_pool2d(sharpened, feat_grid_orig.shape[-2:])
        return F.mse_loss(pooled, feat_grid_orig.detach())

    @staticmethod
    def edge_aware_smoothness_loss(sharpened, image):
        img = F.interpolate(image, size=sharpened.shape[-2:], mode="bilinear", align_corners=False)
        img_dx = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]), dim=1, keepdim=True)
        img_dy = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]), dim=1, keepdim=True)
        feat_dx = torch.mean(torch.abs(sharpened[:, :, :, 1:] - sharpened[:, :, :, :-1]), dim=1, keepdim=True)
        feat_dy = torch.mean(torch.abs(sharpened[:, :, 1:, :] - sharpened[:, :, :-1, :]), dim=1, keepdim=True)
        return (feat_dx * torch.exp(-img_dx)).mean() + (feat_dy * torch.exp(-img_dy)).mean()