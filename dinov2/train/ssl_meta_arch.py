import logging
import math
from functools import partial
import numpy as np
import torch
from dinov2.fsdp import ShardedGradScaler, get_fsdp_modules, get_fsdp_wrapper, reshard_fsdp_model
from dinov2.layers import DINOHead
from dinov2.layers.structure_block import StructureAttentionBlock
from dinov2.layers.feature_sharpener import FeatureSharpener
from dinov2.loss import DINOLoss, KoLeoLoss, iBOTPatchLoss
from dinov2.models import build_model_from_cfg
import timm
from dinov2.models.vision_transformer import BlockChunk
from dinov2.utils.param_groups import fuse_params_groups, get_params_groups_with_decay
from dinov2.utils.utils import has_batchnorms
from torch import nn
from dinov2.models import vision_transformer as vits
try:
    from xformers.ops import fmha
except ImportError:
    raise AssertionError("xFormers is required for training")


logger = logging.getLogger("dinov2")

# Same normalization the augmentation pipeline applies (ImageNet mean/std) --
# needed to denormalize crops back to viewable RGB for wandb logging.
_VIS_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
_VIS_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def smooth_rank_loss(embedding_matrix, eps=1e-7):
    """
    Compute a loss based on the smooth rank measure of a matrix of embeddings.
    This version is adapted for use as a loss function, where lower values are better.

    Args:
        embedding_matrix (torch.Tensor): Matrix of embeddings (n x m). n: number of patch embeddings, m: embedding dimension.

    Returns:
        torch.Tensor: A scalar tensor representing the loss.
    """
    # Ensure the embeddings are float type for SVD
    embedding_matrix = embedding_matrix.float()

    # Perform SVD on the embedding matrix
    _, S, _ = torch.svd(embedding_matrix)

    # Normalize the singular values to sum to 1, add eps to avoid division by zero in log

    p = S / (torch.norm(S, p=1) + eps)
    p = p[: embedding_matrix.shape[1]]
    # Compute the negative entropy of the distribution
    # This encourages the concentration of information (lower entropy is better for loss minimization)
    neg_entropy = -torch.exp(-torch.sum(p * torch.log(p)))  # Add eps to avoid log(0)

    # Return the negative entropy as the loss
    return neg_entropy


def interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))

    # Interpolate the position embeddings without changing the first row (class token)
    patch_pos_embed = nn.functional.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )

    # assert int(w0) == patch_pos_embed.shape[-2]
    # assert int(h0) == patch_pos_embed.shape[-1]
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

    # Concatenate the class token with the interpolated position embeddings
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)


def _bilinear_resize_matrix(src_size, dst_size, device, dtype, mode="bicubic", antialias=True):
    """Builds the linear operator B such that resizing a flattened src_size x
    src_size 'image' equals B @ flat_image. antialias=True matters here
    because this is almost always *downsampling* (e.g. 14 -> 8); without
    it, F.interpolate point-samples instead of properly box-filtering,
    which makes B (and thus PI-resize's pinv of B) noisy and unstable.
    Despite the function name (kept for backward compatibility with
    callers), the default mode is bicubic, not bilinear -- bicubic with
    antialiasing is what actually gets used unless overridden."""
    src = torch.eye(src_size * src_size, device=device, dtype=dtype).reshape(
        src_size * src_size, 1, src_size, src_size)
    resized = nn.functional.interpolate(src, size=(dst_size, dst_size), mode=mode,
                                         align_corners=False, antialias=antialias)
    B = resized.reshape(src_size * src_size, dst_size * dst_size).T  # (dst^2, src^2)
    return B


def pi_resize_patch_kernel(kernel, target_size):
    """FlexiViT-style PI-resize: reshapes a square conv kernel to
    target_size x target_size while preserving its function under resizing
    (not just its shape, unlike plain bicubic interpolation of the weights).
    Implicitly assumes patches are drawn from an isotropic (white-noise,
    Sigma=I) distribution -- if you have real patch statistics available,
    a covariance-weighted PI-resize gives a more accurate transplant, but
    this unweighted version is what's wired into the backbone loader below."""
    out_ch, in_ch, kh, kw = kernel.shape
    assert kh == kw, "PI-resize as implemented here assumes a square kernel"
    if kh == target_size:
        return kernel
    B = _bilinear_resize_matrix(kh, target_size, kernel.device, kernel.dtype)  # (target^2, kh^2)
    # rcond=1e-4 regularizes the pseudo-inverse against ill-conditioning --
    # without it, near-zero singular values in B (more likely without
    # antialiasing, but possible regardless) can blow up small numerical
    # noise into large, spurious values in the resized kernel.
    P = torch.linalg.pinv(B.T, rcond=1e-4)  # (target^2, kh^2), the PI-resize transform matrix
    flat = kernel.reshape(out_ch * in_ch, kh * kh)  # (out*in, kh^2)
    resized_flat = flat @ P.T  # (out*in, target^2)
    return resized_flat.reshape(out_ch, in_ch, target_size, target_size)


def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14", patch_size=14, img_size=224):
    """Downloads the pretrained checkpoint, then adapts it to a different
    patch_size / crop resolution:
      - PI-resizes patch_embed.proj's conv kernel (e.g. 14x14 -> 8x8) --
        required whenever cfg.student.patch_size differs from the
        checkpoint's native patch size, or the token grid this backbone
        actually produces silently won't match what the rest of the config
        (MaskingGenerator, structure block, morph_feat grid, etc.) expects.
      - Interpolates pos_embed to the resulting (img_size // patch_size)
        token grid, instead of the previous hardcoded 16x16 target.
    """
    model = torch.hub.load("facebookresearch/dinov2", modelname)

    old_conv = model.patch_embed.proj
    old_patch_size = old_conv.kernel_size[0]
    if patch_size != old_patch_size:
        logger.info(f"OPTIONS -- PI-resizing patch_embed kernel {old_patch_size}x{old_patch_size} "
                    f"-> {patch_size}x{patch_size}")
        new_weight = pi_resize_patch_kernel(old_conv.weight.data, patch_size)
        new_conv = nn.Conv2d(
            old_conv.in_channels, old_conv.out_channels,
            kernel_size=patch_size, stride=patch_size,
            bias=old_conv.bias is not None,
        )
        new_conv.weight.data = new_weight
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()
        model.patch_embed.proj = new_conv
        if hasattr(model.patch_embed, "patch_size"):
            model.patch_embed.patch_size = (patch_size, patch_size)
        if hasattr(model.patch_embed, "img_size"):
            model.patch_embed.img_size = (img_size, img_size)
        # CRITICAL: DinoVisionTransformer.interpolate_pos_encoding divides by
        # self.patch_size (a top-level model attribute, separate from
        # patch_embed.patch_size above) whenever the input token count
        # doesn't already match self.pos_embed's stored size -- i.e. for any
        # crop resolution other than the one pos_embed was initialized for
        # (global_crops_size here). Leaving this stale at the checkpoint's
        # original patch_size silently produces the wrong token grid for
        # every OTHER resolution (local crops), while global crops happen to
        # hit interpolate_pos_encoding's "already matches, return as-is"
        # shortcut and never exercise the broken math at all.
        if hasattr(model, "patch_size"):
            model.patch_size = patch_size

    grid_side = img_size // patch_size
    tensor_corr_shape = interpolate_pos_encoding(model.pos_embed, grid_side, grid_side)
    pos_embed = nn.Parameter(torch.zeros_like(tensor_corr_shape))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    logger.info(f"OPTIONS -- pos_embed interpolated to {grid_side}x{grid_side} = {grid_side * grid_side} patches")
    return model
def load_backbone_checkpoint(backbone: nn.Module, checkpoint_path: str):
    """
    Load backbone weights from multiple checkpoint formats:

      - do_test() eval checkpoints:
            {"teacher": {"backbone.xxx": ..., "dino_head.xxx": ...}}

      - FSDP training checkpoints:
            {"model": {"student.backbone.xxx": ...}}

      - student-only checkpoints:
            {"student": {"xxx": ...}}
            or {"student": {"backbone.xxx": ...}}

      - backbone-only checkpoints:
            {"backbone": {"xxx": ...}}
            (optionally with "_fsdp_wrapped_module." prefixes)

      - bare backbone state_dict

    Keys whose shapes do not match the current backbone are skipped.
    """

    logger.info(f"OPTIONS -- pretrained weights: loading from {checkpoint_path}")
    chkpt = torch.load(checkpoint_path, map_location="cpu")

    # ------------------------------------------------------------------
    # Extract the appropriate state dict depending on checkpoint format
    # ------------------------------------------------------------------
    if "teacher" in chkpt:
        raw_sd = chkpt["teacher"]

    elif "model" in chkpt:
        raw_sd = chkpt["model"]

    elif "student" in chkpt:
        raw_sd = chkpt["student"]

    elif "backbone" in chkpt:
        # Remove FSDP wrapper if present
        raw_sd = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in chkpt["backbone"].items()
        }

    else:
        raw_sd = chkpt

    # ------------------------------------------------------------------
    # Strip known prefixes to obtain a pure backbone state_dict
    # ------------------------------------------------------------------
    backbone_sd = {}
    for k, v in raw_sd.items():

        if k.startswith("teacher.backbone."):
            backbone_sd[k[len("teacher.backbone."):]] = v

        elif k.startswith("student.backbone."):
            backbone_sd[k[len("student.backbone."):]] = v

        elif k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v

        else:
            # Covers:
            #   {"student": {"blocks.0...": ...}}
            #   {"backbone": {"blocks.0...": ...}}
            #   bare backbone state_dict
            backbone_sd[k] = v

    target_sd = backbone.state_dict()

    # ------------------------------------------------------------------
    # Keep only tensors with matching names AND shapes
    # ------------------------------------------------------------------
    filtered_sd = {}
    skipped = []

    for k, v in backbone_sd.items():
        if k in target_sd and target_sd[k].shape == v.shape:
            filtered_sd[k] = v
        else:
            skipped.append(k)

    missing, unexpected = backbone.load_state_dict(filtered_sd, strict=False)

    logger.info(
        f"Loaded {len(filtered_sd)}/{len(backbone_sd)} backbone tensors "
        f"(skipped {len(skipped)} due to shape mismatch or unknown keys)."
    )

    if skipped:
        logger.info(f"Skipped keys (first 10): {skipped[:10]}")

    if missing:
        logger.info(
            f"Missing keys initialized randomly (first 10): {list(missing)[:10]}"
        )

    if unexpected:
        logger.info(
            f"Unexpected keys ignored (first 10): {list(unexpected)[:10]}"
        )

    if len(filtered_sd) == 0:
        logger.warning(
            "WARNING -- 0 tensors were loaded from the checkpoint. "
            "The backbone is effectively training from scratch. "
            "Check that the checkpoint architecture "
            "(embed_dim/depth/patch_size) matches the current model."
        )

class SSLMetaArch(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.fp16_scaler = ShardedGradScaler() if cfg.compute_precision.grad_scaler else None

        student_model_dict = dict()
        teacher_model_dict = dict()

        if cfg.student.arch in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14"]:
            student_backbone = get_downloaded_dino_vit_interpolated(
                cfg.student.arch, patch_size=cfg.student.patch_size, img_size=cfg.crops.global_crops_size
            )
            teacher_backbone = get_downloaded_dino_vit_interpolated(
                cfg.student.arch, patch_size=cfg.student.patch_size, img_size=cfg.crops.global_crops_size
            )
            embed_dict = {"dinov2_vits14": 384, "dinov2_vitb14": 768, "dinov2_vitl14": 1024, "dinov2_vitg14": 1536}
            embed_dim = embed_dict[cfg.student.arch]
        else:
            student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg)

        student_model_dict["backbone"] = student_backbone
        teacher_model_dict["backbone"] = teacher_backbone
        logger.info(f"OPTIONS -- architecture : embed_dim: {embed_dim}")

        if cfg.student.pretrained_weights:
            # Load into the student backbone. `prepare_for_distributed_training`
            # (called later, before FSDP wrapping) copies student -> teacher for
            # every submodule, so the teacher inherits these weights too -- no
            # need to load into teacher_backbone separately here.
            load_backbone_checkpoint(student_backbone, cfg.student.pretrained_weights)

        self.embed_dim = embed_dim
        self.dino_out_dim = cfg.dino.head_n_prototypes

        self.do_dino = cfg.dino.loss_weight > 0
        self.do_koleo = cfg.dino.koleo_loss_weight > 0
        self.do_ibot =True
        self.do_ibot = cfg.ibot.loss_weight > 0
        self.do_smooth_rank_loss = cfg.dino.smooth_rank_loss_weight > 0
        self.ibot_separate_head = cfg.ibot.separate_head

        # --- morphology / feature-sharpener options -------------------------
        self.do_morphology = cfg.morphology.enabled
        self.do_feature_sharpener = cfg.feature_sharpener.enabled

        logger.info("OPTIONS -- DINO")

        if self.do_dino:
            logger.info(f"OPTIONS -- DINO -- loss_weight: {cfg.dino.loss_weight}")
            logger.info(f"OPTIONS -- DINO -- head_n_prototypes: {cfg.dino.head_n_prototypes}")
            logger.info(f"OPTIONS -- DINO -- head_bottleneck_dim: {cfg.dino.head_bottleneck_dim}")
            logger.info(f"OPTIONS -- DINO -- head_hidden_dim: {cfg.dino.head_hidden_dim}")
            self.dino_loss_weight = cfg.dino.loss_weight
            dino_head = partial(
                DINOHead,
                in_dim=embed_dim,
                out_dim=cfg.dino.head_n_prototypes,
                hidden_dim=cfg.dino.head_hidden_dim,
                bottleneck_dim=cfg.dino.head_bottleneck_dim,
                nlayers=cfg.dino.head_nlayers,
            )
            self.dino_loss = DINOLoss(self.dino_out_dim)
            if self.do_koleo:
                logger.info("OPTIONS -- DINO -- applying KOLEO regularization")
                self.koleo_loss = KoLeoLoss()

        else:
            logger.info("OPTIONS -- DINO -- not using DINO")

        if self.do_dino or self.do_ibot:
            student_model_dict["dino_head"] = dino_head()
            teacher_model_dict["dino_head"] = dino_head()

        logger.info("OPTIONS -- IBOT")
        logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_ratio_tuple: {cfg.ibot.mask_ratio_min_max}")
        logger.info(f"OPTIONS -- IBOT masking -- ibot_mask_sample_probability: {cfg.ibot.mask_sample_probability}")
        if self.do_ibot:
            self.ibot_loss_weight = cfg.ibot.loss_weight
            assert max(cfg.ibot.mask_ratio_min_max) > 0, "please provide a positive mask ratio tuple for ibot"
            assert cfg.ibot.mask_sample_probability > 0, "please provide a positive mask probability for ibot"
            self.ibot_out_dim = cfg.ibot.head_n_prototypes if self.ibot_separate_head else cfg.dino.head_n_prototypes
            self.ibot_patch_loss = iBOTPatchLoss(self.ibot_out_dim)
            if self.ibot_separate_head:
                logger.info(f"OPTIONS -- IBOT -- loss_weight: {cfg.ibot.loss_weight}")
                logger.info(f"OPTIONS -- IBOT -- head_n_prototypes: {cfg.ibot.head_n_prototypes}")
                logger.info(f"OPTIONS -- IBOT -- head_bottleneck_dim: {cfg.ibot.head_bottleneck_dim}")
                logger.info(f"OPTIONS -- IBOT -- head_hidden_dim: {cfg.ibot.head_hidden_dim}")
                ibot_head = partial(
                    DINOHead,
                    in_dim=embed_dim,
                    out_dim=cfg.ibot.head_n_prototypes,
                    hidden_dim=cfg.ibot.head_hidden_dim,
                    bottleneck_dim=cfg.ibot.head_bottleneck_dim,
                    nlayers=cfg.ibot.head_nlayers,
                )
                student_model_dict["ibot_head"] = ibot_head()
                teacher_model_dict["ibot_head"] = ibot_head()
            else:
                logger.info("OPTIONS -- IBOT -- head shared with DINO")

        if self.do_morphology:
            student_model_dict["structure_block"] = StructureAttentionBlock(
                embed_dim, use_morphology_bias=(cfg.morphology.mode == "morph")
            )
            teacher_model_dict["structure_block"] = StructureAttentionBlock(
                embed_dim, use_morphology_bias=(cfg.morphology.mode == "morph")
            )
            logger.info(f"OPTIONS -- MORPHOLOGY -- enabled, mode: {cfg.morphology.mode}")

        if self.do_feature_sharpener:
            student_model_dict["sharpener"] = FeatureSharpener(
                dim=embed_dim, upsample_factor=cfg.feature_sharpener.upsample_factor
            )
            logger.info(f"OPTIONS -- FEATURE SHARPENER -- enabled, weight: {cfg.feature_sharpener.weight}")

        self.need_to_synchronize_fsdp_streams = True

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # there is no backpropagation through the teacher, so no need for gradients
        for p in self.teacher.parameters():
            p.requires_grad = False
        logger.info(f"Student and Teacher are built: they are both {cfg.student.arch} network.")

    def forward(self, inputs):
        raise NotImplementedError

    def backprop_loss(self, loss):
        if self.fp16_scaler is not None:
            self.fp16_scaler.scale(loss).backward()
        else:
            loss.backward()

    def forward_backward(self, images, teacher_temp):
        n_global_crops = 2
        assert n_global_crops == 2
        n_local_crops = self.cfg.crops.local_crops_number

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        local_crops = images["collated_local_crops"].cuda(non_blocking=True)

        masks = images["collated_masks"].cuda(non_blocking=True)
        mask_indices_list = images["mask_indices_list"].cuda(non_blocking=True)
        n_masked_patches_tensor = images["n_masked_patches"].cuda(non_blocking=True)
        n_masked_patches = mask_indices_list.shape[0]
        upperbound = images["upperbound"]
        masks_weight = images["masks_weight"].cuda(non_blocking=True)
        morph_feats = None
        if self.do_morphology:
            morph_feats = images["collated_global_morph_feats"].cuda(non_blocking=True)

        n_local_crops_loss_terms = max(n_local_crops * n_global_crops, 1)
        n_global_crops_loss_terms = (n_global_crops - 1) * n_global_crops

        do_dino = self.do_dino
        do_ibot = self.do_ibot

        # loss scales
        ibot_loss_scale = 1.0 / n_global_crops

        # teacher output
        @torch.no_grad()
        def get_teacher_output():
            x, n_global_crops_teacher = global_crops, n_global_crops
            teacher_backbone_output_dict = self.teacher.backbone(x, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]

            if self.do_morphology:
                assert morph_feats.shape[0] == ibot_teacher_patch_tokens.shape[0], (
                    f"[teacher] batch size mismatch: morph_feats {morph_feats.shape} vs "
                    f"patch_tokens {ibot_teacher_patch_tokens.shape}"
                )
                assert morph_feats.shape[1] == ibot_teacher_patch_tokens.shape[1], (
                    f"[teacher] patch count mismatch: morph_feats has {morph_feats.shape[1]} rows, "
                    f"patch_tokens has {ibot_teacher_patch_tokens.shape[1]} -- check that "
                    f"collated_global_morph_feats' per-sample patch grid (from "
                    f"MorphAwareDataAugmentationDINO, sized by global_crops_size // patch_size) "
                    f"actually matches the backbone's real patch grid for this crop size"
                )
                _, struct_tok_teacher, _, _ = self.teacher.structure_block(
                    ibot_teacher_patch_tokens, morph_feats
                )
                teacher_cls_tokens = teacher_cls_tokens + struct_tok_teacher

            teacher_cls_tokens = teacher_cls_tokens.chunk(n_global_crops_teacher)
            # watch out: these are chunked and cat'd in reverse so A is matched to B in the global crops dino loss
            teacher_cls_tokens = torch.cat((teacher_cls_tokens[1], teacher_cls_tokens[0]))
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            if do_ibot and not self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens : n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_cls_tokens_after_head = tokens_after_head[:n_cls_tokens]
                masked_teacher_patch_tokens_after_head = tokens_after_head[
                    n_cls_tokens : n_cls_tokens + n_masked_patches
                ]
            elif do_ibot and self.ibot_separate_head:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_patch_tokens_after_head = self.teacher.ibot_head(buffer_tensor_teacher)[
                    :n_masked_patches
                ]
            else:
                teacher_cls_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                masked_teacher_ibot_softmaxed_centered = None

            if self.cfg.train.centering == "centering":
                teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])
                self.dino_loss.update_center(teacher_cls_tokens_after_head)
                if do_ibot:
                    masked_teacher_patch_tokens_after_head = masked_teacher_patch_tokens_after_head.unsqueeze(0)
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.softmax_center_teacher(
                        masked_teacher_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    masked_teacher_ibot_softmaxed_centered = masked_teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_patch_loss.update_center(masked_teacher_patch_tokens_after_head[:n_masked_patches])

            elif self.cfg.train.centering == "sinkhorn_knopp":
                teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                    teacher_cls_tokens_after_head, teacher_temp=teacher_temp
                ).view(n_global_crops_teacher, -1, *teacher_cls_tokens_after_head.shape[1:])

                if do_ibot:
                    masked_teacher_ibot_softmaxed_centered = self.ibot_patch_loss.sinkhorn_knopp_teacher(
                        masked_teacher_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
            elif self.cfg.train.centering == "optimal_transport":
                    teacher_dino_softmaxed_centered_list = (
                        self.dino_loss.optimal_transport_teacher(
                            teacher_cls_tokens_after_head,
                            teacher_temp=teacher_temp,
                        )
                    ).view(
                        n_global_crops_teacher,
                        -1,
                        *teacher_cls_tokens_after_head.shape[1:]
                    )

                    if do_ibot:
                        masked_teacher_ibot_softmaxed_centered = (
                            self.ibot_patch_loss.optimal_transport_teacher(
                                masked_teacher_patch_tokens_after_head,
                                teacher_temp=teacher_temp,
                                n_masked_patches_tensor=n_masked_patches_tensor,
                            )
                        )
            else:
                raise NotImplementedError

            return teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered

        teacher_dino_softmaxed_centered_list, masked_teacher_ibot_softmaxed_centered = get_teacher_output()
        reshard_fsdp_model(self.teacher)

        loss_dict = {}

        loss_accumulator = 0  # for backprop
        student_global_backbone_output_dict, student_local_backbone_output_dict = self.student.backbone(
            [global_crops, local_crops], masks=[masks, None], is_training=True
        )

        inputs_for_student_head_list = []

        # 1a: local crops cls tokens
        # (local crops are never run through the structure block: morph_feats
        # were only computed for the global-crop mask geometry -- see
        # MorphAwareDataAugmentationDINO. Local cls tokens stay embed_dim,
        # same as global ones below, since fusion is additive not concat.)
        student_local_cls_tokens = student_local_backbone_output_dict["x_norm_clstoken"]
        inputs_for_student_head_list.append(student_local_cls_tokens.unsqueeze(0))

        # 1b: global crops cls tokens
        student_global_cls_tokens = student_global_backbone_output_dict["x_norm_clstoken"]
        ibot_student_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]

        if self.do_morphology:
            assert morph_feats.shape[0] == ibot_student_patch_tokens.shape[0], (
                f"[student] batch size mismatch: morph_feats {morph_feats.shape} vs "
                f"patch_tokens {ibot_student_patch_tokens.shape}"
            )
            assert morph_feats.shape[1] == ibot_student_patch_tokens.shape[1], (
                f"[student] patch count mismatch: morph_feats has {morph_feats.shape[1]} rows, "
                f"patch_tokens has {ibot_student_patch_tokens.shape[1]} -- check that "
                f"collated_global_morph_feats' per-sample patch grid (from "
                f"MorphAwareDataAugmentationDINO, sized by global_crops_size // patch_size) "
                f"actually matches the backbone's real patch grid for this crop size"
            )
            _, struct_tok_student, _, _ = self.student.structure_block(
                ibot_student_patch_tokens, morph_feats
            )
            student_global_cls_tokens = student_global_cls_tokens + struct_tok_student

        inputs_for_student_head_list.append(student_global_cls_tokens.unsqueeze(0))

        # 1c: global crops patch tokens
        if do_ibot:
            _dim = student_global_backbone_output_dict["x_norm_clstoken"].shape[-1]
            buffer_tensor_patch_tokens = ibot_student_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_patch_tokens[:n_masked_patches].copy_(
                torch.index_select(ibot_student_patch_tokens.flatten(0, 1), dim=0, index=mask_indices_list)
            )
            if not self.ibot_separate_head:
                inputs_for_student_head_list.append(buffer_tensor_patch_tokens.unsqueeze(0))
            else:
                student_global_masked_patch_tokens_after_head = self.student.ibot_head(buffer_tensor_patch_tokens)[
                    :n_masked_patches
                ]

        # 2: run
        _attn_bias, cat_inputs = fmha.BlockDiagonalMask.from_tensor_list(inputs_for_student_head_list)
        outputs_list = _attn_bias.split(self.student.dino_head(cat_inputs))

        # 3a: local crops cls tokens
        student_local_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3b: global crops cls tokens
        student_global_cls_tokens_after_head = outputs_list.pop(0).squeeze(0)

        # 3c: global crops patch tokens
        if do_ibot and not self.ibot_separate_head:
            student_global_masked_patch_tokens_after_head = outputs_list.pop(0).squeeze(0)[:n_masked_patches]

        if n_local_crops > 1:
            dino_local_crops_loss = self.dino_loss(
                student_output_list=student_local_cls_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / (n_global_crops_loss_terms + n_local_crops_loss_terms)

            # store for display
            loss_dict["dino_local_crops_loss"] = dino_local_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_local_crops_loss

        # process global crops
        loss_scales = 2  # this is here since we process global crops together

        if do_dino:
            # compute loss
            dino_global_crops_loss = (
                self.dino_loss(
                    student_output_list=[student_global_cls_tokens_after_head],
                    teacher_out_softmaxed_centered_list=[
                        teacher_dino_softmaxed_centered_list.flatten(0, 1)
                    ],  # these were chunked and stacked in reverse so A is matched to B
                )
                * loss_scales
                / (n_global_crops_loss_terms + n_local_crops_loss_terms)
            )

            loss_dict["dino_global_crops_loss"] = dino_global_crops_loss

            # accumulate loss
            loss_accumulator += self.dino_loss_weight * dino_global_crops_loss

            student_cls_tokens = student_global_cls_tokens

            if self.do_koleo:
                koleo_loss = self.cfg.dino.koleo_loss_weight * sum(
                    self.koleo_loss(p) for p in student_cls_tokens.chunk(2)
                )  # we don't apply koleo loss between cls tokens of a same image
                loss_accumulator += koleo_loss
                loss_dict["koleo_loss"] = (
                    koleo_loss / loss_scales
                )  # this is to display the same losses as before but we can remove eventually

            if self.do_smooth_rank_loss:
                smooth_rank_l = smooth_rank_loss(student_cls_tokens) * self.cfg.dino.smooth_rank_loss_weight
                loss_accumulator += smooth_rank_l
                loss_dict["smooth_rank_loss"] = smooth_rank_l / loss_scales

        if do_ibot:
            # compute loss
            ibot_patch_loss = (
                self.ibot_patch_loss.forward_masked(
                    student_global_masked_patch_tokens_after_head,
                    masked_teacher_ibot_softmaxed_centered,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                )
                * loss_scales
                * ibot_loss_scale
            )

            # store for display
            loss_dict["ibot_loss"] = ibot_patch_loss / 2

            # accumulate loss
            loss_accumulator += self.ibot_loss_weight * ibot_patch_loss

        # --- FeatureSharpener regularizer (student-only; operates on the raw
        # patch-token grid before the structure block, same as your standalone
        # script) -------------------------------------------------------------
        if self.do_feature_sharpener:
            B_fs, N_fs, D_fs = ibot_student_patch_tokens.shape
            side = int(round(N_fs ** 0.5))
            assert side * side == N_fs, (
                f"FeatureSharpener needs a square patch grid, got {N_fs} patches; "
                f"check crops.global_crops_size / student.patch_size"
            )
            grid = ibot_student_patch_tokens.transpose(1, 2).reshape(B_fs, D_fs, side, side)
            sharpened = self.student.sharpener(grid, global_crops)
            fs_loss = FeatureSharpener.down_consistency_loss(
                sharpened, grid
            ) + 0.1 * FeatureSharpener.edge_aware_smoothness_loss(sharpened, global_crops)
            loss_dict["fs_loss"] = fs_loss
            loss_accumulator += self.cfg.feature_sharpener.weight * fs_loss

        self.backprop_loss(loss_accumulator)

        self.fsdp_synchronize_streams()
        return loss_dict, student_cls_tokens
 
    def fsdp_synchronize_streams(self):
        if self.need_to_synchronize_fsdp_streams:
            torch.cuda.synchronize()
            for attr in {"_unshard_stream", "_post_backward_stream", "_pre_unshard_stream", "_all_reduce_stream", "_default_stream"}:
                stream = getattr(self.teacher.backbone, attr)
                setattr(self.student.dino_head, attr, stream)
                setattr(self.teacher.dino_head, attr, stream)
                setattr(self.student.backbone, attr, stream)
            self.need_to_synchronize_fsdp_streams = False

    @torch.no_grad()
    def get_morphology_visualization(self, images, n_samples=4):
        """Visualization-only, no-grad, NOT part of forward_backward -- safe
        to call at any point in the training loop (e.g. every N iterations)
        without affecting gradients, optimizer state, or the teacher EMA.

        Runs the student backbone + structure_block on the first n_samples
        global crops in this batch, purely to inspect what the morphology
        bias is doing.

        Returns None if cfg.morphology.enabled is False. Otherwise returns:
            crops_np:  (n, H, W, 3) uint8, denormalized RGB crops
            raw_mask:  (n, H, W) float32 in {0,1}, the FULL-RESOLUTION
                       nucleus mask (same H x W as the crop) -- or None if
                       the collate function wasn't updated to carry
                       collated_global_nucleus_masks through (older
                       collate_morph_data_and_cast without that key)
            morph_cov: (n, side, side) float32, the coarse 8x8-ish
                       patch-coverage channel actually fed to the model
                       (channel 0 of collated_global_morph_feats)
            attn_grid: (n, side, side) float32, the struct token's own
                       attention over patches (query=struct, key=patches)
            attn_mask_corr: (n,) float32, Pearson correlation between
                       attn_grid and morph_cov, flattened, per sample.
                       +1 = attention perfectly tracks coverage, -1 =
                       attention is exactly anti-correlated with coverage
                       (attends AWAY from the nucleus), 0 = no relationship.
                       This is the actual quantitative answer to "is
                       struct-token attention doing what it should" --
                       don't rely on eyeballing the panels for this, log
                       the mean of this array as a wandb scalar every step.
            lam: float, current value of exp(structure_block.log_lambda) --
                       the morphology bias strength. If this stays near
                       its init (1.0) while attn_mask_corr moves, the bias
                       term itself isn't what's changing; if lam grows
                       while attn_mask_corr stays flat or negative, the
                       bias is being outweighed by the learned
                       content-based attention term, which is possible
                       even with a strictly positive bias (see
                       structure_block.py's docstring on this).
        """
        if not self.do_morphology:
            return None

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        morph_feats = images["collated_global_morph_feats"].cuda(non_blocking=True)
        n = min(n_samples, global_crops.shape[0])

        crops = global_crops[:n]
        feats = morph_feats[:n]

        backbone_out = self.student.backbone(crops, is_training=True)
        patch_tokens = backbone_out["x_norm_patchtokens"]
        _, _, attn_map, _ = self.student.structure_block(patch_tokens, feats)

        side = int(round(attn_map.shape[1] ** 0.5))
        assert side * side == attn_map.shape[1], (
            f"structure_block attn_map has {attn_map.shape[1]} entries, not a perfect square"
        )
        attn_grid = attn_map.reshape(n, side, side).float().cpu().numpy()

        morph_side = int(round(feats.shape[1] ** 0.5))
        morph_cov = feats[..., 0].reshape(n, morph_side, morph_side).float().cpu().numpy()

        # Quantitative correlation, not just a picture -- computed on the
        # SAME resolution grid for both (side x side), no upsampling
        # involved, so this is exactly what the model itself sees, not an
        # artifact of how the panel happens to render.
        attn_flat = attn_grid.reshape(n, -1)
        cov_flat = morph_cov.reshape(n, -1)
        attn_c = attn_flat - attn_flat.mean(axis=1, keepdims=True)
        cov_c = cov_flat - cov_flat.mean(axis=1, keepdims=True)
        denom = np.sqrt((attn_c ** 2).sum(axis=1)) * np.sqrt((cov_c ** 2).sum(axis=1)) + 1e-8
        attn_mask_corr = (attn_c * cov_c).sum(axis=1) / denom  # (n,) in [-1, 1]

        lam = torch.exp(self.student.structure_block.log_lambda).item()

        raw_masks_tensor = images.get("collated_global_nucleus_masks", None)
        if raw_masks_tensor is not None:
            raw_mask = raw_masks_tensor[:n].float().cpu().numpy()  # (n, H, W)
        else:
            raw_mask = None

        mean = _VIS_IMAGENET_MEAN.to(crops.device)
        std = _VIS_IMAGENET_STD.to(crops.device)
        denorm = (crops * std + mean).clamp(0, 1)
        crops_np = (denorm.permute(0, 2, 3, 1).float().cpu().numpy() * 255).astype(np.uint8)

        return crops_np, raw_mask, morph_cov, attn_grid, attn_mask_corr, lam

    @torch.no_grad()
    def get_backbone_attention_visualization(self, images, n_samples=4, block_index=-1):
        """Visualization-only, no-grad. Captures the backbone's OWN CLS
        token attention over patches -- the plain ViT attention, unrelated
        to structure_block -- from block `block_index` (default: -1, the
        last block).

        The backbone's attention layers are MemEffAttention (xFormers'
        fused memory_efficient_attention kernel), which never materializes
        a dense (B, heads, N, N) attention tensor -- that's the entire
        point of the fused kernel, it skips building it. To capture real
        attention weights, this temporarily flips the module-level
        XFORMERS_AVAILABLE flag off for the duration of ONE extra forward
        pass, which makes every MemEffAttention instance fall back to
        plain Attention.forward (which supports the return_attention /
        last_attn hook). The flag is restored immediately after in a
        finally block -- this has zero effect on the real training
        forward passes, only this separate diagnostic call.

        Returns None if the backbone exposes no Attention/MemEffAttention
        instances (shouldn't happen, but guards against an unexpected
        architecture) or if block_index is out of range. Otherwise
        returns cls_attn_grid: (n, side, side) float32 -- the CLS token's
        attention over patches at the requested block, averaged over
        heads.
        """
        from dinov2.layers.attention import Attention
        import dinov2.layers.attention as attention_module

        global_crops = images["collated_global_crops"].cuda(non_blocking=True)
        n = min(n_samples, global_crops.shape[0])
        crops = global_crops[:n]

        attn_modules = [m for m in self.student.backbone.modules() if isinstance(m, Attention)]
        if not attn_modules:
            logger.warning("no Attention/MemEffAttention modules found in backbone -- skipping")
            return None
        try:
            target_module = attn_modules[block_index]
        except IndexError:
            logger.warning(f"block_index {block_index} out of range for {len(attn_modules)} attention layers")
            return None

        original_flag = attention_module.XFORMERS_AVAILABLE
        attention_module.XFORMERS_AVAILABLE = False
        for m in attn_modules:
            m.return_attention = True
            m.last_attn = None
        try:
            backbone_out = self.student.backbone(crops, is_training=True)
        finally:
            attention_module.XFORMERS_AVAILABLE = original_flag
            for m in attn_modules:
                m.return_attention = False

        attn = target_module.last_attn  # (B, heads, total_tokens, total_tokens)
        for m in attn_modules:
            m.last_attn = None  # free memory promptly, these are full dense tensors
        if attn is None:
            logger.warning("last_attn was not populated -- backbone forward may not have gone through this module")
            return None

        # Token layout is [CLS, register tokens (if any), patch tokens...],
        # but how many register tokens actually exist in THIS backbone
        # instance is not reliably knowable from config: cfg.train and
        # cfg.student both have a num_register_tokens field (different
        # values in practice), and the get_downloaded_dino_vit_interpolated
        # loader used for dinov2_vits14/etc. doesn't implement register
        # tokens at all regardless of what either config value says. Rather
        # than guess which config key is right, derive the CLS+register
        # prefix length directly from the SAME forward pass's own patch
        # token count -- x_norm_patchtokens is exactly the patch count
        # used everywhere else in this codebase (structure_block, etc.),
        # so whatever's left over in attn's last dim after that must be
        # the prefix (CLS + any registers), whatever that count actually is.
        n_patches = backbone_out["x_norm_patchtokens"].shape[1]
        prefix_len = attn.shape[-1] - n_patches
        assert prefix_len >= 1, (
            f"attn has {attn.shape[-1]} tokens total but x_norm_patchtokens reports "
            f"{n_patches} patches -- prefix_len={prefix_len} is impossible (must be >=1 for CLS alone)"
        )
        cls_attn = attn[:, :, 0, prefix_len:].mean(1)  # mean over heads -> (B, N_patches)

        side = int(round(cls_attn.shape[1] ** 0.5))
        assert side * side == cls_attn.shape[1], (
            f"CLS attention has {cls_attn.shape[1]} patch entries (attn total={attn.shape[-1]}, "
            f"prefix_len={prefix_len}, n_patches={n_patches}), not a perfect square"
        )
        return cls_attn.reshape(n, side, side).float().cpu().numpy()

    def update_teacher(self, m):
        student_param_list = []
        teacher_param_list = []
        with torch.no_grad():
            for k in self.student.keys():
                for ms, mt in zip(get_fsdp_modules(self.student[k]), get_fsdp_modules(self.teacher[k])):
                    student_param_list += ms.params
                    teacher_param_list += mt.params
            torch._foreach_mul_(teacher_param_list, m)
            torch._foreach_add_(teacher_param_list, student_param_list, alpha=1 - m)

    def train(self):
        super().train()
        self.teacher.eval()

    def get_maybe_fused_params_for_submodel(self, m):
        params_groups = get_params_groups_with_decay(
            model=m,
            lr_decay_rate=self.cfg.optim.layerwise_decay,
            patch_embed_lr_mult=self.cfg.optim.patch_embed_lr_mult,
        )
        fused_params_groups = fuse_params_groups(params_groups)
        logger.info("fusing param groups")

        for g in fused_params_groups:
            g["foreach"] = True
        return fused_params_groups

    def get_params_groups(self):
        all_params_groups = []
        for m in self.student.values():
            all_params_groups += self.get_maybe_fused_params_for_submodel(m)
        return all_params_groups

    def prepare_for_distributed_training(self):
        logger.info("DISTRIBUTED FSDP -- preparing model for distributed training")
        if has_batchnorms(self.student):
            raise NotImplementedError
        # below will synchronize all student subnetworks across gpus:
        for k, v in self.student.items():
            self.teacher[k].load_state_dict(self.student[k].state_dict())
            student_model_cfg = self.cfg.compute_precision.student[k]
            self.student[k] = get_fsdp_wrapper(student_model_cfg, modules_to_wrap={BlockChunk})(self.student[k])
            teacher_model_cfg = self.cfg.compute_precision.teacher[k]
            self.teacher[k] = get_fsdp_wrapper(teacher_model_cfg, modules_to_wrap={BlockChunk})(self.teacher[k])

    @staticmethod
    def interpolate_pos_encoding(x, w, h):
        # from Benedikt Roth, adapted from interpolate_pos_encoding in dinov2/dinov2/models/vision_transformer.py
        N = x.shape[1] - 1
        dim = x.shape[-1]
        w0 = w / int(math.sqrt(N))
        h0 = h / int(math.sqrt(N))

        # Interpolate the position embeddings without changing the first row (class token)
        patch_pos_embed = nn.functional.interpolate(
            x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0, h0),
            mode="bicubic",
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)

        # Concatenate the class token with the interpolated position embeddings
        return torch.cat((x[:, :1], patch_pos_embed), dim=1)