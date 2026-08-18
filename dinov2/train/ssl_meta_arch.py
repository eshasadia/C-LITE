import logging
import math
from functools import partial
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


def pi_resize_patch_kernel(kernel, target_size):
    """FlexiViT-style PI-resize: reshapes a square conv kernel to
    target_size x target_size while preserving its function under resizing
    (not just its shape, unlike plain bicubic interpolation of the weights)."""
    out_ch, in_ch, kh, kw = kernel.shape
    assert kh == kw, "PI-resize as implemented here assumes a square kernel"
    if kh == target_size:
        return kernel
    src = torch.eye(kh * kh, device=kernel.device, dtype=kernel.dtype).reshape(kh * kh, 1, kh, kh)
    resized = nn.functional.interpolate(src, size=(target_size, target_size), mode="bilinear", align_corners=False)
    B = resized.reshape(kh * kh, target_size * target_size).T  # (target^2, kh^2)
    P = torch.linalg.pinv(B.T)  # (target^2, kh^2), the PI-resize transform matrix
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

        # --- morphology-aware structure block + FeatureSharpener -------------
        # NOTE: the struct token is *added* to the cls token (not concatenated),
        # so embed_dim is unchanged and dino_head/ibot_head above don't need a
        # doubled in_dim. This also keeps local-crop cls tokens (which never go
        # through the structure block -- see forward_backward) dimensionally
        # compatible with global-crop cls tokens inside the same xFormers
        # BlockDiagonalMask batch, and keeps the shared dino/ibot-head buffer
        # path (ibot_separate_head=False) intact.
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

        # --- morphology feats for the global crops, same batch order as
        # collated_global_crops (see collate_morph_data_and_cast) -----------
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

            # --- fuse in the morphology structure token, BEFORE the
            # chunk/reverse/cat below, so morph_feats (still in original
            # collated-crop order) lines up with teacher_cls_tokens/patch
            # tokens (also still in that order at this point) -------------
            if self.do_morphology:
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