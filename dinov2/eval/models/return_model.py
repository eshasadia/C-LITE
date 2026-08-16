# import re
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import resnet
# from transformers import AutoImageProcessor, BeitFeatureExtractor, Data2VecVisionModel, ViTModel
# from transformers import AutoImageProcessor, AutoModel
# import timm
# from timm.layers import SwiGLUPacked
# import math
# import torch
# from torch import nn
# try:
#     from segment_anything import sam_model_registry
# except ImportError:
#      sam_model_registry = None

    
# # def _interpolate_pos_encoding(x, w, h):
# #     N = x.shape[1] - 1
# #     dim = x.shape[-1]
# #     w0 = w / int(math.sqrt(N))
# #     h0 = h / int(math.sqrt(N))
# #     patch_pos_embed = nn.functional.interpolate(
# #         x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
# #         scale_factor=(w0, h0),
# #         mode="bicubic",
# #     )
# #     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
# #     return torch.cat((x[:, :1], patch_pos_embed), dim=1)

# # def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14"):
# #     model = torch.hub.load("facebookresearch/dinov2", modelname)
# #     tensor_corr_shape = _interpolate_pos_encoding(model.pos_embed, 16, 16)
# #     pos_embed = nn.Parameter(torch.zeros(1, 257))
# #     pos_embed.data = tensor_corr_shape
# #     model.pos_embed = pos_embed
# #     return model
# # # Try to import local modules
# try:
#     from models.resnet_retccl import resnet50 as retccl_res50
# except ImportError:
#     retccl_res50 = None

# try:
#     from models.lemon_loader import get_lemon_vit
# except ImportError:
#     get_lemon_vit = None

# try:
#     from dinov2.models.vision_transformer import DinoVisionTransformer
# except ImportError:
#     DinoVisionTransformer = None



# # def get_models(modelname, saved_model_path=None):
# #     """
# #     Main function to load various pre-trained models for feature extraction.
    
# #     Args:
# #         modelname: Name of the model to load
# #         saved_model_path: Path to checkpoint (required for some models)
    
# #     Returns:
# #         PyTorch model ready for feature extraction
# #     """
# #     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# #     # --- histology-pretrained models
# #     if modelname.lower() == "retccl":
# #         model = get_retCCL(saved_model_path)
# #     elif modelname.lower() == "lemon":
# #         if get_lemon_vit is None:
# #             raise ImportError("LEMON loader not available. Ensure models/lemon_loader.py exists.")
# #         model = get_lemon_vit(saved_model_path, model_name="vits8", img_size=40)
# #     elif modelname.lower() == "owkin":
# #         model = Phikon()
# #     elif modelname.lower() == "virchow":
# #         model = Virchow()
# #     elif modelname.lower() == "uni2":
# #         model = UNI2()
# #     elif modelname.lower() == "cellvitpp":
# #         model = CellViTPlusPlus(saved_model_path, input_size=1024)
# #     # --- vision foundation models
# #     elif modelname.lower() == "resnet50":
# #         model = get_res50()
# #     elif modelname.lower() == "resnet50_full":
# #         model = get_full_res50()
# #     elif modelname.lower() == "beit_fb":
# #         model = BeitModel(device)
# #     elif modelname.lower() == "h0_mini":
# #         model = H0mini(device)
# #     elif modelname.lower() == "distilled_vit_small":
# #         model = get_distilled_vit_small(saved_model_path)
# #     # --- our finetuned models
# #     elif modelname.lower() in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_ssl"]:
# #         model = get_dino_finetuned_downloaded(saved_model_path, modelname)
# #     elif modelname.lower() == "celldino":
# #         if DinoVisionTransformer is None:
# #             raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
# #         model = get_celldino_finetuned_downloaded(saved_model_path, modelname)
# #     elif modelname.lower() in ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vitg16"]:
# #         model = get_dinov3_finetuned_downloaded(saved_model_path, modelname)
# #     else:
# #         raise ValueError(f"Model {modelname} not found")

# #     model = model.to(device)
# #     model.eval()

# #     return model


# # # ============================================================================
# # # DISTILLED VIT-SMALL LOADER (NEW)
# # # ============================================================================
# # def get_distilled_vit_small(model_path):
# #         """
# #         Load the distilled ViT-Small/14 model (multi-teacher KD: UNI2, Virchow2,
# #         H-optimus-0 -> ViT-S/14, from train_knowledgedistill.py).

# #         Accepts checkpoints in these formats:
# #         1. {"student": {"key": value, ...}}            -- do_test() eval snapshot,
# #                                                             RECOMMENDED: backbone only,
# #                                                             no distillation-head or
# #                                                             optimizer state to strip.
# #         2. {"model": {"student_backbone.key": value,    -- full training
# #                         "distill_head.key": value, ...},     checkpoint_*.pth
# #             "optimizer": {...}, "iteration": N}             (save_checkpoint()).
# #                                                             distill_head.* keys are
# #                                                             discarded; only the
# #                                                             backbone is loaded.
# #         3. {"teacher": {"backbone.key": value, ...}}    -- older single-teacher
# #         4. {"backbone": {"key": value, ...}}               KD experiment formats,
# #         5. Plain state_dict {"key": value, ...}            kept for backward compat.

# #         Returns a timm ViT-Small/14 (DINOv2-style) model ready for feature
# #         extraction (CLS token).
# #         """
# #         ARCH = "vit_small_patch14_dinov2.lvd142m"
# #         IMG_SIZE = 224

# #         if model_path is None:
# #             print(f"No checkpoint provided, loading ImageNet-pretrained {ARCH}")
# #             model = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=IMG_SIZE)
# #             return model

# #         pretrained = torch.load(model_path, map_location="cpu", weights_only=False)

# #         # Extract state_dict from various checkpoint formats
# #         if isinstance(pretrained, dict):
# #             if "student" in pretrained:
# #                 # do_test() eval snapshot: {"student": student_backbone.state_dict()}
# #                 # -- already backbone-only, no head/optimizer keys to strip.
# #                 state_dict = pretrained["student"]
# #                 print(f"Loaded from 'student' key, {len(state_dict)} params")

# #             elif "teacher" in pretrained:
# #                 # Format: {"teacher": {"backbone.cls_token": ..., ...}}
# #                 raw_state_dict = pretrained["teacher"]
# #                 state_dict = {}
# #                 for key, value in raw_state_dict.items():
# #                     if "dino_head" in key or "ibot_head" in key:
# #                         continue
# #                     new_key = key.replace("backbone.", "")
# #                     state_dict[new_key] = value
# #                 print(f"Loaded from 'teacher' key, {len(state_dict)} params")

# #             elif "backbone" in pretrained:
# #                 # Format: {"backbone": {"cls_token": ..., ...}, "classifier": ...}
# #                 state_dict = pretrained["backbone"]
# #                 print(f"Loaded from 'backbone' key, {len(state_dict)} params")

# #             elif "state_dict" in pretrained:
# #                 state_dict = pretrained["state_dict"]
# #                 print(f"Loaded from 'state_dict' key, {len(state_dict)} params")

# #             elif "model" in pretrained:
# #                 # save_checkpoint() full training checkpoint:
# #                 # {"model": DistillationModel.state_dict(), "optimizer": ..., "iteration": ...}
# #                 # DistillationModel.state_dict() keys are prefixed "student_backbone."
# #                 # and "distill_head." -- keep only the backbone, drop the head, and
# #                 # strip the prefix so keys match the plain timm model.
# #                 raw_state_dict = pretrained["model"]
# #                 state_dict = {}
# #                 for key, value in raw_state_dict.items():
# #                     if not key.startswith("student_backbone."):
# #                         continue  # discards every distill_head.* key
# #                     state_dict[key[len("student_backbone."):]] = value
# #                 print(f"Loaded from 'model' key (student_backbone.* only), {len(state_dict)} params")

# #             else:
# #                 # Assume it's a plain state_dict
# #                 state_dict = pretrained
# #                 print(f"Loaded as plain state_dict, {len(state_dict)} params")
# #         else:
# #             state_dict = pretrained
# #             print(f"Loaded as plain state_dict, {len(state_dict)} params")

# #         # Remove any remaining prefixes (module.)
# #         clean_state_dict = {}
# #         for key, value in state_dict.items():
# #             new_key = key.replace("module.", "")
# #             clean_state_dict[new_key] = value

# #         # Infer embed_dim from checkpoint (sanity check, not architecture selection --
# #         # ARCH is fixed above to match training exactly, patch14 not patch16)
# #         if "patch_embed.proj.weight" in clean_state_dict:
# #             patch_weight = clean_state_dict["patch_embed.proj.weight"]
# #             embed_dim, in_chans, patch_h, patch_w = patch_weight.shape
# #             print(f"Checkpoint patch_embed: embed_dim={embed_dim}, patch_size={patch_h}, in_chans={in_chans}")
# #             if patch_h != 14:
# #                 print(f"[warn] checkpoint patch_size={patch_h} != expected 14 for {ARCH} -- "
# #                     f"this checkpoint may not match the distilled ViT-S/14 architecture, "
# #                     f"expect a shape-mismatch RuntimeError below if so")

# #         # Build timm model matching training exactly
# #         model = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=IMG_SIZE)

# #         # Load weights
# #         msg = model.load_state_dict(clean_state_dict, strict=False)

# #         print(f"\n==== LOAD REPORT (distilled_vit_small) ====")
# #         if msg.missing_keys:
# #             print(f"Missing keys ({len(msg.missing_keys)}):")
# #             for k in msg.missing_keys:
# #                 print(f"    {k}")
# #         else:
# #             print("Missing keys: none")

# #         if msg.unexpected_keys:
# #             print(f"Unexpected keys ({len(msg.unexpected_keys)}):")
# #             for k in msg.unexpected_keys:
# #                 print(f"    {k}")
# #         else:
# #             print("Unexpected keys: none")

# #         return model


# # def get_retCCL(model_path):
# #     """Load RetCCL ResNet50 model"""
# #     if retccl_res50 is None:
# #         raise ImportError("RetCCL model not available")
    
# #     model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
# #     pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
# #     model.fc = nn.Identity()
# #     model.load_state_dict(pretext_model, strict=True)
# #     return model

# # def get_dino_finetuned_downloaded(saved_model_path, modelname="dinov2_vits14"):
# #     model = get_downloaded_dino_vit_interpolated(modelname)
# #     pretrained = torch.load(saved_model_path, map_location="cpu")

# #     # Support multiple checkpoint formats:
# #     # 1. old EMA-teacher format:  {"teacher": {"backbone.xxx": ...}}
# #     # 2. new virchow-distill format: {"backbone": {"xxx": ...}}
# #     #    (optionally with FSDP "_fsdp_wrapped_module." prefix)
# #     if "teacher" in pretrained:
# #         state_dict = {
# #             k.replace("backbone.", ""): v
# #             for k, v in pretrained["teacher"].items()
# #             if k.startswith("backbone.")
# #         }
# #     elif "student" in pretrained:
# #         state_dict = pretrained["student"]
# #         # state_dict = {
# #         #     k.replace("backbone.", ""): v
# #         #     for k, v in pretrained["student"].items()
# #         #     if k.startswith("backbone.")
# #         # }
# #     elif "backbone" in pretrained:
# #         state_dict = {
# #             k.replace("_fsdp_wrapped_module.", ""): v
# #             for k, v in pretrained["backbone"].items()
# #         }
# #     else:
# #         raise KeyError(
# #             f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'backbone' key. "
# #             f"Available keys: {list(pretrained.keys())}"
# #         )

# #     missing, unexpected = model.load_state_dict(state_dict, strict=False)
# #     if missing:
# #         print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
# #     if unexpected:
# #         print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

# #     model.eval()
# #     return model

# # def get_celldino_finetuned_downloaded(saved_model_path, modelname):
# #     """Load CellDINO model with automatic architecture inference"""
# #     pretrained = torch.load(saved_model_path, map_location="cpu")

# #     # Extract state_dict safely
# #     if "teacher" in pretrained:
# #         state_dict = pretrained["teacher"]
# #     elif "student" in pretrained:
# #         state_dict = pretrained["student"]
# #     elif "state_dict" in pretrained:
# #         state_dict = pretrained["state_dict"]
# #     elif "model" in pretrained:
# #         state_dict = pretrained["model"]
# #     else:
# #         state_dict = pretrained

# #     # Remove DDP/module prefixes
# #     clean_state_dict = {}
# #     for key, value in state_dict.items():
# #         new_key = key.replace("module.", "")
# #         new_key = new_key.replace("backbone.", "")
# #         clean_state_dict[new_key] = value

# #     # Infer architecture from checkpoint
# #     patch_weight = clean_state_dict["patch_embed.proj.weight"]
# #     embed_dim, in_chans, patch_h, patch_w = patch_weight.shape

# #     assert patch_h == patch_w, f"Non-square patch embedding found: {patch_h}x{patch_w}"
# #     patch_size = patch_h

# #     print(f"Inferred from checkpoint:")
# #     print(f"  embed_dim  = {embed_dim}")
# #     print(f"  in_chans   = {in_chans}")
# #     print(f"  patch_size = {patch_size}")

# #     # Infer model depth/heads from embed_dim
# #     if embed_dim == 384:
# #         depth = 12
# #         num_heads = 6
# #     elif embed_dim == 768:
# #         depth = 12
# #         num_heads = 12
# #     elif embed_dim == 1024:
# #         depth = 24
# #         num_heads = 16
# #     elif embed_dim == 1536:
# #         depth = 40
# #         num_heads = 24
# #     else:
# #         raise ValueError(f"Unsupported embed_dim={embed_dim}. Please specify architecture manually.")

# #     # Infer image size from pos_embed
# #     img_size = 128  # fallback default

# #     if "pos_embed" in clean_state_dict:
# #         num_tokens = clean_state_dict["pos_embed"].shape[1]
# #         num_patches = num_tokens - 1
# #         grid_size = int(num_patches ** 0.5)

# #         if grid_size * grid_size == num_patches:
# #             img_size = grid_size * patch_size
# #             print(f"Inferred img_size = {img_size} from pos_embed")
# #         else:
# #             print("Could not infer square image size from pos_embed; using fallback img_size=128")

# #     # Build compatible DinoVisionTransformer
# #     model = DinoVisionTransformer(
# #         img_size=img_size,
# #         patch_size=patch_size,
# #         in_chans=in_chans,
# #         embed_dim=embed_dim,
# #         depth=depth,
# #         num_heads=num_heads,
# #         mlp_ratio=4.0,
# #         qkv_bias=True,
# #         ffn_bias=True,
# #         proj_bias=True,
# #     )

# #     # Remove classifier/incompatible heads
# #     keys_to_remove = []
# #     for k in clean_state_dict.keys():
# #         if any(x in k for x in ["head", "classifier", "fc", "dino_head", "ibot_head"]):
# #             keys_to_remove.append(k)

# #     for k in keys_to_remove:
# #         clean_state_dict.pop(k, None)

# #     # Load weights safely
# #     if "pos_embed" in clean_state_dict:
# #         if clean_state_dict["pos_embed"].shape != model.pos_embed.shape:
# #             print(
# #                 f"Skipping pos_embed due to shape mismatch: "
# #                 f"{clean_state_dict['pos_embed'].shape} vs {model.pos_embed.shape}"
# #             )
# #             clean_state_dict.pop("pos_embed")

# #     msg = model.load_state_dict(clean_state_dict, strict=False)
# #     print("Load result:", msg)

# #     return model


# # def get_dinov3_finetuned_downloaded(model_path, modelname):
# #     """Load DINOv3 model"""
# #     pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
# #     processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
# #     model = AutoModel.from_pretrained(
# #         pretrained_model_name, 
# #         device_map="auto"
# #     )
# #     return model


# # def get_res50():
# #     """Load ResNet50 with early stopping (remove last residual block)"""
# #     model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

# #     class Reshape(nn.Module):
# #         def forward(self, x):
# #             return x.reshape(x.shape[0], -1)

# #     # Delete last res block as this has been shown to work better
# #     model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

# #     return model


# # def get_full_res50():
# #     """Load full ResNet50"""
# #     model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")
# #     return model


# # # ============================================================================
# # # TRANSFORMS
# # # ============================================================================

# # def get_transforms(model_name):
# #     """
# #     Get appropriate image transforms for each model.
# #     """
# #     # Default ImageNet normalization
# #     mean = (0.485, 0.456, 0.406)
# #     std = (0.229, 0.224, 0.225)

# #     if model_name.lower() in ["ctranspath", "resnet50", "beit_fb", "resnet50_full"]:
# #         size = 224
# #     elif model_name.lower() == "lemon":
# #         size = 40
# #     elif model_name.lower() == "owkin":
# #         image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
# #         mean, std = image_processor.image_mean, image_processor.image_std
# #         size = image_processor.size["height"]
# #     elif model_name.lower() == "retccl":
# #         size = 256
# #     elif model_name.lower() == "kimianet":
# #         size = 1000
# #     elif model_name.lower() == "cellvitpp":
# #         size = 1024
# #     elif model_name.lower() == "imagebind":
# #         size = 224
# #         mean = (0.48145466, 0.4578275, 0.40821073)
# #         std = (0.26862954, 0.26130258, 0.27577711)
# #     elif model_name.lower() in [
# #         "dinov2_vits14",
# #         "dinov2_ssl",
# #         "dinov3_vits16",
# #         "dinov2_vitb14",
# #         "dinov2_vitl14",
# #         "dinov2_vitg14",
# #         "dinov2_finetuned",
# #         "dinov2_vits14_interpolated",
# #         "dinov2_finetuned_downloaded",
# #         "remedis",
# #         "vim_finetuned",
# #         "virchow",
# #         "uni2",            # <-- ADDED
# #         "h0_mini",
# #         "celldino",
# #         "distilled_vit_small"
# #     ]:
# #         size = 224
# #     else:
# #         raise ValueError(f"Model name '{model_name}' not found")

# #     size = (size, size)

# #     transforms_list = [
# #         transforms.Resize(size), 
# #         transforms.ToTensor(), 
# #         transforms.Normalize(mean=mean, std=std)
# #     ]

# #     if "beit_fb" in model_name.lower():
# #         transforms_list = [
# #             transforms.Resize(size),
# #             transforms.ToTensor(),
# #         ]

# #     preprocess_transforms = transforms.Compose(transforms_list)
# #     return preprocess_transforms


# # # ============================================================================
# # # MODEL WRAPPERS
# # # ============================================================================

# # class BeitModel(torch.nn.Module):
# #     """BEIT/Data2Vec model wrapper"""
# #     def __init__(self, device, pretrained_model="facebook/data2vec-vision-base", image_size=224, patch_size=16):
# #         super(BeitModel, self).__init__()
# #         self.feature_extractor = BeitFeatureExtractor.from_pretrained(pretrained_model)
# #         self.image_size = image_size
# #         self.patch_size = patch_size
# #         self.model = Data2VecVisionModel.from_pretrained(pretrained_model)
# #         self.device = device
# #         self.avg_pooling = nn.AdaptiveAvgPool1d((1))

# #     def forward(self, images):
# #         inputs = self.feature_extractor(images=images, return_tensors="pt")
# #         inputs = inputs["pixel_values"].to(self.device)
# #         outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
# #         encoder_hidden_states = outputs.hidden_states

# #         # Take only the 13th layer
# #         features = encoder_hidden_states[12][:, 1:, :].permute(0, 2, 1)
# #         features = self.avg_pooling(features)

# #         return features.squeeze(dim=2)


# # class Phikon(nn.Module):
# #     """Owkin Phikon model wrapper"""
# #     def __init__(self):
# #         super(Phikon, self).__init__()
# #         self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

# #     def forward(self, x):
# #         outputs = self.model(x)
# #         features = outputs.last_hidden_state[:, 0, :]
# #         return features


# # class H0mini(nn.Module):
# #     """h0 mini model wrapper"""
# #     def __init__(self):
# #         super(H0mini, self).__init__()
# #         self.model = timm.create_model("hf-hub:bioptimus/H0-mini",pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,)

# #     def forward(self, x):
# #         outputs = self.model(x)
# #         features = outputs[:, 0]
# #         return features

# # class Virchow(nn.Module):
# #     """Virchow2 model wrapper"""
# #     def __init__(self):
# #         super(Virchow, self).__init__()
# #         self.model = timm.create_model(
# #             "hf-hub:paige-ai/Virchow2", 
# #             pretrained=True, 
# #             mlp_layer=SwiGLUPacked, 
# #             act_layer=torch.nn.SiLU
# #         )

# #     def forward(self, x):
# #         outputs = self.model(x)
# #         features = outputs[:, 0]
# #         return features


# # class UNI2(nn.Module):
# #     """
# #     MahmoodLab UNI2-h model wrapper.

# #     UNI2-h is a ViT-H/14 (1536-dim, 24-head, 24-layer) histology foundation
# #     model with SwiGLU MLP, layer-scale, 8 register tokens, and no
# #     class-token positional embedding (no_embed_class=True). Loaded via
# #     timm's hf-hub integration with the exact architecture kwargs the
# #     checkpoint was trained with (per the model card on Hugging Face).

# #     Requires a Hugging Face account with access approved for
# #     "MahmoodLab/UNI2-h", and either `huggingface-cli login` or the
# #     HF_TOKEN environment variable set before loading.
# #     """
# #     def __init__(self):
# #         super(UNI2, self).__init__()
# #         timm_kwargs = {
# #             "img_size": 224,
# #             "patch_size": 14,
# #             "depth": 24,
# #             "num_heads": 24,
# #             "init_values": 1e-5,
# #             "embed_dim": 1536,
# #             "mlp_ratio": 2.66667 * 2,
# #             "num_classes": 0,
# #             "no_embed_class": True,
# #             "mlp_layer": SwiGLUPacked,
# #             "act_layer": torch.nn.SiLU,
# #             "reg_tokens": 8,
# #             "dynamic_img_size": True,
# #         }
# #         self.model = timm.create_model(
# #             "hf-hub:MahmoodLab/UNI2-h",
# #             pretrained=True,
# #             **timm_kwargs
# #         )

# #     def forward(self, x):
# #         # CLS token
# #         features = self.model(x)
# #         return features


# # class CellViTPlusPlus(nn.Module):
# #     """
# #     CellViT-SAM-H encoder feature extractor.

# #     Loads the (CellViT-finetuned) SAM-H ViT-H/16 image encoder from an
# #     official CellViT-SAM-H-x40-AMP.pth checkpoint and produces one
# #     embedding per input crop by mean-pooling the encoder's patch tokens.

# #     CAVEAT: this pools over ALL tokens in the crop. The official CellViT++
# #     classification pipeline instead pools only the tokens that fall inside
# #     a cell's segmentation mask within a larger WSI tile. This is a
# #     reasonable proxy for single, centered cell crops -- but it is an
# #     approximation, not a literal reproduction of their reported numbers.
# #     Label results from this accordingly (e.g. "CellViT-SAM-H (encoder
# #     tokens, approx.)"), not as an exact CellViT++ benchmark figure.

# #     Requires:
# #       pip install segment-anything
# #       CellViT-SAM-H-x40-AMP.pth from https://github.com/TIO-IKIM/CellViT-plus-plus
# #       passed via --checkpoint
# #     """
# #     def __init__(self, checkpoint_path, input_size=1024):
# #         super().__init__()
# #         if sam_model_registry is None:
# #             raise ImportError(
# #                 "segment-anything not installed. Run: pip install segment-anything"
# #             )
# #         if checkpoint_path is None:
# #             raise ValueError(
# #                 "--checkpoint is required for cellvitpp: path to CellViT-SAM-H-x40-AMP.pth"
# #             )

# #         # Bare SAM-H encoder shell, then overwrite with CellViT-finetuned weights
# #         sam = sam_model_registry["vit_h"](checkpoint=None)
# #         self.encoder = sam.image_encoder  # ViT-H/16, native input 1024x1024

# #         ckpt = torch.load(checkpoint_path, map_location="cpu")
# #         state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

# #         # CellViT checkpoint releases have used different key prefixes for
# #         # the encoder submodule; try the known ones.
# #         encoder_state = {}
# #         for key, value in state_dict.items():
# #             for prefix in ("encoder.", "model.encoder.", "image_encoder."):
# #                 if key.startswith(prefix):
# #                     encoder_state[key[len(prefix):]] = value
# #                     break

# #         if len(encoder_state) == 0:
# #             raise KeyError(
# #                 "Could not find encoder weights using known prefixes "
# #                 "('encoder.', 'model.encoder.', 'image_encoder.'). "
# #                 f"Top-level checkpoint keys: {list(state_dict.keys())[:10]}. "
# #                 "Inspect the checkpoint structure and adjust the prefix list "
# #                 "in CellViTPlusPlus.__init__."
# #             )

# #         msg = self.encoder.load_state_dict(encoder_state, strict=False)
# #         print(f"[cellvitpp] Loaded SAM-H encoder: "
# #               f"{len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys")
# #         if msg.missing_keys:
# #             print(f"  missing (first 5): {msg.missing_keys[:5]}")
# #         if msg.unexpected_keys:
# #             print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")

# #         # Only interpolate pos_embed if deviating from SAM-H's native 1024 input
# #         if input_size != 1024:
# #             old_pos_embed = self.encoder.pos_embed.data           # (1, 64, 64, C)
# #             old_pos_embed = old_pos_embed.permute(0, 3, 1, 2)      # (1, C, 64, 64)
# #             new_grid = input_size // 16
# #             new_pos_embed = nn.functional.interpolate(
# #                 old_pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False
# #             )
# #             new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
# #             self.encoder.pos_embed = nn.Parameter(new_pos_embed)

# #     def forward(self, x):
# #         feats = self.encoder(x)          # (B, 256, H', W') SAM neck output
# #         feats = feats.mean(dim=[2, 3])   # mean-pool tokens -> (B, 256)
# #         return feats

# import re
# import torch
# import torch.nn as nn
# from torchvision import transforms
# from torchvision.models import resnet
# from transformers import AutoImageProcessor, BeitFeatureExtractor, Data2VecVisionModel, ViTModel
# from transformers import AutoImageProcessor, AutoModel
# import timm
# from timm.layers import SwiGLUPacked
# import math
# import torch
# from torch import nn
# try:
#     from segment_anything import sam_model_registry
# except ImportError:
#     sam_model_registry = None

    
# def _interpolate_pos_encoding(x, w, h):
#     N = x.shape[1] - 1
#     dim = x.shape[-1]
#     w0 = w / int(math.sqrt(N))
#     h0 = h / int(math.sqrt(N))
#     patch_pos_embed = nn.functional.interpolate(
#         x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
#         scale_factor=(w0, h0),
#         mode="bicubic",
#     )
#     patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
#     return torch.cat((x[:, :1], patch_pos_embed), dim=1)

# def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14"):
#     model = torch.hub.load("facebookresearch/dinov2", modelname)
#     tensor_corr_shape = _interpolate_pos_encoding(model.pos_embed, 16, 16)
#     pos_embed = nn.Parameter(torch.zeros(1, 257))
#     pos_embed.data = tensor_corr_shape
#     model.pos_embed = pos_embed
#     return model
# # Try to import local modules
# try:
#     from models.resnet_retccl import resnet50 as retccl_res50
# except ImportError:
#     retccl_res50 = None

# try:
#     from models.lemon_loader import get_lemon_vit
# except ImportError:
#     get_lemon_vit = None

# try:
#     from dinov2.models.vision_transformer import DinoVisionTransformer
# except ImportError:
#     DinoVisionTransformer = None



# def get_models(modelname, saved_model_path=None):
#     """
#     Main function to load various pre-trained models for feature extraction.
    
#     Args:
#         modelname: Name of the model to load
#         saved_model_path: Path to checkpoint (required for some models)
    
#     Returns:
#         PyTorch model ready for feature extraction
#     """
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

#     # --- histology-pretrained models
#     if modelname.lower() == "retccl":
#         model = get_retCCL(saved_model_path)
#     elif modelname.lower() == "lemon":
#         if get_lemon_vit is None:
#             raise ImportError("LEMON loader not available. Ensure models/lemon_loader.py exists.")
#         model = get_lemon_vit(saved_model_path, model_name="vits8", img_size=40)
#     elif modelname.lower() == "owkin":
#         model = Phikon()
#     elif modelname.lower() == "virchow":
#         model = Virchow()
#     elif modelname.lower() == "uni2":
#         model = UNI2()
#     elif modelname.lower() == "cellvitpp":
#         model = CellViTPlusPlus(saved_model_path, input_size=64)
#     # --- vision foundation models
#     elif modelname.lower() == "resnet50":
#         model = get_res50()
#     elif modelname.lower() == "resnet50_full":
#         model = get_full_res50()
#     elif modelname.lower() == "beit_fb":
#         model = BeitModel(device)
#     elif modelname.lower() == "h0_mini":
#         model = H0mini(device)
#     elif modelname.lower() == "distilled_vit_small":
#         model = get_distilled_vit_small(saved_model_path)
#     # --- our finetuned models
#     elif modelname.lower() in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_ssl"]:
#         model = get_dino_finetuned_downloaded(saved_model_path, modelname)
#     elif modelname.lower() == "celldino":
#         if DinoVisionTransformer is None:
#             raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
#         model = get_celldino_finetuned_downloaded(saved_model_path, modelname)
#     elif modelname.lower() == "vit_small_patch7":
#         if DinoVisionTransformer is None:
#             raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
#         model = get_vit_small_patch7_finetuned(saved_model_path, modelname)
#     elif modelname.lower() in ["dinov3_vits16", "dinov3_vitb16", "dinov3_vitl16", "dinov3_vitg16"]:
#         model = get_dinov3_finetuned_downloaded(saved_model_path, modelname)
#     else:
#         raise ValueError(f"Model {modelname} not found")

#     model = model.to(device)
#     model.eval()

#     return model


# # ============================================================================
# # DISTILLED VIT-SMALL LOADER (NEW)
# # ============================================================================
# def get_distilled_vit_small(model_path):
#         """
#         Load the distilled ViT-Small/14 model (multi-teacher KD: UNI2, Virchow2,
#         H-optimus-0 -> ViT-S/14, from train_knowledgedistill.py).

#         Accepts checkpoints in these formats:
#         1. {"student": {"key": value, ...}}            -- do_test() eval snapshot,
#                                                             RECOMMENDED: backbone only,
#                                                             no distillation-head or
#                                                             optimizer state to strip.
#         2. {"model": {"student_backbone.key": value,    -- full training
#                         "distill_head.key": value, ...},     checkpoint_*.pth
#             "optimizer": {...}, "iteration": N}             (save_checkpoint()).
#                                                             distill_head.* keys are
#                                                             discarded; only the
#                                                             backbone is loaded.
#         3. {"teacher": {"backbone.key": value, ...}}    -- older single-teacher
#         4. {"backbone": {"key": value, ...}}               KD experiment formats,
#         5. Plain state_dict {"key": value, ...}            kept for backward compat.

#         Returns a timm ViT-Small/14 (DINOv2-style) model ready for feature
#         extraction (CLS token).
#         """
#         ARCH = "vit_small_patch14_dinov2.lvd142m"
#         IMG_SIZE = 224

#         if model_path is None:
#             print(f"No checkpoint provided, loading ImageNet-pretrained {ARCH}")
#             model = timm.create_model(ARCH, pretrained=True, num_classes=0, img_size=IMG_SIZE)
#             return model

#         pretrained = torch.load(model_path, map_location="cpu", weights_only=False)

#         # Extract state_dict from various checkpoint formats
#         if isinstance(pretrained, dict):
#             if "student" in pretrained:
#                 # do_test() eval snapshot: {"student": student_backbone.state_dict()}
#                 # -- already backbone-only, no head/optimizer keys to strip.
#                 state_dict = pretrained["student"]
#                 print(f"Loaded from 'student' key, {len(state_dict)} params")

#             elif "teacher" in pretrained:
#                 # Format: {"teacher": {"backbone.cls_token": ..., ...}}
#                 raw_state_dict = pretrained["teacher"]
#                 state_dict = {}
#                 for key, value in raw_state_dict.items():
#                     if "dino_head" in key or "ibot_head" in key:
#                         continue
#                     new_key = key.replace("backbone.", "")
#                     state_dict[new_key] = value
#                 print(f"Loaded from 'teacher' key, {len(state_dict)} params")

#             elif "backbone" in pretrained:
#                 # Format: {"backbone": {"cls_token": ..., ...}, "classifier": ...}
#                 state_dict = pretrained["backbone"]
#                 print(f"Loaded from 'backbone' key, {len(state_dict)} params")

#             elif "state_dict" in pretrained:
#                 state_dict = pretrained["state_dict"]
#                 print(f"Loaded from 'state_dict' key, {len(state_dict)} params")

#             elif "model" in pretrained:
#                 # save_checkpoint() full training checkpoint:
#                 # {"model": DistillationModel.state_dict(), "optimizer": ..., "iteration": ...}
#                 # DistillationModel.state_dict() keys are prefixed "student_backbone."
#                 # and "distill_head." -- keep only the backbone, drop the head, and
#                 # strip the prefix so keys match the plain timm model.
#                 raw_state_dict = pretrained["model"]
#                 state_dict = {}
#                 for key, value in raw_state_dict.items():
#                     if not key.startswith("student_backbone."):
#                         continue  # discards every distill_head.* key
#                     state_dict[key[len("student_backbone."):]] = value
#                 print(f"Loaded from 'model' key (student_backbone.* only), {len(state_dict)} params")

#             else:
#                 # Assume it's a plain state_dict
#                 state_dict = pretrained
#                 print(f"Loaded as plain state_dict, {len(state_dict)} params")
#         else:
#             state_dict = pretrained
#             print(f"Loaded as plain state_dict, {len(state_dict)} params")

#         # Remove any remaining prefixes (module.)
#         clean_state_dict = {}
#         for key, value in state_dict.items():
#             new_key = key.replace("module.", "")
#             clean_state_dict[new_key] = value

#         # Infer embed_dim from checkpoint (sanity check, not architecture selection --
#         # ARCH is fixed above to match training exactly, patch14 not patch16)
#         if "patch_embed.proj.weight" in clean_state_dict:
#             patch_weight = clean_state_dict["patch_embed.proj.weight"]
#             embed_dim, in_chans, patch_h, patch_w = patch_weight.shape
#             print(f"Checkpoint patch_embed: embed_dim={embed_dim}, patch_size={patch_h}, in_chans={in_chans}")
#             if patch_h != 14:
#                 print(f"[warn] checkpoint patch_size={patch_h} != expected 14 for {ARCH} -- "
#                     f"this checkpoint may not match the distilled ViT-S/14 architecture, "
#                     f"expect a shape-mismatch RuntimeError below if so")

#         # Build timm model matching training exactly
#         model = timm.create_model(ARCH, pretrained=False, num_classes=0, img_size=IMG_SIZE)

#         # Load weights
#         msg = model.load_state_dict(clean_state_dict, strict=False)

#         print(f"\n==== LOAD REPORT (distilled_vit_small) ====")
#         if msg.missing_keys:
#             print(f"Missing keys ({len(msg.missing_keys)}):")
#             for k in msg.missing_keys:
#                 print(f"    {k}")
#         else:
#             print("Missing keys: none")

#         if msg.unexpected_keys:
#             print(f"Unexpected keys ({len(msg.unexpected_keys)}):")
#             for k in msg.unexpected_keys:
#                 print(f"    {k}")
#         else:
#             print("Unexpected keys: none")

#         return model


# def get_retCCL(model_path):
#     """Load RetCCL ResNet50 model"""
#     if retccl_res50 is None:
#         raise ImportError("RetCCL model not available")
    
#     model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
#     pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
#     model.fc = nn.Identity()
#     model.load_state_dict(pretext_model, strict=True)
#     return model

# def get_dino_finetuned_downloaded(saved_model_path, modelname="dinov2_vits14"):
#     model = get_downloaded_dino_vit_interpolated(modelname)
#     pretrained = torch.load(saved_model_path, map_location="cpu")

#     # Support multiple checkpoint formats:
#     # 1. old EMA-teacher format:  {"teacher": {"backbone.xxx": ...}}
#     # 2. new virchow-distill format: {"backbone": {"xxx": ...}}
#     #    (optionally with FSDP "_fsdp_wrapped_module." prefix)
#     if "teacher" in pretrained:
#         state_dict = {
#             k.replace("backbone.", ""): v
#             for k, v in pretrained["teacher"].items()
#             if k.startswith("backbone.")
#         }
#     elif "student" in pretrained:
#         state_dict = pretrained["student"]
#         # state_dict = {
#         #     k.replace("backbone.", ""): v
#         #     for k, v in pretrained["student"].items()
#         #     if k.startswith("backbone.")
#         # }
#     elif "backbone" in pretrained:
#         state_dict = {
#             k.replace("_fsdp_wrapped_module.", ""): v
#             for k, v in pretrained["backbone"].items()
#         }
#     else:
#         raise KeyError(
#             f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'backbone' key. "
#             f"Available keys: {list(pretrained.keys())}"
#         )

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     if missing:
#         print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
#     if unexpected:
#         print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

#     model.eval()
#     return model


# def get_vit_small_patch7_finetuned(saved_model_path, modelname="vit_small_patch7"):
#     """
#     Load the from-scratch ViT-Small/patch7 ImmunoDINO backbone.

#     This backbone was built in training via build_model_from_cfg (arch="vit_small",
#     patch_size=7, global_crops_size=63) -- NOT via the torch.hub patch14
#     pretrained path used by get_dino_finetuned_downloaded. Do not route this
#     model name through get_downloaded_dino_vit_interpolated: that function
#     downloads a patch14 hub checkpoint and hard-codes a 16x16 (224px/14)
#     positional-embedding interpolation, which is architecturally incompatible
#     with a patch7 model trained on 63px crops (63 // 7 = 9x9 = 81 tokens).

#     Accepts checkpoints in these formats, mirroring load_backbone_checkpoint()
#     in dinov2/train/ssl_meta_arch.py:
#         1. {"teacher": {"backbone.xxx": ..., "dino_head.xxx": ...}}  -- do_test()
#            eval checkpoints (RECOMMENDED: this is what periodic eval snapshots
#            and the final teacher_checkpoint.pth use).
#         2. {"student": {"backbone.xxx": ...}} or {"student": {"xxx": ...}}
#         3. {"backbone": {"xxx": ...}} (optionally with "_fsdp_wrapped_module." prefix)
#         4. Plain backbone state_dict.

#     Returns the teacher backbone from build_model_from_cfg (a
#     DinoVisionTransformer configured to match training exactly, including
#     custom gating/window/register-token fields) -- forward() called with
#     default args returns CLS-token features directly, consumed as-is by
#     extract_features.py.
#     """
#     from omegaconf import OmegaConf
#     from dinov2.models import build_model_from_cfg

#     if saved_model_path is None:
#         raise ValueError("--checkpoint is required for vit_small_patch7")

#     # Rebuild via build_model_from_cfg (same function training used, since
#     # arch="vit_small" isn't in the hardcoded hub-name list in
#     # SSLMetaArch.__init__). This is required -- not optional -- because this
#     # architecture has custom fields (gating_type, gate_bias_init,
#     # window_size, num_register_tokens, swiglufused FFN) that a hand-built
#     # DinoVisionTransformer(...) call cannot reproduce. Values below mirror
#     # the *effective* (merged) training custom.yaml -- the override block's
#     # values win over the base block's for any overlapping key.
#     cfg = OmegaConf.create({
#         "student": {
#             "arch": "vit_small",
#             "patch_size": 7,
#             "layerscale": 1.0e-05,
#             "drop_path_rate": 0.4,
#             "drop_path_uniform": True,
#             "ffn_layer": "mlp",
#             "block_chunks": 0,
#             "qkv_bias": False,
#             "proj_bias": True,
#             "ffn_bias": True,
#             "num_register_tokens": 4,
#             "interpolate_antialias": False,
#             "interpolate_offset": 0.1,
#             "gating_type": "headwise",
#             "gate_bias_init": 4.0,
#             "window_size": 64,
#         },
#         "crops": {
#             "global_crops_size": 63,
#         },
#     })

#     student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
#     model = teacher_backbone  # checkpoint's "teacher" key holds these weights

#     pretrained = torch.load(saved_model_path, map_location="cpu")

#     if "teacher" in pretrained:
#         raw_sd = pretrained["teacher"]
#     elif "student" in pretrained:
#         raw_sd = pretrained["student"]
#     elif "backbone" in pretrained:
#         raw_sd = {
#             k.replace("_fsdp_wrapped_module.", ""): v
#             for k, v in pretrained["backbone"].items()
#         }
#     elif isinstance(pretrained, dict):
#         raw_sd = pretrained
#     else:
#         raise KeyError(
#             f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'student' "
#             f"nor 'backbone' key. Available keys: {list(pretrained.keys())}"
#         )

#     # Strip known prefixes (backbone., teacher.backbone., student.backbone.)
#     # and drop any head keys (dino_head., ibot_head.) which don't belong on
#     # a bare backbone.
#     state_dict = {}
#     for k, v in raw_sd.items():
#         if "dino_head" in k or "ibot_head" in k:
#             continue
#         if k.startswith("teacher.backbone."):
#             state_dict[k[len("teacher.backbone."):]] = v
#         elif k.startswith("student.backbone."):
#             state_dict[k[len("student.backbone."):]] = v
#         elif k.startswith("backbone."):
#             state_dict[k[len("backbone."):]] = v
#         else:
#             state_dict[k] = v

#     missing, unexpected = model.load_state_dict(state_dict, strict=False)
#     if missing:
#         print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
#     if unexpected:
#         print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
#     if len(state_dict) == 0:
#         print("[warn] 0 tensors found in checkpoint -- check checkpoint format / prefixes above.")
#     elif len(missing) == len(list(model.state_dict().keys())):
#         print(
#             "[warn] 0 tensors were actually loaded (all keys mismatched). "
#             "The model is effectively random-init. Check that the cfg dict "
#             "in get_vit_small_patch7_finetuned matches the checkpoint's "
#             "actual training config (block_chunks, num_register_tokens, "
#             "ffn_layer, gating_type, window_size, etc.)."
#         )
#     else:
#         print(f"[ok] Loaded {len(state_dict) - len(missing)}/{len(state_dict)} backbone tensors.")

#     model.eval()
#     return model


# def get_celldino_finetuned_downloaded(saved_model_path, modelname):
#     """Load CellDINO model with automatic architecture inference"""
#     pretrained = torch.load(saved_model_path, map_location="cpu")

#     # Extract state_dict safely
#     if "teacher" in pretrained:
#         state_dict = pretrained["teacher"]
#     elif "student" in pretrained:
#         state_dict = pretrained["student"]
#     elif "state_dict" in pretrained:
#         state_dict = pretrained["state_dict"]
#     elif "model" in pretrained:
#         state_dict = pretrained["model"]
#     else:
#         state_dict = pretrained

#     # Remove DDP/module prefixes
#     clean_state_dict = {}
#     for key, value in state_dict.items():
#         new_key = key.replace("module.", "")
#         new_key = new_key.replace("backbone.", "")
#         clean_state_dict[new_key] = value

#     # Infer architecture from checkpoint
#     patch_weight = clean_state_dict["patch_embed.proj.weight"]
#     embed_dim, in_chans, patch_h, patch_w = patch_weight.shape

#     assert patch_h == patch_w, f"Non-square patch embedding found: {patch_h}x{patch_w}"
#     patch_size = patch_h

#     print(f"Inferred from checkpoint:")
#     print(f"  embed_dim  = {embed_dim}")
#     print(f"  in_chans   = {in_chans}")
#     print(f"  patch_size = {patch_size}")

#     # Infer model depth/heads from embed_dim
#     if embed_dim == 384:
#         depth = 12
#         num_heads = 6
#     elif embed_dim == 768:
#         depth = 12
#         num_heads = 12
#     elif embed_dim == 1024:
#         depth = 24
#         num_heads = 16
#     elif embed_dim == 1536:
#         depth = 40
#         num_heads = 24
#     else:
#         raise ValueError(f"Unsupported embed_dim={embed_dim}. Please specify architecture manually.")

#     # Infer image size from pos_embed
#     img_size = 128  # fallback default

#     if "pos_embed" in clean_state_dict:
#         num_tokens = clean_state_dict["pos_embed"].shape[1]
#         num_patches = num_tokens - 1
#         grid_size = int(num_patches ** 0.5)

#         if grid_size * grid_size == num_patches:
#             img_size = grid_size * patch_size
#             print(f"Inferred img_size = {img_size} from pos_embed")
#         else:
#             print("Could not infer square image size from pos_embed; using fallback img_size=128")

#     # Build compatible DinoVisionTransformer
#     model = DinoVisionTransformer(
#         img_size=img_size,
#         patch_size=patch_size,
#         in_chans=in_chans,
#         embed_dim=embed_dim,
#         depth=depth,
#         num_heads=num_heads,
#         mlp_ratio=4.0,
#         qkv_bias=True,
#         ffn_bias=True,
#         proj_bias=True,
#     )

#     # Remove classifier/incompatible heads
#     keys_to_remove = []
#     for k in clean_state_dict.keys():
#         if any(x in k for x in ["head", "classifier", "fc", "dino_head", "ibot_head"]):
#             keys_to_remove.append(k)

#     for k in keys_to_remove:
#         clean_state_dict.pop(k, None)

#     # Load weights safely
#     if "pos_embed" in clean_state_dict:
#         if clean_state_dict["pos_embed"].shape != model.pos_embed.shape:
#             print(
#                 f"Skipping pos_embed due to shape mismatch: "
#                 f"{clean_state_dict['pos_embed'].shape} vs {model.pos_embed.shape}"
#             )
#             clean_state_dict.pop("pos_embed")

#     msg = model.load_state_dict(clean_state_dict, strict=False)
#     print("Load result:", msg)

#     return model


# def get_dinov3_finetuned_downloaded(model_path, modelname):
#     """Load DINOv3 model"""
#     pretrained_model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
#     processor = AutoImageProcessor.from_pretrained(pretrained_model_name)
#     model = AutoModel.from_pretrained(
#         pretrained_model_name, 
#         device_map="auto"
#     )
#     return model


# def get_res50():
#     """Load ResNet50 with early stopping (remove last residual block)"""
#     model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

#     class Reshape(nn.Module):
#         def forward(self, x):
#             return x.reshape(x.shape[0], -1)

#     # Delete last res block as this has been shown to work better
#     model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

#     return model


# def get_full_res50():
#     """Load full ResNet50"""
#     model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")
#     return model


# # ============================================================================
# # TRANSFORMS
# # ============================================================================

# def get_transforms(model_name):
#     """
#     Get appropriate image transforms for each model.
#     """
#     # Default ImageNet normalization
#     mean = (0.485, 0.456, 0.406)
#     std = (0.229, 0.224, 0.225)

#     if model_name.lower() in ["ctranspath", "resnet50", "beit_fb", "resnet50_full"]:
#         size = 224
#     elif model_name.lower() == "lemon":
#         size = 40
#     elif model_name.lower() == "owkin":
#         image_processor = AutoImageProcessor.from_pretrained("owkin/phikon")
#         mean, std = image_processor.image_mean, image_processor.image_std
#         size = image_processor.size["height"]
#     elif model_name.lower() == "retccl":
#         size = 256
#     elif model_name.lower() == "kimianet":
#         size = 1000
#     elif model_name.lower() == "cellvitpp":
#         size = 64
#     elif model_name.lower() == "imagebind":
#         size = 224
#         mean = (0.48145466, 0.4578275, 0.40821073)
#         std = (0.26862954, 0.26130258, 0.27577711)
#     elif model_name.lower() == "vit_small_patch7":
#         # Must match training: global_crops_size=63, patch_size=7 -> 9x9=81 tokens.
#         # Do NOT resize to 224 here -- that would silently change the patch
#         # grid the model was trained on.
#         size = 63
#     elif model_name.lower() in [
#         "dinov2_vits14",
#         "dinov2_ssl",
#         "dinov3_vits16",
#         "dinov2_vitb14",
#         "dinov2_vitl14",
#         "dinov2_vitg14",
#         "dinov2_finetuned",
#         "dinov2_vits14_interpolated",
#         "dinov2_finetuned_downloaded",
#         "remedis",
#         "vim_finetuned",
#         "virchow",
#         "uni2",            # <-- ADDED
#         "h0_mini",
#         "celldino",
#         "distilled_vit_small"
#     ]:
#         size = 224
#     else:
#         raise ValueError(f"Model name '{model_name}' not found")

#     size = (size, size)

#     transforms_list = [
#         transforms.Resize(size), 
#         transforms.ToTensor(), 
#         transforms.Normalize(mean=mean, std=std)
#     ]

#     if "beit_fb" in model_name.lower():
#         transforms_list = [
#             transforms.Resize(size),
#             transforms.ToTensor(),
#         ]

#     preprocess_transforms = transforms.Compose(transforms_list)
#     return preprocess_transforms


# # ============================================================================
# # MODEL WRAPPERS
# # ============================================================================

# class BeitModel(torch.nn.Module):
#     """BEIT/Data2Vec model wrapper"""
#     def __init__(self, device, pretrained_model="facebook/data2vec-vision-base", image_size=224, patch_size=16):
#         super(BeitModel, self).__init__()
#         self.feature_extractor = BeitFeatureExtractor.from_pretrained(pretrained_model)
#         self.image_size = image_size
#         self.patch_size = patch_size
#         self.model = Data2VecVisionModel.from_pretrained(pretrained_model)
#         self.device = device
#         self.avg_pooling = nn.AdaptiveAvgPool1d((1))

#     def forward(self, images):
#         inputs = self.feature_extractor(images=images, return_tensors="pt")
#         inputs = inputs["pixel_values"].to(self.device)
#         outputs = self.model(inputs, output_hidden_states=True, return_dict=True)
#         encoder_hidden_states = outputs.hidden_states

#         # Take only the 13th layer
#         features = encoder_hidden_states[12][:, 1:, :].permute(0, 2, 1)
#         features = self.avg_pooling(features)

#         return features.squeeze(dim=2)


# class Phikon(nn.Module):
#     """Owkin Phikon model wrapper"""
#     def __init__(self):
#         super(Phikon, self).__init__()
#         self.model = ViTModel.from_pretrained("owkin/phikon", add_pooling_layer=False)

#     def forward(self, x):
#         outputs = self.model(x)
#         features = outputs.last_hidden_state[:, 0, :]
#         return features


# class H0mini(nn.Module):
#     """h0 mini model wrapper"""
#     def __init__(self):
#         super(H0mini, self).__init__()
#         self.model = timm.create_model("hf-hub:bioptimus/H0-mini",pretrained=True, mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU,)

#     def forward(self, x):
#         outputs = self.model(x)
#         features = outputs[:, 0]
#         return features

# class Virchow(nn.Module):
#     """Virchow2 model wrapper"""
#     def __init__(self):
#         super(Virchow, self).__init__()
#         self.model = timm.create_model(
#             "hf-hub:paige-ai/Virchow2", 
#             pretrained=True, 
#             mlp_layer=SwiGLUPacked, 
#             act_layer=torch.nn.SiLU
#         )

#     def forward(self, x):
#         outputs = self.model(x)
#         features = outputs[:, 0]
#         return features


# class UNI2(nn.Module):
#     """
#     MahmoodLab UNI2-h model wrapper.

#     UNI2-h is a ViT-H/14 (1536-dim, 24-head, 24-layer) histology foundation
#     model with SwiGLU MLP, layer-scale, 8 register tokens, and no
#     class-token positional embedding (no_embed_class=True). Loaded via
#     timm's hf-hub integration with the exact architecture kwargs the
#     checkpoint was trained with (per the model card on Hugging Face).

#     Requires a Hugging Face account with access approved for
#     "MahmoodLab/UNI2-h", and either `huggingface-cli login` or the
#     HF_TOKEN environment variable set before loading.
#     """
#     def __init__(self):
#         super(UNI2, self).__init__()
#         timm_kwargs = {
#             "img_size": 224,
#             "patch_size": 14,
#             "depth": 24,
#             "num_heads": 24,
#             "init_values": 1e-5,
#             "embed_dim": 1536,
#             "mlp_ratio": 2.66667 * 2,
#             "num_classes": 0,
#             "no_embed_class": True,
#             "mlp_layer": SwiGLUPacked,
#             "act_layer": torch.nn.SiLU,
#             "reg_tokens": 8,
#             "dynamic_img_size": True,
#         }
#         self.model = timm.create_model(
#             "hf-hub:MahmoodLab/UNI2-h",
#             pretrained=True,
#             **timm_kwargs
#         )

#     def forward(self, x):
#         # CLS token
#         features = self.model(x)
#         return features


# class CellViTPlusPlus(nn.Module):
#     """
#     CellViT-SAM-H encoder feature extractor.

#     Loads the (CellViT-finetuned) SAM-H ViT-H/16 image encoder from an
#     official CellViT-SAM-H-x40-AMP.pth checkpoint and produces one
#     embedding per input crop by mean-pooling the encoder's patch tokens.

#     CAVEAT: this pools over ALL tokens in the crop. The official CellViT++
#     classification pipeline instead pools only the tokens that fall inside
#     a cell's segmentation mask within a larger WSI tile. This is a
#     reasonable proxy for single, centered cell crops -- but it is an
#     approximation, not a literal reproduction of their reported numbers.
#     Label results from this accordingly (e.g. "CellViT-SAM-H (encoder
#     tokens, approx.)"), not as an exact CellViT++ benchmark figure.

#     Requires:
#       pip install segment-anything
#       CellViT-SAM-H-x40-AMP.pth from https://github.com/TIO-IKIM/CellViT-plus-plus
#       passed via --checkpoint
#     """
#     def __init__(self, checkpoint_path, input_size=64):
#         super().__init__()
#         if sam_model_registry is None:
#             raise ImportError(
#                 "segment-anything not installed. Run: pip install segment-anything"
#             )
#         if checkpoint_path is None:
#             raise ValueError(
#                 "--checkpoint is required for cellvitpp: path to CellViT-SAM-H-x40-AMP.pth"
#             )

#         # Bare SAM-H encoder shell, then overwrite with CellViT-finetuned weights
#         sam = sam_model_registry["vit_h"](checkpoint=None)
#         self.encoder = sam.image_encoder  # ViT-H/16, native input 1024x1024
#         ckpt = torch.load(checkpoint_path, map_location="cpu")
#         if isinstance(ckpt, dict) and any(
#             k in ckpt for k in ("model", "model_state_dict", "state_dict")
#         ):
#             state_dict = ckpt.get(
#                 "model", ckpt.get("model_state_dict", ckpt.get("state_dict"))
#             )
#         else:
#             state_dict = ckpt

#         # CellViT checkpoint releases have used different key prefixes for
#         # the encoder submodule; try the known ones.
#         encoder_state = {}
#         for key, value in state_dict.items():
#             for prefix in ("encoder.", "model.encoder.", "image_encoder."):
#                 if key.startswith(prefix):
#                     encoder_state[key[len(prefix):]] = value
#                     break

#         if len(encoder_state) == 0:
#             sample_keys = list(state_dict.keys())[:20]
#             raise KeyError(
#                 "Could not find encoder weights using known prefixes "
#                 "('encoder.', 'model.encoder.', 'image_encoder.'). "
#                 f"First 20 keys actually present in the state_dict: {sample_keys}. "
#                 "Inspect these key names and add the correct prefix to the "
#                 "list in CellViTPlusPlus.__init__ (the checkpoint may use a "
#                 "different prefix than the three currently handled)."
#             )
#         # ckpt = torch.load(checkpoint_path, map_location="cpu")
#         # state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt))

#         # # CellViT checkpoint releases have used different key prefixes for
#         # # the encoder submodule; try the known ones.
#         # encoder_state = {}
#         # for key, value in state_dict.items():
#         #     for prefix in ("encoder.", "model.encoder.", "image_encoder."):
#         #         if key.startswith(prefix):
#         #             encoder_state[key[len(prefix):]] = value
#         #             break

#         # if len(encoder_state) == 0:
#         #     raise KeyError(
#         #         "Could not find encoder weights using known prefixes "
#         #         "('encoder.', 'model.encoder.', 'image_encoder.'). "
#         #         f"Top-level checkpoint keys: {list(state_dict.keys())[:10]}. "
#         #         "Inspect the checkpoint structure and adjust the prefix list "
#         #         "in CellViTPlusPlus.__init__."
#         #     )

#         msg = self.encoder.load_state_dict(encoder_state, strict=False)
#         print(f"[cellvitpp] Loaded SAM-H encoder: "
#               f"{len(msg.missing_keys)} missing, {len(msg.unexpected_keys)} unexpected keys")
#         if msg.missing_keys:
#             print(f"  missing (first 5): {msg.missing_keys[:5]}")
#         if msg.unexpected_keys:
#             print(f"  unexpected (first 5): {msg.unexpected_keys[:5]}")

#         # Only interpolate pos_embed if deviating from SAM-H's native 1024 input
#         if input_size != 1024:
#             old_pos_embed = self.encoder.pos_embed.data           # (1, 64, 64, C)
#             old_pos_embed = old_pos_embed.permute(0, 3, 1, 2)      # (1, C, 64, 64)
#             new_grid = input_size // 16
#             new_pos_embed = nn.functional.interpolate(
#                 old_pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False
#             )
#             new_pos_embed = new_pos_embed.permute(0, 2, 3, 1)
#             self.encoder.pos_embed = nn.Parameter(new_pos_embed)

#     def forward(self, x):
#         feats = self.encoder(x)          # (B, 256, H', W') SAM neck output
#         feats = feats.mean(dim=[2, 3])   # mean-pool tokens -> (B, 256)
#         return feats

import re
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet
import math
import torch
from torch import nn
try:
    from segment_anything import sam_model_registry
except ImportError:
    sam_model_registry = None

    
def _interpolate_pos_encoding(x, w, h):
    N = x.shape[1] - 1
    dim = x.shape[-1]
    w0 = w / int(math.sqrt(N))
    h0 = h / int(math.sqrt(N))
    patch_pos_embed = nn.functional.interpolate(
        x[:, 1:].reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
        scale_factor=(w0, h0),
        mode="bicubic",
    )
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return torch.cat((x[:, :1], patch_pos_embed), dim=1)

def get_downloaded_dino_vit_interpolated(modelname="dinov2_vits14"):
    model = torch.hub.load("facebookresearch/dinov2", modelname)
    tensor_corr_shape = _interpolate_pos_encoding(model.pos_embed, 16, 16)
    pos_embed = nn.Parameter(torch.zeros(1, 257))
    pos_embed.data = tensor_corr_shape
    model.pos_embed = pos_embed
    return model
# Try to import local modules
try:
    from models.resnet_retccl import resnet50 as retccl_res50
except ImportError:
    retccl_res50 = None

try:
    from models.lemon_loader import get_lemon_vit
except ImportError:
    get_lemon_vit = None

try:
    from dinov2.models.vision_transformer import DinoVisionTransformer
except ImportError:
    DinoVisionTransformer = None



def get_models(modelname, saved_model_path=None):
    """
    Main function to load various pre-trained models for feature extraction.
    
    Args:
        modelname: Name of the model to load
        saved_model_path: Path to checkpoint (required for some models)
    
    Returns:
        PyTorch model ready for feature extraction
    """
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- histology-pretrained models
    if modelname.lower() == "retccl":
        model = get_retCCL(saved_model_path)
    elif modelname.lower() == "lemon":
        if get_lemon_vit is None:
            raise ImportError("LEMON loader not available. Ensure models/lemon_loader.py exists.")
        model = get_lemon_vit(saved_model_path, model_name="vits8", img_size=40)
    elif modelname.lower() == "cellvitpp":
        model = CellViTPlusPlus(saved_model_path, input_size=64)
    # --- vision foundation models
    elif modelname.lower() == "resnet50":
        model = get_res50()
    elif modelname.lower() == "resnet50_full":
        model = get_full_res50()
    # --- our finetuned models
    elif modelname.lower() in ["dinov2_vits14", "dinov2_vitb14", "dinov2_vitl14", "dinov2_vitg14", "dinov2_ssl"]:
        model = get_dino_finetuned_downloaded(saved_model_path, modelname)
    elif modelname.lower() == "celldino":
        if DinoVisionTransformer is None:
            raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
        model = get_celldino_finetuned_downloaded(saved_model_path, modelname)
    elif modelname.lower() == "vit_small_patch7":
        if DinoVisionTransformer is None:
            raise ImportError("DinoVisionTransformer not available. Check dinov2 installation.")
        model = get_vit_small_patch7_finetuned(saved_model_path, modelname)
    elif modelname.lower() in ["microformer_small", "microformer_tiny", "microformer_base"]:
        model = get_microformer_finetuned(saved_model_path, modelname)
    else:
        raise ValueError(f"Model {modelname} not found")

    model = model.to(device)
    model.eval()

    return model


def get_retCCL(model_path):
    """Load RetCCL ResNet50 model"""
    if retccl_res50 is None:
        raise ImportError("RetCCL model not available")
    
    model = retccl_res50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
    pretext_model = torch.load(model_path, map_location=torch.device("cpu"))
    model.fc = nn.Identity()
    model.load_state_dict(pretext_model, strict=True)
    return model

def get_dino_finetuned_downloaded(saved_model_path, modelname="dinov2_vits14"):
    model = get_downloaded_dino_vit_interpolated(modelname)
    pretrained = torch.load(saved_model_path, map_location="cpu")

    if "teacher" in pretrained:
        state_dict = {
            k.replace("backbone.", ""): v
            for k, v in pretrained["teacher"].items()
            if k.startswith("backbone.")
        }
    elif "student" in pretrained:
        state_dict = pretrained["student"]
    elif "backbone" in pretrained:
        state_dict = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in pretrained["backbone"].items()
        }
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'backbone' key. "
            f"Available keys: {list(pretrained.keys())}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing)>5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected)>5 else ''}")

    model.eval()
    return model


def get_vit_small_patch7_finetuned(saved_model_path, modelname="vit_small_patch7"):
    from omegaconf import OmegaConf
    from dinov2.models import build_model_from_cfg

    if saved_model_path is None:
        raise ValueError("--checkpoint is required for vit_small_patch7")

    cfg = OmegaConf.create({
        "student": {
            "arch": "vit_small",
            "patch_size": 7,
            "layerscale": 1.0e-05,
            "drop_path_rate": 0.4,
            "drop_path_uniform": True,
            "ffn_layer": "mlp",
            "block_chunks": 0,
            "qkv_bias": False,
            "proj_bias": True,
            "ffn_bias": True,
            "num_register_tokens": 4,
            "interpolate_antialias": False,
            "interpolate_offset": 0.1,
            "gating_type": "headwise",
            "gate_bias_init": 4.0,
            "window_size": 64,
        },
        "crops": {
            "global_crops_size": 63,
        },
    })

    student_backbone, teacher_backbone, embed_dim = build_model_from_cfg(cfg, only_teacher=False)
    model = teacher_backbone

    pretrained = torch.load(saved_model_path, map_location="cpu")

    if "teacher" in pretrained:
        raw_sd = pretrained["teacher"]
    elif "student" in pretrained:
        raw_sd = pretrained["student"]
    elif "backbone" in pretrained:
        raw_sd = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in pretrained["backbone"].items()
        }
    elif isinstance(pretrained, dict):
        raw_sd = pretrained
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'student' "
            f"nor 'backbone' key. Available keys: {list(pretrained.keys())}"
        )

    state_dict = {}
    for k, v in raw_sd.items():
        if "dino_head" in k or "ibot_head" in k:
            continue
        if k.startswith("teacher.backbone."):
            state_dict[k[len("teacher.backbone."):]] = v
        elif k.startswith("student.backbone."):
            state_dict[k[len("student.backbone."):]] = v
        elif k.startswith("backbone."):
            state_dict[k[len("backbone."):]] = v
        else:
            state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if len(state_dict) == 0:
        print("[warn] 0 tensors found in checkpoint -- check checkpoint format / prefixes above.")
    elif len(missing) == len(list(model.state_dict().keys())):
        print("[warn] 0 tensors were actually loaded (all keys mismatched).")
    else:
        print(f"[ok] Loaded {len(state_dict) - len(missing)}/{len(state_dict)} backbone tensors.")

    model.eval()
    return model


def get_microformer_finetuned(saved_model_path, modelname="microformer_small"):
    """
    Load the MicroFormer (SIFT) backbone trained via train_microformer.py /
    dinov2/configs/train/cell_dino/microformer_small_cellcrops.yaml.

    Like vit_small_patch7, MicroFormer isn't in SSLMetaArch's hardcoded ViT
    hub-name list, so it's rebuilt via build_model_from_cfg (the same
    function training used) rather than a hand-built constructor call. The
    cfg dict below mirrors microformer_small_cellcrops.yaml's actual training
    values (arch, patch_size, in_chans, crops.global_crops_size) -- if you
    trained a different variant (microformer_tiny / microformer_base) or
    changed patch_size/img_size in the yaml, pass that exact modelname and
    update PATCH_SIZE / IMG_SIZE below to match, or load_state_dict will
    report a shape mismatch (see the [warn] messages below for diagnosis).

    Accepts the same checkpoint formats as do_test()'s teacher_checkpoint.pth:
        {"teacher": {"backbone.xxx": ..., "dino_head.xxx": ...}}
    plus, defensively, {"student": {...}} / {"backbone": {...}} / a plain
    state_dict, mirroring get_vit_small_patch7_finetuned's fallback handling.

    Unlike the raw DinoVisionTransformer path, MicroFormer's forward(x)
    (is_training=False, the default) already returns the pooled
    x_norm_clstoken embedding directly as a plain tensor -- no extra
    CLS-token indexing needed here.
    """
    from omegaconf import OmegaConf
    from dinov2.models import build_model_from_cfg

    if saved_model_path is None:
        raise ValueError(f"--checkpoint is required for {modelname}")

    PATCH_SIZE = 4
    IMG_SIZE = 64

    cfg = OmegaConf.create({
        "student": {
            "arch": modelname.lower(),  # e.g. "microformer_small"
            "patch_size": PATCH_SIZE,
            "in_chans": 3,
        },
        "crops": {
            "global_crops_size": IMG_SIZE,
        },
    })

    model, _ = build_model_from_cfg(cfg, only_teacher=True)

    pretrained = torch.load(saved_model_path, map_location="cpu")

    if "teacher" in pretrained:
        raw_sd = pretrained["teacher"]
    elif "student" in pretrained:
        raw_sd = pretrained["student"]
    elif "backbone" in pretrained:
        raw_sd = {
            k.replace("_fsdp_wrapped_module.", ""): v
            for k, v in pretrained["backbone"].items()
        }
    elif isinstance(pretrained, dict):
        raw_sd = pretrained
    else:
        raise KeyError(
            f"Checkpoint at {saved_model_path} has neither 'teacher' nor 'student' "
            f"nor 'backbone' key. Available keys: {list(pretrained.keys())}"
        )

    state_dict = {}
    for k, v in raw_sd.items():
        if "dino_head" in k or "ibot_head" in k:
            continue
        if k.startswith("teacher.backbone."):
            state_dict[k[len("teacher.backbone."):]] = v
        elif k.startswith("student.backbone."):
            state_dict[k[len("student.backbone."):]] = v
        elif k.startswith("backbone."):
            state_dict[k[len("backbone."):]] = v
        else:
            state_dict[k] = v

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"[warn] Missing keys ({len(missing)}): {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")
    if len(state_dict) == 0:
        print("[warn] 0 tensors found in checkpoint -- check checkpoint format / prefixes above.")
    elif len(missing) == len(list(model.state_dict().keys())):
        print(
            "[warn] 0 tensors were actually loaded (all keys mismatched). "
            "The model is effectively random-init. Check that PATCH_SIZE/IMG_SIZE "
            "in get_microformer_finetuned match the checkpoint's actual training "
            "config (dinov2/configs/train/cell_dino/microformer_small_cellcrops.yaml)."
        )
    else:
        print(f"[ok] Loaded {len(state_dict) - len(missing)}/{len(state_dict)} backbone tensors.")

    model.eval()
    return model


def get_celldino_finetuned_downloaded(saved_model_path, modelname):
    """Load CellDINO model with automatic architecture inference"""
    pretrained = torch.load(saved_model_path, map_location="cpu")

    if "teacher" in pretrained:
        state_dict = pretrained["teacher"]
    elif "student" in pretrained:
        state_dict = pretrained["student"]
    elif "state_dict" in pretrained:
        state_dict = pretrained["state_dict"]
    elif "model" in pretrained:
        state_dict = pretrained["model"]
    else:
        state_dict = pretrained

    clean_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")
        new_key = new_key.replace("backbone.", "")
        clean_state_dict[new_key] = value

    patch_weight = clean_state_dict["patch_embed.proj.weight"]
    embed_dim, in_chans, patch_h, patch_w = patch_weight.shape

    assert patch_h == patch_w, f"Non-square patch embedding found: {patch_h}x{patch_w}"
    patch_size = patch_h

    if embed_dim == 384:
        depth = 12
        num_heads = 6
    elif embed_dim == 768:
        depth = 12
        num_heads = 12
    elif embed_dim == 1024:
        depth = 24
        num_heads = 16
    elif embed_dim == 1536:
        depth = 40
        num_heads = 24
    else:
        raise ValueError(f"Unsupported embed_dim={embed_dim}. Please specify architecture manually.")

    img_size = 128

    if "pos_embed" in clean_state_dict:
        num_tokens = clean_state_dict["pos_embed"].shape[1]
        num_patches = num_tokens - 1
        grid_size = int(num_patches ** 0.5)

        if grid_size * grid_size == num_patches:
            img_size = grid_size * patch_size

    model = DinoVisionTransformer(
        img_size=img_size,
        patch_size=patch_size,
        in_chans=in_chans,
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
    )

    keys_to_remove = []
    for k in clean_state_dict.keys():
        if any(x in k for x in ["head", "classifier", "fc", "dino_head", "ibot_head"]):
            keys_to_remove.append(k)

    for k in keys_to_remove:
        clean_state_dict.pop(k, None)

    if "pos_embed" in clean_state_dict:
        if clean_state_dict["pos_embed"].shape != model.pos_embed.shape:
            clean_state_dict.pop("pos_embed")

    msg = model.load_state_dict(clean_state_dict, strict=False)
    print("Load result:", msg)

    return model


def get_res50():
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")

    class Reshape(nn.Module):
        def forward(self, x):
            return x.reshape(x.shape[0], -1)

    model = nn.Sequential(*list(model.children())[:-3], nn.AdaptiveAvgPool2d((1, 1)), Reshape())

    return model


def get_full_res50():
    model = resnet.resnet50(weights="ResNet50_Weights.DEFAULT")
    return model


def get_transforms(model_name):
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    if model_name.lower() in ["ctranspath", "resnet50", "beit_fb", "resnet50_full"]:
        size = 224
    elif model_name.lower() == "lemon":
        size = 40
    elif model_name.lower() == "retccl":
        size = 256
    elif model_name.lower() == "kimianet":
        size = 1000
    elif model_name.lower() == "cellvitpp":
        size = 64
    elif model_name.lower() == "vit_small_patch7":
        size = 63
    elif model_name.lower() in ["microformer_small", "microformer_tiny", "microformer_base"]:
        # Must match training: crops.global_crops_size=64, patch_size=4 ->
        # scales (4,8,16) -> finest grid 16x16=256 tokens. Do NOT resize to
        # 224 here -- MicroFormer has no positional-embedding interpolation
        # to fall back on the way ViT does; it works at whatever resolution
        # it's given, but the *checkpoint's features* were learned at 64x64,
        # so evaluating at a different size is an out-of-distribution input,
        # not an equivalent one.
        size = 64
    elif model_name.lower() in [
        "dinov2_vits14",
        "dinov2_ssl",
        "dinov2_vitb14",
        "dinov2_vitl14",
        "dinov2_vitg14",
        "celldino",
    ]:
        size = 224
    else:
        raise ValueError(f"Model name '{model_name}' not found")

    size = (size, size)

    transforms_list = [
        transforms.Resize(size), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ]

    preprocess_transforms = transforms.Compose(transforms_list)
    return preprocess_transforms


class CellViTPlusPlus(nn.Module):
    def __init__(self, checkpoint_path, input_size=64):
        super().__init__()
        if sam_model_registry is None:
            raise ImportError("segment-anything not installed. Run: pip install segment-anything")
        if checkpoint_path is None:
            raise ValueError("--checkpoint is required for cellvitpp")
        sam = sam_model_registry["vit_h"](checkpoint=None)
        self.encoder = sam.image_encoder

    def forward(self, x):
        feats = self.encoder(x)
        feats = feats.mean(dim=[2, 3])
        return feats