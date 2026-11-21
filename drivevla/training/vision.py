"""DINOv3 camera encoder: raw NAVSIM camera images -> visual-state tokens.

DriveIK's visual state is a set of frozen DINOv3 ViT-S/16 patch tokens of the three
front cameras. This module computes them from the raw camera images while training
and evaluating, with the recipe the released checkpoints were trained with:

1. each of ``CAMERAS`` (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT; NAVSIM CAM_L0 /
   CAM_F0 / CAM_R0) is decoded with PIL as RGB, scaled to ``[0, 1]`` floats and resized
   to ``IMAGE_HW`` = (288, 160) (height, width; bilinear, antialiased, no crop, so the
   16:9 image is squeezed);
2. normalised with the ImageNet mean / std;
3. passed through DINOv3 (``facebook/dinov3-vits16-pretrain-lvd1689m``) in fp16 on GPU;
4. the output of the final LayerNorm (HF ``last_hidden_state``) is kept for the patch
   tokens only: the CLS and the 4 register tokens are dropped, leaving 18 x 10 = 180
   tokens per camera in row-major order;
5. the cameras are concatenated in ``CAMERAS`` order: ``(540, 384)``, fp16.

Steps 1-2 run in the dataset (DataLoader workers, CPU, float32), steps 3-5 in the model
forward (``model.BEVLatentActionWorldModelMixin._encode_camera_frames``).

The ViT is implemented here (:class:`DINOv3ViT`) instead of being loaded through
``transformers.AutoModel``, because Transformers only supports DINOv3 from version 4.56
and the training stack runs on 4.44. It mirrors the Hugging Face ``DINOv3ViTModel``
(inference path, SDPA attention) parameter for parameter, so the official Hugging Face
checkpoint loads with ``strict=True``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# DINOv3 checkpoint of the released recipes (Hugging Face id; a local directory with
# config.json + model.safetensors also works).
DEFAULT_DINOV3_MODEL = "facebook/dinov3-vits16-pretrain-lvd1689m"
# Cameras, in token order, keyed as in info["cams"] of the NAVSIM infos pickles.
CAMERAS: Tuple[str, ...] = ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT")
# Network input size (height, width): 18 x 10 patches of 16 px per camera.
IMAGE_HW: Tuple[int, int] = (288, 160)
# ImageNet normalisation used by DINOv3.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Image preprocessing (CPU, dataset side)
# ---------------------------------------------------------------------------


def load_camera_image(path: str, image_hw: Tuple[int, int] = IMAGE_HW) -> torch.Tensor:
    """Read one camera image and preprocess it for DINOv3.

    Numerically the same as ``torchvision.transforms.functional``'s ``to_tensor``,
    ``resize(..., antialias=True)`` and ``normalize`` (checked in the tests), without
    depending on torchvision.

    Args:
        path: image file (NAVSIM: 1920 x 1080 JPEG).
        image_hw: output (height, width).

    Returns:
        float32 tensor ``(3, H, W)``, ImageNet-normalised.
    """
    from PIL import Image

    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8)
    # HWC uint8 -> CHW float in [0, 1] (to_tensor).
    x = torch.from_numpy(rgb.copy()).permute(2, 0, 1).contiguous().to(torch.float32).div(255)
    # Antialiased bilinear resize of the float image (torchvision's tensor resize).
    x = F.interpolate(
        x.unsqueeze(0), size=list(image_hw), mode="bilinear", align_corners=False, antialias=True
    ).squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=torch.float32).view(3, 1, 1)
    return (x - mean) / std


def load_camera_images(
    paths: Sequence[str], image_hw: Tuple[int, int] = IMAGE_HW
) -> torch.Tensor:
    """Preprocess the cameras of one frame: float32 ``(len(paths), 3, H, W)``."""
    return torch.stack([load_camera_image(str(p), image_hw) for p in paths], dim=0)


# ---------------------------------------------------------------------------
# DINOv3 ViT (mirrors transformers' DINOv3ViTModel, inference only)
# ---------------------------------------------------------------------------


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the two halves of the last dim: ``(a, b) -> (-b, a)``."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class _Embeddings(nn.Module):
    """Patch embedding plus the CLS and register tokens (``embeddings.*``)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        patch = int(cfg["patch_size"])
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden))
        # Only used for masked pre-training; kept so the checkpoint loads strictly.
        self.mask_token = nn.Parameter(torch.zeros(1, 1, hidden))
        self.register_tokens = nn.Parameter(torch.zeros(1, int(cfg["num_register_tokens"]), hidden))
        self.patch_embeddings = nn.Conv2d(
            int(cfg.get("num_channels", 3)), hidden, kernel_size=patch, stride=patch
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, 1 + R + N, C)``: CLS, R registers, N patches."""
        b = pixel_values.shape[0]
        patches = self.patch_embeddings(pixel_values).flatten(2).transpose(1, 2)
        return torch.cat(
            [self.cls_token.expand(b, -1, -1), self.register_tokens.expand(b, -1, -1), patches],
            dim=1,
        )


class _Attention(nn.Module):
    """Multi-head self-attention with 2-D RoPE on the patch tokens (``attention.*``)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        hidden = int(cfg["hidden_size"])
        self.num_heads = int(cfg["num_attention_heads"])
        self.head_dim = hidden // self.num_heads
        self.scaling = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden, hidden, bias=bool(cfg.get("query_bias", True)))
        self.k_proj = nn.Linear(hidden, hidden, bias=bool(cfg.get("key_bias", False)))
        self.v_proj = nn.Linear(hidden, hidden, bias=bool(cfg.get("value_bias", True)))
        self.o_proj = nn.Linear(hidden, hidden, bias=bool(cfg.get("proj_bias", True)))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        q, k, v = (
            proj(x).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        )
        # RoPE on the patch tokens only; the CLS / register prefix is left unrotated.
        n_prefix = n - cos.shape[-2]
        q_pre, q_pat = q.split((n_prefix, cos.shape[-2]), dim=-2)
        k_pre, k_pat = k.split((n_prefix, cos.shape[-2]), dim=-2)
        q = torch.cat((q_pre, q_pat * cos + _rotate_half(q_pat) * sin), dim=-2)
        k = torch.cat((k_pre, k_pat * cos + _rotate_half(k_pat) * sin), dim=-2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, scale=self.scaling, is_causal=False)
        return self.o_proj(out.transpose(1, 2).reshape(b, n, -1))


class _LayerScale(nn.Module):
    """Per-channel residual scale (``layer_scale*.lambda1``)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.lambda1 = nn.Parameter(torch.ones(int(cfg["hidden_size"])))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.lambda1


class _MLP(nn.Module):
    """Feed-forward block (``mlp.*``): plain or gated (SwiGLU-style) variant."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        hidden, inter = int(cfg["hidden_size"]), int(cfg["intermediate_size"])
        bias = bool(cfg.get("mlp_bias", True))
        self.gated = bool(cfg.get("use_gated_mlp", False))
        if self.gated:
            self.gate_proj = nn.Linear(hidden, inter, bias=bias)
        self.up_proj = nn.Linear(hidden, inter, bias=bias)
        self.down_proj = nn.Linear(inter, hidden, bias=bias)
        act = str(cfg.get("hidden_act", "gelu"))
        acts = {"gelu": F.gelu, "silu": F.silu, "swish": F.silu, "relu": F.relu}
        if act not in acts:
            raise ValueError(f"Unsupported DINOv3 hidden_act: {act}")
        self.act = acts[act]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.gated:
            return self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act(self.up_proj(x)))


class _Layer(nn.Module):
    """Pre-norm transformer block with LayerScale (``layer.<i>.*``)."""

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        hidden, eps = int(cfg["hidden_size"]), float(cfg.get("layer_norm_eps", 1e-5))
        self.norm1 = nn.LayerNorm(hidden, eps=eps)
        self.attention = _Attention(cfg)
        self.layer_scale1 = _LayerScale(cfg)
        self.norm2 = nn.LayerNorm(hidden, eps=eps)
        self.mlp = _MLP(cfg)
        self.layer_scale2 = _LayerScale(cfg)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.layer_scale1(self.attention(self.norm1(x), cos, sin))
        return x + self.layer_scale2(self.mlp(self.norm2(x)))


class DINOv3ViT(nn.Module):
    """DINOv3 ViT backbone; ``forward`` returns the final-LayerNorm token states.

    Attribute names follow Hugging Face ``DINOv3ViTModel``, so its ``state_dict``
    loads strictly. Only the inference path is implemented: the RoPE coordinate
    augmentations (shift / jitter / rescale) apply only while training DINOv3 itself
    and are never used here, and dropout / drop-path are absent.
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = dict(cfg)
        self.patch_size = int(cfg["patch_size"])
        self.hidden_size = int(cfg["hidden_size"])
        self.num_register_tokens = int(cfg["num_register_tokens"])
        self.rope_theta = float(cfg.get("rope_theta", 100.0))
        self.head_dim = self.hidden_size // int(cfg["num_attention_heads"])
        self.embeddings = _Embeddings(cfg)
        self.layer = nn.ModuleList([_Layer(cfg) for _ in range(int(cfg["num_hidden_layers"]))])
        self.norm = nn.LayerNorm(self.hidden_size, eps=float(cfg.get("layer_norm_eps", 1e-5)))

    def _rope(self, h: int, w: int, device: torch.device, dtype: torch.dtype):
        """cos / sin ``(h * w, head_dim)`` of the 2-D RoPE, computed in float32 and cast
        to ``dtype`` (as in Hugging Face's DINOv3ViTRopePositionEmbedding)."""
        with torch.autocast(device_type=device.type, enabled=False):
            # Built on the CPU and then moved, like the HF buffer (bit-identical values).
            inv_freq = (
                1 / self.rope_theta ** torch.arange(0, 1, 4 / self.head_dim, dtype=torch.float32)
            ).to(device)
            # Patch-centre coordinates in [-1, 1], (y, x) per patch, row-major.
            ch = torch.arange(0.5, h, dtype=torch.float32, device=device) / h
            cw = torch.arange(0.5, w, dtype=torch.float32, device=device) / w
            coords = torch.stack(torch.meshgrid(ch, cw, indexing="ij"), dim=-1).flatten(0, 1)
            coords = 2.0 * coords - 1.0
            angles = 2 * math.pi * coords[:, :, None] * inv_freq[None, None, :]
            angles = angles.flatten(1, 2).tile(2)
            return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """``(B, 3, H, W)`` -> ``(B, 1 + R + N, C)`` (HF ``last_hidden_state``)."""
        dtype = self.embeddings.patch_embeddings.weight.dtype
        pixel_values = pixel_values.to(dtype)
        h, w = pixel_values.shape[-2] // self.patch_size, pixel_values.shape[-1] // self.patch_size
        cos, sin = self._rope(h, w, pixel_values.device, dtype)
        x = self.embeddings(pixel_values)
        for layer in self.layer:
            x = layer(x, cos, sin)
        return self.norm(x)


def load_dinov3(name_or_path: str = DEFAULT_DINOV3_MODEL) -> DINOv3ViT:
    """Build :class:`DINOv3ViT` from a Hugging Face DINOv3 checkpoint.

    Args:
        name_or_path: a local directory holding ``config.json`` and
            ``model.safetensors``, or a Hugging Face model id. The ``facebook/dinov3-*``
            repositories are gated: accept the license on the model page and log in
            (``huggingface-cli login``) once; afterwards the files come from the local
            Hugging Face cache (also with ``HF_HUB_OFFLINE=1``).

    Returns:
        The backbone in float32, in eval mode, with ``requires_grad=False``.
    """
    from safetensors.torch import load_file

    p = Path(str(name_or_path)).expanduser()
    if p.is_dir():
        cfg_file, weight_file = p / "config.json", p / "model.safetensors"
    else:
        from huggingface_hub import hf_hub_download

        cfg_file = Path(hf_hub_download(str(name_or_path), "config.json"))
        weight_file = Path(hf_hub_download(str(name_or_path), "model.safetensors"))
    cfg = json.loads(Path(cfg_file).read_text())
    if cfg.get("model_type") not in (None, "dinov3_vit"):
        raise ValueError(f"{name_or_path} is not a DINOv3 ViT checkpoint (model_type={cfg.get('model_type')!r})")
    model = DINOv3ViT(cfg)
    state = load_file(str(weight_file))
    model.load_state_dict({k: v.to(torch.float32) for k, v in state.items()}, strict=True)
    return model.eval().requires_grad_(False)


# ---------------------------------------------------------------------------
# Frozen encoder used by the model forward
# ---------------------------------------------------------------------------


class CameraEncoder:
    """Frozen DINOv3 that maps preprocessed camera images to visual-state tokens.

    This is deliberately not an ``nn.Module``: train.py stores it as a plain attribute
    of the model (``model.camera_encoder``), so the DINOv3 weights stay out of the
    model's ``state_dict`` and checkpoints, the optimizer, DDP / ZeRO-3 and the model's
    own ``.to()`` / dtype casts. It moves itself to the input device on first use and
    always runs in fp16 on GPU (float32 on CPU) with autocast disabled, as for the
    released checkpoints, independent of the training precision.
    A deep copy of the model (the frozen GRPO reference) shares the same encoder.

    Attributes:
        model_name: checkpoint the backbone was loaded from.
        cameras: camera names in token order.
        image_hw: expected input (height, width).
        hidden_size: token width C (384 for ViT-S).
        patches_per_camera: patch tokens per camera (180 for 288 x 160 / 16).
        num_tokens: tokens per frame (cameras x patches_per_camera, 540).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_DINOV3_MODEL,
        cameras: Sequence[str] = CAMERAS,
        image_hw: Tuple[int, int] = IMAGE_HW,
    ):
        self.model_name = str(model_name)
        self.cameras = tuple(cameras)
        self.image_hw = (int(image_hw[0]), int(image_hw[1]))
        self.vit = load_dinov3(self.model_name)
        p = self.vit.patch_size
        if self.image_hw[0] % p or self.image_hw[1] % p:
            raise ValueError(f"image size {self.image_hw} is not a multiple of the patch size {p}")
        self.hidden_size = int(self.vit.hidden_size)
        self.patches_per_camera = (self.image_hw[0] // p) * (self.image_hw[1] // p)
        self.num_tokens = self.patches_per_camera * len(self.cameras)
        self._device: Optional[torch.device] = None

    def __deepcopy__(self, memo):
        # Stateless and frozen: share it instead of copying the backbone.
        return self

    def __repr__(self) -> str:
        return (
            f"CameraEncoder({self.model_name}, cameras={list(self.cameras)}, "
            f"image_hw={self.image_hw}, tokens={self.num_tokens}x{self.hidden_size})"
        )

    @torch.no_grad()
    def __call__(self, images: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Encode frames of preprocessed camera images.

        Args:
            images: ``(F, n_cams, 3, H, W)`` from :func:`load_camera_images` (one row
                per frame).
            device: device to run on (the model's device).

        Returns:
            ``(F, n_cams * patches_per_camera, C)`` patch tokens on ``device``; fp16 on
            GPU, float32 on CPU. Computed under ``no_grad``.
        """
        device = torch.device(device)
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        if self._device != device:
            self.vit.to(device=device, dtype=dtype)
            self._device = device
        n_frames, n_cams = int(images.shape[0]), int(images.shape[1])
        if n_cams != len(self.cameras) or tuple(images.shape[-2:]) != self.image_hw:
            raise ValueError(
                f"camera images {tuple(images.shape)} do not match {self!r}"
            )
        flat = images.reshape(n_frames * n_cams, *images.shape[2:]).to(device=device, dtype=dtype)
        with torch.autocast(device_type=device.type, enabled=False):
            tokens = self.vit(flat)[:, -self.patches_per_camera :, :]
        return tokens.reshape(n_frames, n_cams * self.patches_per_camera, self.hidden_size)
