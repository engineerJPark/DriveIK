"""Projector that maps the DINOv3 visual features into the LLM embedding space.

In DriveIK the "visual features" are frozen DINOv3 patch tokens (``bev_tokens``,
computed by ``drivevla/training/vision.py``). In the released recipes these are
``(540, 384)`` fp16 tensors: 3 front cameras (CAM_FRONT_LEFT, CAM_FRONT, CAM_FRONT_RIGHT) x 18 x 10
patches of a ViT-S/16 at 288 x 160 input, 384 channels. A projector built here maps
``(..., mm_hidden_size)`` -> ``(..., hidden_size)`` (Qwen2.5-0.5B: 896), and the result is
spliced into the input embeddings at the ``<SCENE>`` placeholder
(``llava/model/llava_arch.py``).

``build_vision_projector`` is called from ``LlavaMetaModel.__init__`` and from
``drivevla/training/data.py::ensure_mm_projectors``. The released checkpoints
use ``mm_projector_type="linear"`` with ``mm_hidden_size=384``.
"""

import torch.nn as nn


def build_vision_projector(config, delay_load=False, **kwargs):
    """Build the visual-feature projector selected by ``config.mm_projector_type``.

    Args:
        config: Model config. Reads ``mm_projector_type`` (default ``"linear"``),
            ``mm_hidden_size`` (input width, e.g. 384 for the DINOv3 caches) and
            ``hidden_size`` (LLM width).
        delay_load: Unused; kept for upstream LLaVA signature compatibility.
        **kwargs: Unused.

    Only ``"linear"`` is supported: ``Linear(mm_hidden_size, hidden_size)``.

    Returns:
        nn.Module mapping ``(..., mm_hidden_size)`` -> ``(..., hidden_size)``.

    Raises:
        ValueError: for any other ``mm_projector_type``.
    """
    projector_type = getattr(config, "mm_projector_type", "linear")
    # Single affine map (the projector used by the released checkpoints).
    if projector_type == "linear":
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    # Unknown name: fail loudly. (LlavaMetaModel.__init__ catches this and falls back to
    # a plain Linear; ensure_mm_projectors does not.)
    raise ValueError(f"Unknown projector type: {projector_type}")
