"""Public model API of the ``llava`` package.

Importing this package imports ``llava.model.language_model.llava_qwen``, which as a side
effect registers the ``"llava_qwen"`` model type with ``transformers.AutoConfig`` /
``AutoModelForCausalLM`` / ``AutoTokenizer`` so that DriveIK checkpoints (whose
``config.json`` says ``model_type: llava_qwen``) can be loaded with the Auto classes.
"""

from .language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM

__all__ = ["LlavaQwenConfig", "LlavaQwenForCausalLM"]
