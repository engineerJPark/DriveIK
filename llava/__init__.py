"""Cached-feature Qwen model components used by DriveIK training.

This package is a trimmed fork of the LLaVA / LLaVA-NeXT code base. In DriveIK it
provides only the language-model side of the driving VLA:

* ``llava.constants``      - negative placeholder ids (``<SCENE>``, ``<EGO>``, ...) and
                             the literal tag tokens added to the tokenizer.
* ``llava.conversation``   - the single ChatML conversation template ("v3") whose system
                             prompt contains the ``<SCENE>`` / ``<EGO>`` placeholders.
* ``llava.mm_utils``       - ``tokenizer_uniad_token``, which tokenizes a prompt and
                             replaces each placeholder string by its negative id.
* ``llava.model``          - ``LlavaQwenForCausalLM`` (Qwen2 + multimodal glue). Its
                             ``prepare_inputs_labels_for_multimodal_uniad_vlm`` (in
                             ``llava/model/llava_arch.py``) projects the frozen DINOv3
                             camera tokens (``bev_tokens``) and the ego status vector into
                             the LLM embedding space and splices them in place of the
                             placeholder ids.
* ``llava.utils``          - rank-aware ``print`` helpers.

There is no raw-image vision encoder here: training and evaluation read cached
DINOv3 feature files from disk (see README.md, "Data").
"""
