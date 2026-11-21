#    Copyright 2024 Hao Zhang
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

"""Qwen2-based LLaVA model used as the DriveIK language backbone.

Class layout (multiple inheritance, LLaVA style):

* ``LlavaQwenConfig``  - ``Qwen2Config`` with ``model_type = "llava_qwen"``.
* ``LlavaQwenModel``   - ``LlavaMetaModel`` + ``Qwen2Model``: the decoder stack plus the
  multimodal modules created in ``LlavaMetaModel.__init__`` (``ego_projector`` and the
  ``mm_projector_*`` projectors for the DINOv3 camera tokens).
* ``LlavaQwenForCausalLM`` - ``Qwen2ForCausalLM`` + ``LlavaMetaForCausalLM``: adds the LM
  head, the input-preparation routine that splices projected camera / ego features into
  the token embeddings (``prepare_inputs_labels_for_multimodal_uniad_vlm`` from
  ``llava/model/llava_arch.py``), and a multimodal-aware ``forward``.

DriveIK training (``drivevla/training/train.py``) loads ``LlavaQwenForCausalLM`` and then
binds ``BEVLatentActionWorldModelMixin.forward`` (``drivevla/training/model.py``) onto the
instance, replacing the class-level ``forward`` defined here. That mixin adds the
next-visual-state loss and the IK diffusion head.

At import time the module registers ``"llava_qwen"`` with the ``transformers`` Auto
classes (bottom of the file).
"""

from typing import List, Optional, Tuple, Union, Dict
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from transformers import Qwen2Config, Qwen2Model, Qwen2ForCausalLM, Qwen2Tokenizer


class LlavaQwenConfig(Qwen2Config):
    """``Qwen2Config`` registered under ``model_type="llava_qwen"``.

    Besides the Qwen2 fields, DriveIK checkpoints store extra attributes read by the
    multimodal code, e.g. ``use_mm_proj``, ``mm_hidden_size``, ``mm_projector_type``,
    ``ego_feature_dim`` and ``world_model_use_heads``. They are kept by the
    ``PretrainedConfig`` machinery as plain attributes.
    """

    model_type = "llava_qwen"


def _should_build_world_model_heads(config) -> bool:
    """Return True if ``config`` asks for the auxiliary ``traj_head`` / ``point_head``.

    True when ``config.world_model_use_heads`` is set (stored in config.json once the
    heads have been built). DriveIK configs have no vision tower, so for a config
    without the flag the heads are built via the ``use_mm_proj`` / ``mm_hidden_size``
    conditions in ``LlavaQwenForCausalLM.__init__``.
    """
    return bool(getattr(config, "world_model_use_heads", False))


def _build_xy_head(hidden_size: int, out_dim: int = 2) -> nn.Sequential:
    """Two-layer GELU MLP ``(..., hidden_size) -> (..., out_dim)``.

    Used for ``traj_head`` / ``point_head`` (``out_dim=2``, i.e. an (x, y) output). Its
    layer indices 0 and 2 match what ``_maybe_build_xy_head_from_state_dict`` expects.
    """
    return nn.Sequential(
        nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, out_dim)
    )


class LlavaQwenModel(LlavaMetaModel, Qwen2Model):
    """Qwen2 decoder stack plus DriveIK's multimodal modules.

    MRO: ``LlavaQwenModel -> LlavaMetaModel -> Qwen2Model``. ``LlavaMetaModel.__init__``
    first calls ``Qwen2Model.__init__`` (via ``super()``) and then adds ``ego_projector``
    and, for cached-feature configs, ``mm_projector_{scene,track,map,bev}``.
    """

    config_class = LlavaQwenConfig

    def __init__(self, config: Qwen2Config):
        """Build the decoder and the multimodal modules from ``config``."""
        super(LlavaQwenModel, self).__init__(config)


class LlavaQwenForCausalLM(Qwen2ForCausalLM, LlavaMetaForCausalLM):
    """Causal LM used by DriveIK: Qwen2 + DINOv3 camera-token / ego feature injection.

    Modules:
        model (LlavaQwenModel): decoder + ``ego_projector`` + ``mm_projector_*``.
        lm_head (nn.Linear): ``hidden_size -> vocab_size``, no bias (weights tied to the
            input embedding when ``config.tie_word_embeddings`` is True, as for
            Qwen2.5-0.5B).
        traj_head / point_head (optional): ``_build_xy_head`` MLPs
            ``hidden_size -> 2``. Not used by ``forward`` in this file;
            ``drivevla/training/checkpoint.py`` excludes them from the optimizer
            parameter groups.

    Multimodal inputs are passed as extra keyword arguments:
        uniad_pth: dict containing ``"bev_tokens"`` of shape ``(N_tok, mm_hidden_size)``
            (``(540, 384)`` DINOv3 tokens in the released recipes; the world-model
            forward encodes camera images into this form first).
        uniad_data: dict ``{"img_metas": [{"can_bus": Tensor[ego_dim]}]}`` carrying the
            ego-status vector for ``<EGO>``.
    """

    config_class = LlavaQwenConfig
    # Missing traj_head / point_head weights are not reported when loading checkpoints
    # that were saved without them.
    _keys_to_ignore_on_load_missing = ["traj_head\\..*", "point_head\\..*"]

    def __init__(self, config):
        """Build the model from ``config``.

        Steps: run ``Qwen2ForCausalLM.__init__``; force ``model_type`` and the RoPE
        scaling field; replace ``self.model`` / ``self.lm_head`` with the LLaVA versions;
        optionally add ``traj_head`` / ``point_head``; register the load-time autobuild
        hook; finally ``post_init()`` (weight init + embedding tying).
        """
        # Plain Qwen2 causal-LM construction (also builds a Qwen2Model and lm_head that
        # are replaced a few lines below).
        Qwen2ForCausalLM.__init__(self, config)
        config.model_type = "llava_qwen"
        # RoPE scaling: if the config carries a `rope_parameters` field, write a
        # "default" rope_scaling dict with the config's rope_theta (fallback 10000.0);
        # otherwise clear rope_scaling. Either way any other scaling type stored in
        # config.rope_scaling is overwritten.
        if getattr(config, "rope_parameters", None) is not None:
            config.rope_scaling = {
                "rope_type": "default",
                "rope_theta": getattr(config, "rope_theta", 10000.0),
            }
        else:
            config.rope_scaling = None
        # Swap in the multimodal decoder (Qwen2Model + ego / visual projectors) and a
        # fresh LM head.
        self.model = LlavaQwenModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Auxiliary (x, y) heads for world-model / cached-feature configs. Saved DriveIK
        # checkpoints have use_mm_proj=True and mm_hidden_size=384 in config.json, so the
        # branch runs when they are reloaded; a fresh Qwen2.5 base config does not
        # trigger it (train.py sets use_mm_proj only after construction).
        if (
            _should_build_world_model_heads(config)
            or bool(getattr(config, "use_mm_proj", False))
            or getattr(config, "mm_hidden_size", None) is not None
        ):
            if not hasattr(self, "traj_head"):
                self.traj_head = _build_xy_head(config.hidden_size, out_dim=2)
            if not hasattr(self, "point_head"):
                self.point_head = _build_xy_head(config.hidden_size, out_dim=2)
            # Persist the decision in the config so reloads build the heads again.
            try:
                self.config.world_model_use_heads = True
            except Exception:
                pass
        # Make from_pretrained() able to create projector / head modules that exist in a
        # checkpoint but were not built from the config (see the hook's docstring).
        self._register_world_model_autobuild_hook()
        # Hugging Face post-construction (weight initialization, embedding tying, ...).
        self.post_init()

    def get_model(self):
        """Return the inner ``LlavaQwenModel`` (required by ``LlavaMetaForCausalLM``)."""
        return self.model

    def _register_world_model_autobuild_hook(self) -> None:
        """Auto-create optional BEV projector / heads when loading checkpoints.

        HF `from_pretrained()` loads the checkpoint right after __init__. If a checkpoint was
        trained with extra modules (e.g., `model.mm_projector_bev`, `traj_head`) but the current
        config does not trigger their construction, those weights show up as "not used" and
        are silently dropped. This hook inspects the incoming state_dict and instantiates the
        missing modules with shapes inferred from the checkpoint so resume/inference are robust.
        """
        # Register at most once per instance.
        if getattr(self, "_wm_autobuild_hook_registered", False):
            return
        self._wm_autobuild_hook_registered = True

        def _hook(*args):
            """``load_state_dict`` pre-hook that builds missing modules from the keys.

            Accepts both hook signatures: the public
            ``register_load_state_dict_pre_hook`` passes 8 arguments (module first);
            the private ``_register_load_state_dict_pre_hook(..., with_module=False)``
            passes 7 (no module, so ``self`` is used). Only acts for the root module
            (empty prefix). Exceptions are appended to ``error_msgs`` instead of raised.
            """
            # Unpack according to the calling convention; ignore unknown arities.
            if len(args) == 8:
                (
                    module,
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                ) = args
            elif len(args) == 7:
                (
                    state_dict,
                    prefix,
                    local_metadata,
                    strict,
                    missing_keys,
                    unexpected_keys,
                    error_msgs,
                ) = args
                module = self
            else:
                return
            # Submodule loads (non-empty prefix) are not handled here.
            if prefix not in ("", None):
                return
            # Create mm_projector_bev / traj_head / point_head if needed, inferring their
            # shapes from the incoming state_dict.
            try:
                module._maybe_build_mm_projector_bev_from_state_dict(state_dict)
                module._maybe_build_xy_head_from_state_dict("traj_head", state_dict)
                module._maybe_build_xy_head_from_state_dict("point_head", state_dict)
            except Exception as e:
                try:
                    error_msgs.append(f"[world_model_autobuild_hook] {type(e).__name__}: {e}")
                except Exception:
                    pass

        # Prefer the public PyTorch API; fall back to the older private one; give up
        # silently if neither exists.
        try:
            self.register_load_state_dict_pre_hook(_hook)
        except Exception:
            try:
                self._register_load_state_dict_pre_hook(_hook, with_module=False)
            except Exception:
                pass

    def _maybe_build_xy_head_from_state_dict(
        self, name: str, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Create ``self.<name>`` (``traj_head`` / ``point_head``) from checkpoint shapes.

        Args:
            name: Attribute name of the head.
            state_dict: Checkpoint tensors about to be loaded.

        Behavior:
            * If the head already exists with real (non-meta) parameters, do nothing.
            * Otherwise remove it and look for ``{prefix}{name}.0.weight`` and
              ``{prefix}{name}.2.weight`` (prefix ``""`` or ``"module."``); if both exist,
              build ``Sequential(Linear, GELU, Linear)`` with in/out sizes taken from the
              weight shapes and biases only where the checkpoint has them.
            * Else, if ``{prefix}{name}.weight`` exists, build a single ``Linear``.
            * If no matching keys are found, the attribute stays removed (or ``None``).

        New modules are created with PyTorch defaults (CPU, default dtype); the
        subsequent state-dict load fills in the weights.
        """
        # Existing head: keep it unless any of its parameters is still on `meta`.
        if hasattr(self, name):
            try:
                mod = getattr(self, name)
                if mod is not None and (
                    not any((getattr(p, "is_meta", False) for p in mod.parameters()))
                ):
                    return
            except Exception:
                return
            # Drop the meta-device head so it can be rebuilt below.
            try:
                delattr(self, name)
            except Exception:
                setattr(self, name, None)
        # Keys may be stored with or without a DDP-style "module." prefix.
        prefixes = ["", "module."]
        for p in prefixes:
            # Two-layer MLP layout: Linear at index 0, GELU at 1, Linear at 2.
            # nn.Linear weights are (out_features, in_features).
            k0 = f"{p}{name}.0.weight"
            k2 = f"{p}{name}.2.weight"
            if k0 in state_dict and k2 in state_dict:
                w0 = state_dict[k0]
                w2 = state_dict[k2]
                b0 = f"{p}{name}.0.bias" in state_dict
                b2 = f"{p}{name}.2.bias" in state_dict
                layer0 = nn.Linear(int(w0.shape[1]), int(w0.shape[0]), bias=b0)
                layer2 = nn.Linear(int(w2.shape[1]), int(w2.shape[0]), bias=b2)
                setattr(self, name, nn.Sequential(layer0, nn.GELU(), layer2))
                return
            # Single Linear layout.
            kw = f"{p}{name}.weight"
            if kw in state_dict:
                w = state_dict[kw]
                kb = f"{p}{name}.bias"
                has_b = kb in state_dict
                setattr(self, name, nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=has_b))
                return

    def _maybe_build_mm_projector_bev_from_state_dict(
        self, state_dict: Dict[str, torch.Tensor]
    ) -> None:
        """Create ``self.model.mm_projector_bev`` from checkpoint shapes if needed.

        Same idea as ``_maybe_build_xy_head_from_state_dict`` for the camera-token
        projector: keep an existing projector whose parameters are all real; otherwise
        build ``Linear`` (keys ``model.mm_projector_bev.weight``, with an optional
        ``"module."`` prefix). If nothing matches, an existing meta projector is left
        as ``None``.

        Args:
            state_dict: Checkpoint tensors about to be loaded.
        """
        # Nothing to do before the inner model exists.
        m = getattr(self, "model", None)
        if m is None:
            return
        # Existing projector: keep it unless it still has meta parameters, in which case
        # clear it so it can be rebuilt.
        proj = getattr(m, "mm_projector_bev", None)
        if proj is not None:
            try:
                if not any((getattr(p, "is_meta", False) for p in proj.parameters())):
                    return
            except Exception:
                return
            try:
                m.mm_projector_bev = None
            except Exception:
                return
        # Search both key spellings ("" and "module." prefix).
        prefixes = ["", "module."]
        for p in prefixes:
            # Single Linear projector (the released "linear" checkpoints: 384 -> 896).
            kw = f"{p}model.mm_projector_bev.weight"
            if kw in state_dict:
                w = state_dict[kw]
                kb = f"{p}model.mm_projector_bev.bias"
                has_b = kb in state_dict
                m.mm_projector_bev = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=has_b)
                return

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        return_dict: Optional[bool] = None,
        modalities: Optional[List[str]] = ["image"],
        cache_position=None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Multimodal causal-LM forward (class-level version).

        DriveIK training/evaluation rebinds ``forward`` on the instance to
        ``BEVLatentActionWorldModelMixin.forward`` (``drivevla/training/train.py``); this
        method is the plain language-model path.

        Args:
            input_ids: ``(B, L)`` token ids that may contain negative placeholder ids
                (``<SCENE>`` = -201, ``<EGO>`` = -205, ...).
            attention_mask: ``(B, L)``, 1 for real tokens.
            position_ids, past_key_values, use_cache, output_attentions,
            output_hidden_states, return_dict, cache_position: forwarded to Qwen2.
            inputs_embeds: If given, multimodal preparation is skipped and these
                ``(B, L', hidden)`` embeddings are used directly.
            labels: ``(B, L)`` LM targets (``IGNORE_INDEX`` = -100 where unsupervised).
            images, image_sizes, modalities: upstream LLaVA arguments. ``images`` is not
                forwarded (``None`` is passed instead); ``modalities`` is passed through.
            **kwargs: May contain ``uniad_pth`` (frame dict with ``"bev_tokens"``) and
                ``uniad_data`` (``{"img_metas": [{"can_bus": ...}]}``); these are popped
                here. Remaining kwargs go to ``Qwen2ForCausalLM.forward``.

        Returns:
            ``CausalLMOutputWithPast`` from Qwen2 (``loss`` is the shifted token
            cross-entropy when ``labels`` are given).

        Side effects:
            In training mode, the ego-curriculum step counter is incremented by the
            preparation step.
        """
        # Pull the DriveIK-specific inputs out of kwargs so they are not forwarded to
        # the Qwen2 forward, which would not accept them.
        uniad_pth = kwargs.pop("uniad_pth", None)
        uniad_data = kwargs.pop("uniad_data", None)
        # Build inputs_embeds by replacing placeholder ids with projected camera tokens /
        # the ego embedding. labels, attention_mask and position_ids are expanded to the
        # new length. input_ids comes back as None unless the routine returned early
        # (no features given, or a single-token decoding step).
        if inputs_embeds is None:
            input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal_uniad_vlm(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    None,
                    modalities,
                    image_sizes,
                    uniad_data=uniad_data,
                    uniad_pth=uniad_pth,
                )
            )
        # Qwen2ForCausalLM.forward computes logits and, if labels are given, the
        # next-token cross-entropy loss.
        if cache_position is not None:
            kwargs["cache_position"] = cache_position
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

# Make "llava_qwen" checkpoints loadable through the Auto classes (config, model and the
# Qwen2 tokenizer).
AutoConfig.register("llava_qwen", LlavaQwenConfig)
AutoModelForCausalLM.register(LlavaQwenConfig, LlavaQwenForCausalLM)
AutoTokenizer.register(LlavaQwenConfig, Qwen2Tokenizer)
