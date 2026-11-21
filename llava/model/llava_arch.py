#    Copyright 2023 Haotian Liu
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

"""Multimodal glue between DINOv3 camera features and the Qwen2 LLM.

This is the DriveIK version of LLaVA's ``llava_arch.py``. There is no image encoder
here: the camera features arrive as frozen DINOv3 patch tokens (``bev_tokens``),
encoded from the camera images just before this step
(``drivevla/training/vision.py``, called from the world-model forward), plus a
per-sample ego-status vector (``can_bus``).

Data flow of one training / evaluation sample (batch size 1 in DriveIK):

1. ``drivevla/training/data.py`` renders the v3 ChatML prompt (``llava/conversation.py``)
   whose system block contains ``<scene_start><SCENE><scene_end>`` and
   ``<ego_start><EGO><ego_end>``, and tokenizes it with ``tokenizer_uniad_token``, which
   writes ``SCENE_TOKEN_INDEX`` (-201) and ``EGO_TOKEN_INDEX`` (-205) into ``input_ids``.
2. ``LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal_uniad_vlm`` (below):

   * projects ``uniad_pth["bev_tokens"]`` (``(540, 384)`` for the released recipes:
     3 front cameras x 18 x 10 patches = 540 tokens with 384 channels each) with
     ``mm_projector_bev`` to ``(540, hidden_size)``;
   * projects ``uniad_data["img_metas"][-1]["can_bus"]`` with ``ego_projector`` to one
     ``(1, hidden_size)`` row, or uses a zero row if the ego curriculum drops it;
   * embeds the text pieces between placeholders with ``embed_tokens`` and concatenates
     ``text, camera rows, text, ego row, text, ...`` in prompt order. Each placeholder id
     therefore expands to as many embedding rows as it has features (540 for
     ``<SCENE>``, 1 for ``<EGO>``);
   * expands ``labels`` the same way (``IGNORE_INDEX`` for every inserted row), and
     rebuilds ``attention_mask`` / ``position_ids`` for the new, longer sequence.

3. The Qwen2 decoder runs on the returned ``inputs_embeds`` (``input_ids=None``). In
   DriveIK training / evaluation this happens inside
   ``BEVLatentActionWorldModelMixin.forward`` (``drivevla/training/model.py``), which
   calls the routine above and then runs the decoder and the DriveIK losses / IK head.

Classes:

* ``LlavaMetaModel``: mixed into the decoder (``LlavaQwenModel``); owns ``ego_projector``
  and the ``mm_projector_*`` modules.
* ``LlavaMetaForCausalLM``: mixed into the causal LM (``LlavaQwenForCausalLM``); owns the
  ego-status curriculum and the input-preparation routine.
"""

from abc import ABC, abstractmethod

import torch

import torch.nn as nn


from .multimodal_projector.builder import build_vision_projector

# Label ignore value and the negative placeholder ids that may appear in input_ids
# (see llava/constants.py).
from llava.constants import (
    IGNORE_INDEX,
    SCENE_TOKEN_INDEX,
    EGO_TOKEN_INDEX,
)


class LlavaMetaModel:
    """Mixin for the decoder model that adds DriveIK's input projectors.

    Must precede the Hugging Face base model in the MRO (``LlavaQwenModel(LlavaMetaModel,
    Qwen2Model)``): ``__init__`` first initializes the base model through ``super()``
    and then registers:

    * ``ego_projector``: ``Linear(ego_dim, H) -> GELU -> Linear(H, H)`` mapping the
      ego-status / can_bus vector to one embedding row for ``<EGO>`` (``H`` =
      ``config.hidden_size``; ``ego_dim`` = ``config.ego_feature_dim``, default 18).
    * ``ego_curriculum_step``: non-persistent scalar ``long`` buffer counting the
      training-mode calls of ``prepare_inputs_labels_for_multimodal_uniad_vlm`` (drives
      ``LlavaMetaForCausalLM._ego_keep_prob``). Being non-persistent, it is not saved in
      checkpoints and starts at 0 in every process, including after a resume.
    * ``mm_projector_scene`` / ``_track`` / ``_map`` / ``_bev``: four independently
      parameterized projectors ``(.., mm_hidden_size) -> (.., H)``, built only for
      DINOv3-token configs (no vision tower, ``use_mm_proj=True``). Only
      ``mm_projector_bev`` (or ``mm_projector_scene`` as fallback) is used for the
      camera tokens; ``drivevla/training/data.py::ensure_mm_projectors``
      additionally aliases ``mm_projector_scene`` to ``mm_projector_bev`` through the
      instance ``__dict__`` and freezes the track / map projectors.
    """

    def __init__(self, config):
        """Initialize the base decoder, then add the ego and visual-feature projectors.

        Args:
            config: Model config. Reads ``use_ego_status`` (default True),
                ``ego_feature_dim`` (default 18), ``hidden_size``, ``mm_vision_tower`` /
                ``vision_tower``, ``use_mm_proj``, ``mm_hidden_size`` and
                ``mm_projector_type`` (set to ``"linear"`` here if missing).

        Raises:
            ValueError: if ``use_mm_proj`` is set without a vision tower but
                ``mm_hidden_size`` is missing.
        """
        # Build the actual decoder (Qwen2Model) first.
        super(LlavaMetaModel, self).__init__(config)
        # Ego-status projector: can_bus vector (ego_dim,) -> one embedding row (hidden,).
        self.use_ego_status = bool(getattr(config, "use_ego_status", True))
        if self.use_ego_status:
            ego_dim = int(getattr(config, "ego_feature_dim", 18))
            self.ego_projector = nn.Sequential(
                nn.Linear(ego_dim, config.hidden_size),
                nn.GELU(),
                nn.Linear(config.hidden_size, config.hidden_size),
            )
            # Step counter for the ego-status curriculum (not stored in checkpoints).
            self.register_buffer(
                "ego_curriculum_step", torch.zeros((), dtype=torch.long), persistent=False
            )
        # Cached-feature mode: no vision tower configured, but projection of cached
        # features requested. Build one projector per feature stream.
        _mm_tower = getattr(config, "mm_vision_tower", None)
        if (_mm_tower is None or str(_mm_tower) == "") and bool(
            getattr(config, "use_mm_proj", False)
        ):
            # Default to a single Linear projector and require the DINOv3 feature width.
            if not hasattr(config, "mm_projector_type"):
                config.mm_projector_type = "linear"
            mm_hidden = getattr(config, "mm_hidden_size", None)
            if mm_hidden is None:
                raise ValueError(
                    "use_mm_proj=True but config.mm_hidden_size is missing; cannot build mm_projector_* for the DINOv3 tokens."
                )

            def _build_proj():
                """Projector from ``config``; plain ``Linear`` if building it fails."""
                try:
                    return build_vision_projector(config)
                except Exception:
                    return nn.Linear(int(mm_hidden), int(config.hidden_size), bias=True)

            # Create each projector only if it does not exist yet. These are separate
            # modules with separate weights.
            if getattr(self, "mm_projector_scene", None) is None:
                self.mm_projector_scene = _build_proj()
            if getattr(self, "mm_projector_track", None) is None:
                self.mm_projector_track = _build_proj()
            if getattr(self, "mm_projector_map", None) is None:
                self.mm_projector_map = _build_proj()
            # The camera-token projector used by prepare_inputs_labels_for_multimodal_*;
            # left as None if construction fails.
            if getattr(self, "mm_projector_bev", None) is None:
                try:
                    self.mm_projector_bev = _build_proj()
                except Exception:
                    self.mm_projector_bev = None

    @property
    def mm_projector(self):
        """
        Backward-compatible accessor.
        Registering the alias as an nn.Module (i.e. ``self.mm_projector = ...``) makes
        safetensors saving fail, because the same tensors would then be stored under
        two names (shared tensors). Hence a property (without setter) that resolves the
        alias on every read instead.

        Returns ``mm_projector_scene`` if present, else ``mm_projector_bev``.

        Raises:
            AttributeError: if neither projector exists.
        """
        if getattr(self, "mm_projector_scene", None) is not None:
            return self.mm_projector_scene
        if getattr(self, "mm_projector_bev", None) is not None:
            return self.mm_projector_bev
        raise AttributeError(
            "No projector exists (neither mm_projector_scene nor mm_projector_bev)."
        )

    def get_vision_tower(self):
        """Return the vision-tower module, or None.

        DriveIK never builds a vision tower (features are cached), so this returns None
        for the released configs. A list-wrapped tower (upstream LLaVA convention) is
        unwrapped.
        """
        vision_tower = getattr(self, "vision_tower", None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower


class LlavaMetaForCausalLM(ABC):
    """Mixin for the causal-LM wrapper: ego curriculum and multimodal input preparation.

    Subclasses provide ``get_model()`` (returning the ``LlavaMetaModel`` decoder) and
    ``config``; ``self.training`` comes from ``nn.Module``.
    """

    @abstractmethod
    def get_model(self):
        """Return the inner decoder model (a ``LlavaMetaModel``)."""
        pass

    def get_vision_tower(self):
        """Delegate to ``get_model().get_vision_tower()`` (None in DriveIK)."""
        return self.get_model().get_vision_tower()

    def _ego_keep_prob(self) -> float:
        """
        Curriculum:
        step < warmup: 0
        warmup <= step < warmup+ramp: (step-warmup)/ramp
        step >= warmup+ramp: 1

        Probability with which the projected ego-status embedding is fed at ``<EGO>``
        during training (otherwise a zero row is fed).

        ``step`` is ``get_model().ego_curriculum_step``. ``warmup`` / ``ramp`` come from
        ``config.ego_warmup_steps`` (default 10000) and ``config.ego_ramp_steps``
        (default 70000); the released configs do not set them, so the defaults apply.
        If ``ramp <= 0`` the schedule is a hard switch from 0 to 1 at ``warmup``.

        Returns:
            float in ``[0, 1]``.
        """
        # Schedule parameters (in calls of the preparation routine, not optimizer steps).
        cfg = self.config
        warmup = int(getattr(cfg, "ego_warmup_steps", 10000))
        ramp = int(getattr(cfg, "ego_ramp_steps", 70000))
        # Lazily create the counter if the decoder was built without it
        # (e.g. use_ego_status=False).
        mm = self.get_model()
        if not hasattr(mm, "ego_curriculum_step") or mm.ego_curriculum_step is None:
            mm.register_buffer(
                "ego_curriculum_step", torch.zeros((), dtype=torch.long), persistent=False
            )
        step = int(mm.ego_curriculum_step.item())
        # Piecewise-linear schedule: 0 during warmup, linear ramp, then 1.
        if ramp <= 0:
            return 0.0 if step < warmup else 1.0
        if step < warmup:
            return 0.0
        if step < warmup + ramp:
            return float(step - warmup) / float(ramp)
        return 1.0

    def _ego_should_use(self, device: torch.device) -> bool:
        """During training, use ego with the curriculum probability; in eval/inference always use ego.

        Eval mode: True. Training mode: Bernoulli(``_ego_keep_prob()``), with the 0 / 1
        endpoints handled without sampling.

        Args:
            device: Device for the ``torch.rand`` draw (uses the torch RNG).

        Returns:
            bool: True to feed ``ego_projector(can_bus)``, False to feed a zero row.
        """
        # Evaluation / inference: always use the real ego embedding.
        if not self.training:
            return True
        # Training: follow the curriculum probability.
        p = self._ego_keep_prob()
        if p <= 0.0:
            return False
        if p >= 1.0:
            return True
        return bool((torch.rand((), device=device) < p).item())

    def _ego_step_update(self):
        """Increment the curriculum step once per forward call.

        Called at the end of every non-early-returning
        ``prepare_inputs_labels_for_multimodal_uniad_vlm`` call; a no-op in eval mode.
        Every training-mode call counts (independent of gradient accumulation or of how
        many forward passes one optimizer step makes). Creates the counter buffer if it
        is missing.
        """
        # Evaluation / inference does not advance the curriculum.
        if not self.training:
            return
        # Lazily create the counter if missing, then increment it in place.
        mm = self.get_model()
        if not hasattr(mm, "ego_curriculum_step") or mm.ego_curriculum_step is None:
            mm.register_buffer(
                "ego_curriculum_step", torch.zeros((), dtype=torch.long), persistent=False
            )
        mm.ego_curriculum_step += 1

    def prepare_inputs_labels_for_multimodal_uniad_vlm(
        self,
        input_ids,
        position_ids,
        attention_mask,
        past_key_values,
        labels,
        images,
        modalities=["image"],
        image_sizes=None,
        uniad_data=None,
        uniad_pth=None,
    ):
        """Replace placeholder ids by feature embeddings and build ``inputs_embeds``.

        For every sample, the prompt is cut at each placeholder id; the text pieces are
        embedded with ``embed_tokens`` and the placeholders are replaced as follows:

        ========================  ==================================================
        placeholder               inserted rows (all labelled ``IGNORE_INDEX``)
        ========================  ==================================================
        ``<SCENE>`` (-201)        projected camera tokens ``(N_tok, H)``; ``N_tok =
                                  540`` in the released recipes
        ``<EGO>`` (-205)          ``ego_projector(can_bus)`` ``(1, H)``, or one zero
                                  row (no ego data or curriculum drop)
        ========================  ==================================================

        Afterwards the per-sample sequences are padded to the longest sample with zero
        embeddings (right padding), and new attention masks / position ids are built.
        Finally, in training mode, the ego-curriculum counter is incremented.

        Args:
            input_ids: ``(B, L)`` long tensor containing text ids and negative
                placeholder ids. DriveIK always uses ``B = 1`` (the collator in
                ``drivevla/training/data.py`` asserts it).
            position_ids: ``(B, L)`` or None. Only its dtype / device and whether it was
                None matter; values are rebuilt as ``0..len-1`` per sample.
            attention_mask: ``(B, L)`` or None. Positions with 0 are removed before
                splicing (padding is redone afterwards).
            past_key_values: KV cache; passed through untouched.
            labels: ``(B, L)`` LM targets or None.
            images, modalities, image_sizes: upstream LLaVA arguments; unused.
            uniad_data: dict with ``"img_metas"`` holding the ego vector, e.g.
                ``{"img_metas": [{"can_bus": Tensor[ego_dim]}]}``. Passing it without
                ``uniad_pth`` raises.
            uniad_pth: feature dict with ``"bev_tokens"`` of shape
                ``(N_tok, mm_hidden_size)`` (or ``(B, N_tok, mm_hidden_size)``).

        Returns:
            tuple ``(input_ids, position_ids, attention_mask, past_key_values,
            inputs_embeds, labels)``:

            * Early return (``input_ids`` is None, neither ``uniad_pth`` nor
              ``uniad_data`` given, or ``L == 1`` as in cached decoding steps): the
              inputs unchanged with ``inputs_embeds=None``.
            * Otherwise ``input_ids=None``, ``inputs_embeds`` of shape
              ``(B, L_new, H)`` in the embedding dtype, ``labels`` ``(B, L_new)`` (None if
              ``labels`` was None), ``attention_mask`` ``(B, L_new)`` in the caller's
              dtype (None if it was None), ``position_ids`` ``(B, L_new)`` (None if it
              was None), and ``past_key_values`` unchanged.

        Raises:
            RuntimeError: ``uniad_data`` given without ``uniad_pth``.
            ValueError: no ``<SCENE>`` anywhere in the batch.
            AttributeError: neither ``mm_projector_bev`` nor ``mm_projector_scene``
                exists.
        """
        # Nothing to splice without input ids.
        if input_ids is None:
            return (input_ids, position_ids, attention_mask, past_key_values, None, labels)
        # Early exit when no multimodal inputs are given, or for single-token decoding
        # steps during generate(), where the prompt features already live in the KV
        # cache.
        if uniad_pth is None and uniad_data is None or input_ids.shape[1] == 1:
            return (input_ids, position_ids, attention_mask, past_key_values, None, labels)
        # The visual features come from the frame dict (there is no vision tower).
        if uniad_pth is None:
            raise RuntimeError(
                "uniad_data was provided but vision_tower is None. Either pass the DINOv3 tokens (uniad_pth) or set a valid mm_vision_tower in the checkpoint/config."
            )
        # Prompt validation: at least one <SCENE> must exist in the batch so the camera
        # tokens have somewhere to go. (The v3 system prompt always contains one.)
        if not torch.any(input_ids == SCENE_TOKEN_INDEX):
            raise ValueError(
                "BEV tower batch has no <SCENE>. Add <SCENE> to your prompt so BEV features can be inserted."
            )
        # Project the DINOv3 camera tokens into the LLM embedding space.
        # bev_tokens: (N_tok, C) e.g. (540, 384) fp16 from the camera encoder, or (B, N_tok, C).
        bev_tokens = uniad_pth["bev_tokens"]
        # Match the device / dtype of the token embedding table.
        dev = self.get_model().embed_tokens.weight.device
        dtype = self.get_model().embed_tokens.weight.dtype
        # Prefer the dedicated camera projector; fall back to the scene projector.
        proj = getattr(self.get_model(), "mm_projector_bev", None)
        if proj is None:
            proj = getattr(self.get_model(), "mm_projector_scene", None)
        if proj is None:
            raise AttributeError("Model missing mm_projector_bev/mm_projector_scene for BEV tower.")
        # (N_tok, C) -> (N_tok, H): one embedding row per camera patch token.
        scene_feature = proj(bev_tokens.to(device=dev, dtype=dtype))
        # Remember the caller's originals: whether they were None decides what is
        # returned, and their dtypes are reused for the rebuilt tensors.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        # Fill in defaults so the per-sample code can assume real tensors: an all-True
        # mask (a given mask is converted to bool for indexing), positions 0..L-1 (only
        # its dtype / device are used later), and fully ignored labels.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(
                0, input_ids.shape[1], dtype=torch.long, device=input_ids.device
            )
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        # Strip padding: keep only attended positions of each sample, giving lists of
        # variable-length 1-D tensors.
        input_ids = [
            cur_input_ids[cur_attention_mask]
            for (cur_input_ids, cur_attention_mask) in zip(input_ids, attention_mask)
        ]
        labels = [
            cur_labels[cur_attention_mask]
            for (cur_labels, cur_attention_mask) in zip(labels, attention_mask)
        ]
        # Per-sample outputs: (L_i, H) embeddings and (L_i,) labels.
        new_input_embeds = []
        new_labels = []

        def _pick_feat(feat, bidx: int):
            """Select sample ``bidx`` from a batched ``(B, N, H)`` feature tensor.

            Anything else (a 2-D ``(N, H)`` tensor or an out-of-range index) is
            returned unchanged, so a 2-D feature tensor is shared by every sample.
            """
            if torch.is_tensor(feat) and feat.ndim == 3:
                if 0 <= bidx < feat.shape[0]:
                    return feat[bidx]
            return feat

        # ---------------------------------------------------------------------------
        # Per-sample splicing loop. cur_input_ids: (L_i,) ids of one sample after
        # padding removal.
        # ---------------------------------------------------------------------------
        for batch_idx, cur_input_ids in enumerate(input_ids):
            # Sorted positions of all placeholders, bracketed by the sentinels -1 and L_i
            # so that consecutive pairs delimit the text segments:
            #   text[0] = ids[0 : p0], text[k] = ids[p_{k-1}+1 : p_k], last = ids[p_last+1 :].
            # There are (#placeholders + 1) text segments (possibly empty).
            special_token_indices = [-1]
            special_token_indices.extend(
                torch.where(cur_input_ids == SCENE_TOKEN_INDEX)[0].tolist()
            )
            special_token_indices.extend(torch.where(cur_input_ids == EGO_TOKEN_INDEX)[0].tolist())
            special_token_indices.sort()
            special_token_indices.append(cur_input_ids.shape[0])
            # Cut ids and labels into the text segments between placeholders.
            cur_input_ids_nospecial = []
            cur_labels = labels[batch_idx]
            cur_labels_nospecial = []
            for i in range(len(special_token_indices) - 1):
                cur_input_ids_nospecial.append(
                    cur_input_ids[special_token_indices[i] + 1 : special_token_indices[i + 1]]
                )
                cur_labels_nospecial.append(
                    cur_labels[special_token_indices[i] + 1 : special_token_indices[i + 1]]
                )
            # Embed all text segments in a single embed_tokens call (placeholders removed,
            # so every id is a valid vocabulary index), then split back per segment.
            split_sizes = [x.shape[0] for x in cur_labels_nospecial]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_nospecial))
            cur_input_embeds_no_special = torch.split(cur_input_embeds, split_sizes, dim=0)
            # Output pieces of this sample (concatenated after the loop).
            cur_new_input_embeds = []
            cur_new_labels = []
            # ---- Ego-status embedding for <EGO> (computed once per sample) ----
            # ego_embed stays None (-> zero row at <EGO>) if the model has no ego
            # projector or no can_bus vector can be found.
            ego_embed = None
            mm = self.get_model()
            if getattr(mm, "use_ego_status", False) and hasattr(mm, "ego_projector"):
                ego_vec = None
                # Find the can_bus vector inside uniad_data["img_metas"]. Any error
                # while doing so is swallowed and treated as "no ego vector".
                if uniad_data is not None:
                    try:
                        metas = uniad_data.get("img_metas", None)

                        def pick_last_meta(m):
                            """Return the meta dict holding ``can_bus``, preferring the last entry.

                            * dict with ``"can_bus"`` -> itself;
                            * dict of dicts (e.g. keyed by frame) -> its last value;
                            * non-empty list / tuple -> recurse into the last element;
                            * anything else -> returned as is.

                            The same uniad_data is used for every sample in the batch.
                            """
                            if isinstance(m, dict):
                                if "can_bus" in m:
                                    return m
                                if len(m) > 0 and isinstance(next(iter(m.values())), dict):
                                    last_key = next(reversed(m))
                                    return m[last_key]
                            if isinstance(m, (list, tuple)) and len(m) > 0:
                                return pick_last_meta(m[-1])
                            return m

                        meta_last = pick_last_meta(metas)
                        if isinstance(meta_last, dict) and "can_bus" in meta_last:
                            ego_vec = meta_last["can_bus"]
                    except Exception:
                        ego_vec = None
                if ego_vec is not None:
                    # Flatten to (D,) in the embedding dtype / device, then zero-pad or
                    # truncate to config.ego_feature_dim (the ego_projector input width;
                    # the vector's own length if the config has no such field).
                    device = mm.embed_tokens.weight.device
                    dtype = mm.embed_tokens.weight.dtype
                    ego_vec = torch.as_tensor(ego_vec, device=device, dtype=dtype).flatten()
                    ego_dim = int(getattr(self.config, "ego_feature_dim", ego_vec.numel()))
                    if ego_vec.numel() < ego_dim:
                        pad = ego_vec.new_zeros(ego_dim - ego_vec.numel())
                        ego_vec = torch.cat([ego_vec, pad], dim=0)
                    elif ego_vec.numel() > ego_dim:
                        ego_vec = ego_vec[:ego_dim]
                    # Ego curriculum (training) / always-on (eval): either the projected
                    # vector (1, ego_dim) -> (1, H) or an all-zero (1, H) row.
                    use_ego = self._ego_should_use(device=device)
                    if use_ego:
                        ego_embed = mm.ego_projector(ego_vec.unsqueeze(0)).to(dtype=dtype)
                    else:
                        ego_embed = torch.zeros(
                            (1, self.config.hidden_size), device=device, dtype=dtype
                        )
            # Interleave: text segment i, then the rows for the placeholder that ends it.
            for i in range(len(special_token_indices) - 1):
                # Emit the text segment with its original labels.
                cur_new_input_embeds.append(cur_input_embeds_no_special[i])
                cur_new_labels.append(cur_labels_nospecial[i])
                # If this segment is followed by a placeholder (i.e. it is not the final
                # segment ending at the L_i sentinel), emit that placeholder's rows.
                if special_token_indices[i + 1] < cur_input_ids.shape[0]:
                    token_type = cur_input_ids[special_token_indices[i + 1]]
                    # <SCENE>: insert the projected camera tokens (DriveIK's main path);
                    # they are never supervised.
                    if token_type == SCENE_TOKEN_INDEX:
                        sf = _pick_feat(scene_feature, batch_idx)
                        cur_new_input_embeds.append(sf)
                        cur_new_labels.append(
                            torch.full(
                                (sf.shape[0],),
                                IGNORE_INDEX,
                                device=cur_labels.device,
                                dtype=cur_labels.dtype,
                            )
                        )
                    # <EGO>: one row, either the ego embedding or zeros.
                    elif token_type == EGO_TOKEN_INDEX:
                        # Zero row if there is no ego embedding. (A curriculum drop
                        # already made ego_embed a zero row above.)
                        if ego_embed is None:
                            dummy = torch.zeros(
                                (1, self.config.hidden_size),
                                device=self.get_model().embed_tokens.weight.device,
                                dtype=self.get_model().embed_tokens.weight.dtype,
                            )
                            cur_new_input_embeds.append(dummy)
                        else:
                            cur_new_input_embeds.append(ego_embed)
                        # Exactly one ignored label in both cases.
                        cur_new_labels.append(
                            torch.full(
                                (1,), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype
                            )
                        )
            # Concatenate this sample's pieces (all moved to the embedding device):
            # (L_new_i, H) embeddings and (L_new_i,) labels.
            dev = self.get_model().embed_tokens.weight.device
            cur_new_input_embeds = [x.to(dev) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        # ---------------------------------------------------------------------------
        # Batch assembly.
        # ---------------------------------------------------------------------------
        # Allocate padded outputs: labels default to IGNORE_INDEX, the mask to False and
        # positions to 0; the loop below fills the real positions of every sample.
        max_len = max((x.shape[0] for x in new_input_embeds))
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = []
        new_labels_padded = torch.full(
            (batch_size, max_len),
            IGNORE_INDEX,
            dtype=new_labels[0].dtype,
            device=new_labels[0].device,
        )
        attention_mask = torch.zeros(
            (batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device
        )
        position_ids = torch.zeros(
            (batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device
        )
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            # Right padding: real tokens first, zero embeddings after.
            new_input_embeds_padded.append(
                torch.cat(
                    (
                        cur_new_embed,
                        torch.zeros(
                            (max_len - cur_len, cur_new_embed.shape[1]),
                            dtype=cur_new_embed.dtype,
                            device=cur_new_embed.device,
                        ),
                    ),
                    dim=0,
                )
            )
            # Fill the left-aligned span; positions 0..cur_len-1.
            if cur_len > 0:
                new_labels_padded[i, :cur_len] = cur_new_labels
                attention_mask[i, :cur_len] = True
                position_ids[i, :cur_len] = torch.arange(
                    0, cur_len, dtype=position_ids.dtype, device=position_ids.device
                )
        # (B, L_new, H) embeddings.
        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        # Mirror the caller's None-ness: return None for labels / attention_mask /
        # position_ids that were not provided; cast the mask back to the caller's dtype.
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)
        if _position_ids is None:
            position_ids = None
        # Advance the ego curriculum (training mode only).
        self._ego_step_update()
        # input_ids=None: the decoder must consume inputs_embeds.
        return (None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels)
