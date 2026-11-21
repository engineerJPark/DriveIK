"""World-model forward pass and training losses.

This module defines the DriveIK training forward pass. ``train.py`` does not
subclass the LLaVA model with :class:`BEVLatentActionWorldModelMixin`; instead
it binds the mixin's methods onto a ``LlavaQwenForCausalLM`` instance
(``base_model.forward = BEVLatentActionWorldModelMixin.forward.__get__(base_model)``
and similarly for the helpers). Inside these methods ``self`` is therefore the
LLaVA-Qwen model, with extra attributes set by ``train.py`` (token ids, loss
weights, ``ik_head``, ``ik_attn_pool``, ...).

One forward pass (batch size 1; the collator asserts it):

0. ``_encode_camera_frames``: the camera images of the current and horizon frames
   are run through the frozen DINOv3 encoder (``self.camera_encoder``, see
   :mod:`drivevla.training.vision`), giving ``bev_tokens`` (540, 384) per frame.
1. ``prepare_inputs_labels_for_multimodal_uniad_vlm`` (``llava/model/llava_arch.py``)
   replaces the ``<SCENE>`` placeholder with the current-frame DINOv3 camera
   tokens (``bev_tokens``, projected to the LLM width by
   ``mm_projector_bev``) and ``<EGO>`` with the projected ego/CAN-bus vector.
2. The Qwen2.5 backbone (``self.model``) runs on the resulting embeddings.
3. Losses:

   * ``lm_loss``: next-token cross-entropy on the assistant text, with every
     ``<BEV_STATE>`` token and everything between ``<bev_start>`` and
     ``<bev_end>`` masked out.
   * ``bev_loss`` (next visual state prediction): MSE between the last hidden
     states at the ``<BEV_STATE>`` placeholder positions and the *next* frame's
     projected DINOv3 tokens (one block of ``bev_pred_len`` = 540 tokens per
     horizon in ``bev_horizon_steps``; the recipes use a single horizon of 8
     steps = 4 s, 0.5 s per step).
   * ``ik_loss``: diffusion (eps-MSE) loss of the IK head
     (:class:`drivevla.training.heads.CrossAttentionDiffusionIK`) on the GT
     future trajectory, conditioned only on the current-frame tokens and the
     predicted next-frame hidden states.
   * ``obj_loss`` / ``box_loss`` (and ``*_next``): per-camera object presence /
     box auxiliary losses on the camera-query features of the current frame and
     of the predicted next frame.

   ``loss = bev_w * bev_loss + qa_w * lm_loss + ik_w * ik_loss (+ object losses)``.

Trajectory convention: ego frame, x = lateral, y = forward, meters; waypoints
are 0.5 s apart (8 for NAVSIM = 4 s). The IK head works on
trajectories divided by ``ik_traj_scale`` (default 20 m).

Note: text generation through ``generate()`` is not supported by this forward
(nothing in this release calls it). ``logits`` are only computed when ``labels`` contain
supervised tokens (``generate`` passes none), and cached decoding
steps (one new token, no scene features) come back from the multimodal
preparation with ``inputs_embeds=None`` while ``input_ids=None`` is passed to the
backbone, so they get no input embeddings at all.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import CausalLMOutputWithPast
from collections.abc import Mapping
from llava.constants import IGNORE_INDEX
from llava.utils import rank0_print


class FlexibleCausalLMOutputWithPast(CausalLMOutputWithPast):
    """``CausalLMOutputWithPast`` that also stores arbitrary extra keys.

    - DDP pytree unflatten must work: it calls ``output_type(**kwargs)``.
    - accelerate ``convert_to_fp32`` must work: it calls ``type(output)(mapping)``.
    => ``__init__`` accepts both forms (``*args`` and ``**kwargs``).

    The world-model forward stores many extra entries (``lm_loss``, ``bev_loss``,
    ``ik_loss``, ``wm_*`` debug values, ...) on top of the standard
    ``loss`` / ``logits`` / ``past_key_values`` / ``hidden_states`` /
    ``attentions`` fields; they are accessible both as ``out["key"]`` and as
    attributes (``ModelOutput.__setitem__`` sets both).
    """

    def __init__(self, *args, **kwargs):
        """Build the output from a mapping, positional fields and/or keywords.

        Args:
            *args: either a single mapping (or anything ``dict()`` accepts, such
                as a list of ``(key, value)`` pairs), or values of the base
                dataclass fields in declaration order (loss, logits,
                past_key_values, hidden_states, attentions).
            **kwargs: field values and extra keys. Explicit keywords win over
                entries of a positional mapping.

        Raises:
            TypeError: if more positional values are given than the base class
                has fields.
        """
        # Single positional argument: treat it as a mapping (accelerate path);
        # if it cannot be converted with dict(), fall through and treat it as
        # the first positional field instead.
        if len(args) == 1:
            src = args[0]
            if isinstance(src, Mapping):
                kwargs = {**dict(src), **kwargs}
                args = ()
            else:
                try:
                    kwargs = {**dict(src), **kwargs}
                    args = ()
                except Exception:
                    pass
        # Map remaining positional values onto the base dataclass field names
        # (without overriding values already given by keyword).
        base_fields = getattr(CausalLMOutputWithPast, "__dataclass_fields__", {})
        if args:
            field_names = list(base_fields.keys())
            for i, v in enumerate(args):
                if i >= len(field_names):
                    raise TypeError(
                        f"Too many positional args for {self.__class__.__name__}: {len(args)}"
                    )
                kwargs.setdefault(field_names[i], v)
            args = ()
        # Known fields go to the dataclass constructor; everything else is
        # added afterwards as an extra ModelOutput key.
        base_kwargs = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in base_fields}
        super().__init__(**base_kwargs)
        for k, v in kwargs.items():
            self[k] = v


class BEVLatentActionWorldModelMixin:
    """Adds next-BEV MSE loss and (optional) IK loss.

    The methods are bound onto a ``LlavaQwenForCausalLM`` instance by
    ``train.py`` (see the module docstring), so ``self`` provides
    ``get_model()``, ``get_vision_tower()``, ``prepare_inputs_labels_for_multimodal_uniad_vlm``,
    ``self.model`` (Qwen2 backbone), ``self.lm_head`` and ``self.config``.

    Attributes expected on ``self`` (set by ``train.py`` / the trainer):
        bev_state_token_id, bev_start_token_id, bev_end_token_id: tokenizer ids
            of ``<BEV_STATE>``, ``<bev_start>`` and ``<bev_end>`` (each must be
            a single token).
        bev_loss_weight, qa_loss_weight, ik_loss_weight: loss weights.
            ``train.py`` initialises ``ik_loss_weight`` to 0 (curriculum stage
            1); the curriculum callback (``TwoStageCurriculumCallback`` in
            ``trainer.py``) then writes the ``--ik-loss-weight`` value at train
            begin. Under checkpoint evaluation no callback runs, so it stays at
            0 there.
        ik_stop_grad: detach the IK conditioning from the LLM. Initialised from
            ``--ik-stop-grad`` (set in all recipes), but the curriculum callback
            sets it to False when it switches to stage 2 at train begin, so SFT
            runs with False and the IK loss also trains the LLM through the
            predicted next-state tokens. The GRPO trainer forces it to True
            before every forward.
        ik_head: :class:`~drivevla.training.heads.CrossAttentionDiffusionIK`.
        ik_attn_pool / object_query_head:
            :class:`~drivevla.training.heads.CameraConditionedQueryPool` (4
            camera queries) and :class:`~drivevla.training.heads.ObjectQueryHead`
            for the object losses.
        Also read: num_bev_horizons, bev_pred_len, ik_traj_scale, ik_diff_mc,
            ik_weight_longitudinal, ik_weight_lateral, ik_debug_sample_steps,
            obj/box loss weights, curriculum_stage.
    """

    bev_state_token_id: int
    bev_start_token_id: int
    bev_end_token_id: int
    bev_loss_weight: float
    qa_loss_weight: float
    ik_loss_weight: float
    ik_stop_grad: bool

    @staticmethod
    def _mask_spans_between_tokens_inplace(
        labels: torch.Tensor, start_id: int, end_id: int, ignore_index: int = IGNORE_INDEX
    ) -> None:
        """Set labels strictly between each start token and the next end token to ``ignore_index``.

        Used to remove the ``<bev_start> <BEV_STATE>*K <bev_end>`` blocks from the
        language-model loss: the start and end tokens themselves stay
        supervised, everything in between is masked.

        Args:
            labels: (B, L) int64 label tensor, modified in place. ``None`` is a no-op.
            start_id: token id that opens a span.
            end_id: token id that closes a span.
            ignore_index: value written into masked positions (-100).

        Raises:
            ValueError: if ``labels`` is not 2-D.

        Note:
            Spans are found in ``labels`` (not in ``input_ids``), so only
            supervised start/end tokens are considered. If a start has no later
            end token, positions ``s+1 .. L-2`` are masked (the last position is
            left untouched).
        """
        # Nothing to do without labels; enforce the (B, L) layout.
        if labels is None:
            return
        if labels.ndim != 2:
            raise ValueError(f"labels must be (B,L), got shape={tuple(labels.shape)}")
        start_id = int(start_id)
        end_id = int(end_id)
        B, L = labels.shape
        for b in range(B):
            # All start positions in this row.
            starts = (labels[b] == start_id).nonzero(as_tuple=False).flatten()
            if starts.numel() == 0:
                continue
            for s in starts.tolist():
                # First end token at or after the start; if none, run to L - 1.
                rel = (labels[b, s:] == end_id).nonzero(as_tuple=False)
                if rel.numel() == 0:
                    e = L - 1
                else:
                    e = s + int(rel[0].item())
                # Mask the open interval (s, e).
                labels[b, s + 1 : e] = ignore_index

    def _encode_camera_frames(self, uniad_pth, next_uniad_pths):
        """Replace camera-image frames by their DINOv3 tokens.

        Every frame dict holds ``"camera_images"`` (``(n_cams, 3, H, W)`` from the
        dataset) and is encoded by ``self.camera_encoder``; all frames of the call
        (current + horizons) go through the encoder in one batch. ``None`` entries
        are returned as they are. The input dicts are not modified.

        Args:
            uniad_pth: current-frame dict (or None).
            next_uniad_pths: list of horizon-frame dicts (or None).

        Returns:
            ``(uniad_pth, next_uniad_pths)`` with ``{"bev_tokens": (N, C)}`` (fp16 on
            GPU) in place of each camera-image frame.

        Raises:
            ValueError: a frame is not a dict with ``"camera_images"``.
            RuntimeError: the model has no ``camera_encoder`` (train.py attaches it).
        """
        frames = [uniad_pth] + list(next_uniad_pths or [])
        idx = [i for i, f in enumerate(frames) if f is not None]
        if not idx:
            return uniad_pth, next_uniad_pths
        for i in idx:
            if not (isinstance(frames[i], Mapping) and "camera_images" in frames[i]):
                raise ValueError(f"frame {i} has no camera_images (got {type(frames[i]).__name__})")
        encoder = getattr(self, "camera_encoder", None)
        if encoder is None:
            raise RuntimeError("Batch holds camera images but the model has no camera_encoder.")
        # Frames may arrive on different devices (the evaluation loop moves only the
        # current frame dict), so move each one before stacking.
        device = self.get_model().embed_tokens.weight.device
        images = torch.stack(
            [torch.as_tensor(frames[i]["camera_images"]).to(device) for i in idx]
        )
        tokens = encoder(images, device)
        for j, i in enumerate(idx):
            rest = {k: v for k, v in frames[i].items() if k != "camera_images"}
            frames[i] = {**rest, "bev_tokens": tokens[j]}
        return frames[0], (frames[1:] if next_uniad_pths is not None else None)

    def _project_scene_feature_from_vt_result(self, vt_result: Dict[str, Any]) -> torch.Tensor:
        """Project DINOv3 camera tokens into the LLM embedding space.

        This is the same projection that ``prepare_inputs_labels_for_multimodal_uniad_vlm``
        uses for the ``<SCENE>`` input, so the result is directly comparable to
        LLM hidden states. It is used for the next-frame BEV-loss targets and
        for the IK head's current-frame tokens.

        Args:
            vt_result: frame dict from :meth:`_encode_camera_frames` containing
                ``"bev_tokens"`` of shape (N, C) (N = 540 front-camera patch
                tokens, C = DINOv3 feature width); a leading batch dim is passed
                through unchanged.

        Returns:
            (N, H) tensor (H = LLM hidden size) on the embedding device/dtype.
            Gradients flow through the projector unless the caller uses
            ``no_grad`` (``forward`` does, for both of its uses).

        Raises:
            AttributeError: if the model has no BEV/scene projector.
            ValueError: if ``"bev_tokens"`` is missing.
        """
        # Embedding device/dtype define where the projected tokens live.
        mm = self.get_model()
        dev = mm.embed_tokens.weight.device
        dtype = mm.embed_tokens.weight.dtype
        if "bev_tokens" in vt_result:
            # Load the raw DINOv3 tokens onto the model device/dtype.
            bev_tokens = vt_result["bev_tokens"]
            if not torch.is_tensor(bev_tokens):
                bev_tokens = torch.as_tensor(bev_tokens)
            bev_tokens = bev_tokens.to(device=dev, dtype=dtype)
            # Prefer mm_projector_bev; mm_projector_scene is an alias created by
            # ensure_mm_projectors in data.py.
            projector = getattr(mm, "mm_projector_bev", None) or getattr(
                mm, "mm_projector_scene", None
            )
            if projector is None:
                raise AttributeError("Model has no mm_projector_bev/mm_projector_scene")
            return projector(bev_tokens)
        raise ValueError("Expected a frame dict containing bev_tokens")

    @staticmethod
    def _align_len(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Resample a token sequence to ``target_len`` tokens.

        Used to match a BEV target sequence to the number of predicted
        ``<BEV_STATE>`` tokens (both are 540 in the released recipes, so this is
        normally the identity).

        Args:
            x: (n, h) sequence.
            target_len: desired length m.

        Returns:
            (m, h) tensor: ``x`` itself if n == m; truncation / zero padding at
            the end if n <= 1 or m <= 1; otherwise linear interpolation along
            the token axis (``align_corners=False``).
        """
        n, h = x.shape
        m = int(target_len)
        # Already aligned.
        if n == m:
            return x
        # Degenerate lengths: interpolation is not meaningful -> cut or pad zeros.
        if n <= 1 or m <= 1:
            if n >= m:
                return x[:m]
            pad = torch.zeros((m - n, h), device=x.device, dtype=x.dtype)
            return torch.cat([x, pad], dim=0)
        # (n, h) -> (1, h, n) so F.interpolate resamples the token axis, then back to (m, h).
        xt = x.t().unsqueeze(0)
        yt = F.interpolate(xt, size=m, mode="linear", align_corners=False)
        return yt.squeeze(0).t()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        images=None,
        image_sizes=None,
        return_dict=True,
        modalities=["image"],
        uniad_pth=None,
        uniad_data=None,
        next_uniad_pths=None,
        gt_traj=None,
        query_box_targets=None,
        query_has_obj=None,
        query_box_targets_next=None,
        query_has_obj_next=None,
        **kwargs,
    ):
        """World-model forward: LLM pass + next-visual-state, LM and IK losses.

        Bound as ``base_model.forward`` by ``train.py``. Inputs come from
        ``DataCollatorForLatentActionWM`` (batch size 1). The system prompt of
        the v3 ChatML template holds the ``<SCENE>`` / ``<EGO>`` placeholders,
        the user turn holds the driving context (ego status, command, previous
        action), and the assistant turn is one
        ``<bev_start> <BEV_STATE>*540 <bev_end>`` block per horizon.

        Args:
            input_ids: (1, L) token ids with negative placeholder ids
                (``SCENE_TOKEN_INDEX`` = -201, ``EGO_TOKEN_INDEX`` = -205).
            attention_mask: (1, L) bool/int mask.
            position_ids, past_key_values: forwarded to the backbone.
            inputs_embeds: if given, multimodal preparation is skipped and these
                embeddings are used as-is (this is how HF ``generate`` passes
                the prepared prompt on its first step).
            labels: (1, L) labels, IGNORE_INDEX (-100) on the prompt. All
                losses are computed only when labels are given.
            images, image_sizes, modalities: LLaVA image arguments, forwarded to
                the multimodal preparation (not used by DriveIK; the camera
                images arrive in ``uniad_pth``).
            uniad_pth: current-frame visual input ``{"camera_images": (3, 3, 288,
                160)}`` (encoded here by ``self.camera_encoder``); the dataset
                always provides it.
            uniad_data: ``{"img_metas": [{"can_bus": tensor}]}``; supplies the
                ego/CAN-bus vector for ``<EGO>``.
            next_uniad_pths: list (one per horizon) of next-frame visual inputs in
                the same form (or None) -- the BEV-loss targets.
            gt_traj: (T, 2) GT future ego trajectory in meters (T = 6 or 8
                waypoints at 0.5 s; x lateral, y forward), or None.
            query_box_targets / query_has_obj: (1, 4, 6) / (1, 4) current-frame
                per-query object targets.
            query_box_targets_next / query_has_obj_next: (1, K, 4, 6) /
                (1, K, 4) targets per future horizon.
            return_dict: if False, return ``(loss, logits)``.
            **kwargs: extra collator fields (ignored) and the debug switches
                ``return_mm_labels_for_debug``, ``return_ik_pred_for_debug``
                (sample an IK trajectory; used by evaluation and GRPO) and
                ``return_attn_debug`` (last-layer attention shares).

        Returns:
            :class:`FlexibleCausalLMOutputWithPast` with ``loss``, ``logits``
            (None unless the LM loss path ran), ``past_key_values``,
            ``attentions``, the individual losses (``lm_loss``, ``bev_loss``,
            ``ik_loss`` and the object losses) and ``wm_*`` diagnostics (e.g.
            ``wm_ik_pred_traj`` (1, T, 2) in meters, ``wm_attn_*_ratio`` in
            percent).

        Side effects:
            Writes the IK conditioning tokens to ``self.ik_head._grpo_*`` (read
            by the GRPO trainer) when the IK branch runs.

        Note: text generation through ``generate()`` is not supported here --
        see the module docstring (logits only computed with
        labels, cached decoding steps get no input embeddings).
        """
        # ---- Debug switches (popped so they are not forwarded anywhere) ----
        return_mm_labels_for_debug = bool(kwargs.pop("return_mm_labels_for_debug", False))
        return_ik_pred_for_debug = bool(kwargs.pop("return_ik_pred_for_debug", False))
        return_attn_debug = bool(kwargs.pop("return_attn_debug", False))
        # ---- Visual input: camera images -> frozen DINOv3 tokens ----
        # After this, every frame holds "bev_tokens" (540, C).
        uniad_pth, next_uniad_pths = self._encode_camera_frames(uniad_pth, next_uniad_pths)
        # ---- Initialise every debug / diagnostic value to None ----
        mm_input_ids_dbg = None
        mm_labels_dbg = None
        mm_attn_dbg = None
        # Length of the text sequence *before* placeholder expansion (used by
        # the attention debug to infer how many tokens each <SCENE> expands to).
        dbg_text_L = (
            int(input_ids.shape[1]) if torch.is_tensor(input_ids) and input_ids.ndim == 2 else None
        )
        # Sequence-length / cross-entropy target-count diagnostics.
        dbg_mm_L = None
        dbg_ce_raw = None
        dbg_bev_span_mask_applied = False
        # BEV prediction statistics (first horizon of sample 0).
        dbg_bev_k = None
        dbg_tgt_len = None
        dbg_pred_norm = None
        dbg_tgt_norm = None
        dbg_cos = None
        # Sampled IK trajectory / GT in meters (filled with return_ik_pred_for_debug).
        dbg_ik_pred_traj = None
        dbg_ik_gt_traj = None
        # Attention-share diagnostics (filled with return_attn_debug).
        dbg_attn_bev_weight = None
        dbg_attn_ego_weight = None
        dbg_attn_text_weight = None
        dbg_attn_bev_ratio = None
        dbg_attn_ego_ratio = None
        dbg_attn_text_ratio = None
        dbg_attn_bev_per_tok = None
        dbg_attn_ego_per_tok = None
        dbg_attn_text_per_tok = None
        # Number of <SCENE> placeholders in the unexpanded prompt (for the
        # attention debug).
        _orig_num_scene_tokens = 0
        # ---- Multimodal input preparation ----
        # Without precomputed embeddings, expand the placeholders: <SCENE> ->
        # the 540 projected DINOv3 tokens of `uniad_pth`, <EGO> -> the projected
        # CAN-bus vector. The returned labels are expanded to the same length
        # (IGNORE_INDEX at inserted positions); the returned input ids are None
        # after expansion. For a one-token input (cached decoding step) or
        # missing scene inputs, the preparation returns early with
        # inputs_embeds=None, and since input_ids=None is passed to the backbone
        # below such a step has no input embeddings (known generate() issue).
        if inputs_embeds is None:
            # Count the <SCENE> placeholders before they are expanded.
            if torch.is_tensor(input_ids) and input_ids.ndim == 2:
                from llava.constants import SCENE_TOKEN_INDEX

                _orig_num_scene_tokens = int((input_ids == SCENE_TOKEN_INDEX).sum().item())
            # Expand placeholders into embeddings; position_ids / attention_mask /
            # labels are replaced by their expanded versions.
            mm_input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = (
                self.prepare_inputs_labels_for_multimodal_uniad_vlm(
                    input_ids,
                    position_ids,
                    attention_mask,
                    past_key_values,
                    labels,
                    images,
                    modalities,
                    image_sizes,
                    uniad_data=uniad_data,
                    uniad_pth=uniad_pth,
                )
            )
            # Optional detached copies of the expanded tensors for debugging.
            if return_mm_labels_for_debug:
                if torch.is_tensor(mm_input_ids):
                    mm_input_ids_dbg = mm_input_ids.detach()
                if torch.is_tensor(labels):
                    mm_labels_dbg = labels.detach()
                if torch.is_tensor(attention_mask):
                    mm_attn_dbg = attention_mask.detach()
        # ---- Qwen2 backbone pass on the (expanded) input embeddings ----
        # input_ids is always None here; the backbone runs from inputs_embeds.
        _fwd_kwargs = dict(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            return_dict=True,
        )
        # The attention debug needs the per-layer hidden states.
        if return_attn_debug:
            _fwd_kwargs["output_hidden_states"] = True
        outputs = self.model(**_fwd_kwargs)
        # (1, L_mm, H) final-norm hidden states; read at <BEV_STATE> positions
        # for the next-visual-state prediction.
        hidden = outputs.last_hidden_state
        # ---- LM head: only when an LM loss can be computed ----
        # Logits are computed iff the QA/LM weight is > 0 and the labels
        # contain at least one supervised token. Training samples always do
        # (the <bev_start>/<bev_end> tokens are supervised). Calls without
        # labels (generate()) get logits=None -- part of the known generate()
        # issue. `getattr(..., default) or 0.0` maps a None/0 weight to 0.0.
        qa_w = float(getattr(self, "qa_loss_weight", 1.0) or 0.0)
        _has_labels = (
            labels is not None
            and torch.is_tensor(labels)
            and (int(labels.ne(IGNORE_INDEX).sum()) > 0)
        )
        if qa_w > 0 and _has_labels:
            logits = self.lm_head(hidden)
        else:
            logits = None
        # ---- One-time (per model instance) attention-debug entry log on rank 0 ----
        if hasattr(self, "_attn_dbg_entry_done"):
            pass
        elif return_attn_debug:
            self._attn_dbg_entry_done = True
            try:
                import torch.distributed as dist

                is_rank0 = not dist.is_initialized() or dist.get_rank() == 0
                if is_rank0:
                    lbl_ok = torch.is_tensor(labels) and labels.ndim == 2
                    hs_ok = hasattr(outputs, "hidden_states") and outputs.hidden_states is not None
                    rank0_print(
                        f"[DEBUG ATTN ENTRY] return_attn_debug={return_attn_debug}, labels_ok={lbl_ok}, hs_ok={hs_ok}"
                    )
            except Exception:
                pass
        # ---- Attention-share diagnostic (eval_wm/attn_* in the README) ----
        # FlashAttention does not return attention weights, so the last decoder
        # layer's attention is recomputed by hand (no grad, fp32): RMSNorm ->
        # q/k projections -> RoPE -> GQA key expansion -> causal softmax,
        # averaged over heads. It then measures how much attention the
        # assistant (supervised) positions pay to the visual (<SCENE>-expanded),
        # ego (<EGO>) and remaining text input positions. Batch element 0 only;
        # any failure is logged and the diagnostic stays None.
        if return_attn_debug and torch.is_tensor(labels) and (labels.ndim == 2):
            try:
                with torch.no_grad():
                    hs_all = outputs.hidden_states
                    if hs_all is not None and len(hs_all) >= 2:
                        import math as _math

                        # hidden_states = (embeddings, layer_1, ..., final-normed
                        # last layer); [-2] is the input of the last layer.
                        hs_input_last_raw = hs_all[-2].detach()
                        B_a, L_a, H_a = hs_input_last_raw.shape
                        last_layer = self.model.layers[-1]
                        last_attn = last_layer.self_attn
                        # Head geometry (GQA: num_heads query heads share
                        # num_kv_heads key/value heads).
                        cfg = self.model.config
                        num_heads = cfg.num_attention_heads
                        num_kv_heads = getattr(cfg, "num_key_value_heads", num_heads)
                        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // num_heads)
                        num_kv_groups = num_heads // num_kv_heads
                        # Pre-attention norm + q/k projections in fp32 (Qwen2 has q/k biases).
                        hs_normed = last_layer.input_layernorm(hs_input_last_raw).float()
                        q_proj_w = last_attn.q_proj.weight.detach().float()
                        k_proj_w = last_attn.k_proj.weight.detach().float()
                        Q = torch.nn.functional.linear(hs_normed, q_proj_w)
                        K = torch.nn.functional.linear(hs_normed, k_proj_w)
                        if last_attn.q_proj.bias is not None:
                            Q = Q + last_attn.q_proj.bias.detach().float()
                        if last_attn.k_proj.bias is not None:
                            K = K + last_attn.k_proj.bias.detach().float()
                        # (B, L, heads*D) -> (B, heads, L, D).
                        Q = Q.view(B_a, L_a, num_heads, head_dim).transpose(1, 2)
                        K = K.view(B_a, L_a, num_kv_heads, head_dim).transpose(1, 2)
                        # RoPE with positions 0..L-1 (the passed position_ids
                        # are not reused here).
                        _pos_ids = (
                            torch.arange(L_a, device=hs_input_last_raw.device)
                            .unsqueeze(0)
                            .expand(B_a, -1)
                        )
                        # Newer transformers: model-level rotary_emb(x, pos) -> (cos, sin)
                        # of shape (B, L, D); unsqueeze to broadcast over heads.
                        if hasattr(self.model, "rotary_emb"):
                            _cos, _sin = self.model.rotary_emb(hs_input_last_raw, _pos_ids)
                            _cos = _cos.float().unsqueeze(1)
                            _sin = _sin.float().unsqueeze(1)
                        # Older transformers: per-layer rotary_emb(x, seq_len)
                        # returns full tables that are indexed by position.
                        else:
                            _rotary = last_attn.rotary_emb
                            _cos_full, _sin_full = _rotary(hs_input_last_raw, seq_len=L_a)
                            _cos = _cos_full[_pos_ids].float().unsqueeze(1)
                            _sin = _sin_full[_pos_ids].float().unsqueeze(1)

                        def _rotate_half(_x):
                            # Standard RoPE helper: [x1, x2] -> [-x2, x1].
                            _x1 = _x[..., : _x.shape[-1] // 2]
                            _x2 = _x[..., _x.shape[-1] // 2 :]
                            return torch.cat((-_x2, _x1), dim=-1)

                        # Apply rotary position embedding to queries and keys.
                        Q = Q * _cos + _rotate_half(Q) * _sin
                        K = K * _cos + _rotate_half(K) * _sin
                        # GQA: repeat each KV head for its group of query heads.
                        if num_kv_groups > 1:
                            K = K[:, :, None, :, :].expand(
                                B_a, num_kv_heads, num_kv_groups, L_a, head_dim
                            )
                            K = K.reshape(B_a, num_heads, L_a, head_dim)
                        # Scaled dot-product scores with a causal mask (no
                        # padding mask; batch size is 1), softmax over keys,
                        # then average over heads -> (B, L, L).
                        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / _math.sqrt(head_dim)
                        causal = torch.triu(
                            torch.ones(L_a, L_a, device=attn_scores.device, dtype=torch.bool),
                            diagonal=1,
                        )
                        attn_scores.masked_fill_(causal.unsqueeze(0).unsqueeze(0), float("-inf"))
                        attn_weights = torch.softmax(attn_scores, dim=-1)
                        attn_avg = attn_weights.mean(dim=1)
                        # Infer how many embeddings each <SCENE> placeholder
                        # expanded to: expanded_L = text_L - n_scene + n_scene * k
                        # (each <EGO> stays one position), so
                        # k = (expanded_L - text_L + n_scene) / n_scene.
                        _orig_text_L = dbg_text_L or 0
                        _expanded_L = L_a
                        _n_scene = (
                            max(1, _orig_num_scene_tokens) if _orig_num_scene_tokens > 0 else 0
                        )
                        if _n_scene > 0 and _orig_text_L > 0:
                            _scene_feat_len = (_expanded_L - _orig_text_L + _n_scene) // _n_scene
                        else:
                            _scene_feat_len = 0
                        # Query rows = supervised (assistant) positions of sample 0;
                        # candidate key columns = all unsupervised (input) positions.
                        b = 0
                        lbl = labels[b]
                        asst_mask = lbl.ne(IGNORE_INDEX)
                        n_asst = int(asst_mask.sum().item())
                        if n_asst > 0 and _expanded_L > 0:
                            input_mask = ~asst_mask
                            bev_mask = torch.zeros(L_a, dtype=torch.bool, device=attn_avg.device)
                            ego_mask = torch.zeros(L_a, dtype=torch.bool, device=attn_avg.device)
                            # Walk the *unexpanded* input ids, advancing an
                            # expanded-sequence cursor, to mark which expanded
                            # positions came from <SCENE> (visual) and <EGO>.
                            if (
                                input_ids is not None
                                and torch.is_tensor(input_ids)
                                and (_expanded_L > 0)
                            ):
                                from llava.constants import SCENE_TOKEN_INDEX, EGO_TOKEN_INDEX

                                cur_pos = 0
                                scene_feats = []
                                ego_feats = []
                                for tid in input_ids[b].tolist():
                                    if cur_pos >= _expanded_L:
                                        break
                                    # <SCENE>: covers the next _scene_feat_len positions.
                                    if tid == SCENE_TOKEN_INDEX:
                                        scene_feats.extend(
                                            range(
                                                cur_pos, min(cur_pos + _scene_feat_len, _expanded_L)
                                            )
                                        )
                                        cur_pos += _scene_feat_len
                                    # <EGO>: exactly one expanded position.
                                    elif tid == EGO_TOKEN_INDEX:
                                        ego_feats.extend(
                                            range(cur_pos, min(cur_pos + 1, _expanded_L))
                                        )
                                        cur_pos += 1
                                    # Ordinary text token: one position.
                                    else:
                                        cur_pos += 1
                                # Turn the collected index lists into boolean masks.
                                if scene_feats:
                                    bev_mask[
                                        torch.tensor(
                                            scene_feats, dtype=torch.long, device=attn_avg.device
                                        )
                                    ] = True
                                if ego_feats:
                                    ego_mask[
                                        torch.tensor(
                                            ego_feats, dtype=torch.long, device=attn_avg.device
                                        )
                                    ] = True
                            # Text = every other input (unsupervised) position.
                            text_mask = input_mask & ~bev_mask & ~ego_mask
                            # Attention rows of the assistant positions: (n_asst, L).
                            attn_b = attn_avg[b]
                            asst_positions = asst_mask.nonzero(as_tuple=False).squeeze(1)
                            attn_from_asst = attn_b[asst_positions]
                            # Total attention mass (summed over assistant rows)
                            # going to each input category.
                            w_bev = (
                                attn_from_asst[:, bev_mask].sum().item() if bev_mask.any() else 0.0
                            )
                            w_ego = (
                                attn_from_asst[:, ego_mask].sum().item() if ego_mask.any() else 0.0
                            )
                            w_text = (
                                attn_from_asst[:, text_mask].sum().item()
                                if text_mask.any()
                                else 0.0
                            )
                            w_total = w_bev + w_ego + w_text
                            # Report: *_weight = mean mass per assistant row,
                            # *_ratio = percentage share among the three input
                            # categories (attention to other assistant tokens is
                            # excluded), *_per_tok = mass per row per key token.
                            if w_total > 0:
                                dbg_attn_bev_weight = w_bev / n_asst
                                dbg_attn_ego_weight = w_ego / n_asst
                                dbg_attn_text_weight = w_text / n_asst
                                dbg_attn_bev_ratio = w_bev / w_total * 100.0
                                dbg_attn_ego_ratio = w_ego / w_total * 100.0
                                dbg_attn_text_ratio = w_text / w_total * 100.0
                                _n_bev_tok = int(bev_mask.sum().item())
                                _n_ego_tok = int(ego_mask.sum().item())
                                _n_text_tok = int(text_mask.sum().item())
                                dbg_attn_bev_per_tok = (
                                    w_bev / (_n_bev_tok * n_asst) if _n_bev_tok > 0 else 0.0
                                )
                                dbg_attn_ego_per_tok = (
                                    w_ego / (_n_ego_tok * n_asst) if _n_ego_tok > 0 else 0.0
                                )
                                dbg_attn_text_per_tok = (
                                    w_text / (_n_text_tok * n_asst) if _n_text_tok > 0 else 0.0
                                )
            # The diagnostic must never break training/eval: log and continue.
            except Exception as _attn_e:
                import traceback as _tb

                rank0_print(
                    f"[WARN][ATTN_FWD] Attention debug failed: {type(_attn_e).__name__}: {_attn_e}"
                )
                rank0_print(
                    f"[WARN][ATTN_FWD] Traceback:\n{''.join(_tb.format_exception(type(_attn_e), _attn_e, _attn_e.__traceback__))}"
                )
        # A scalar 0 that is attached to the autograd graph (via `hidden`), so
        # "empty" losses built from it still have a grad_fn and backward()
        # reaches the LLM parameters (keeps DDP gradient sync consistent).
        # Fallback: a fresh leaf zero with requires_grad=True.
        try:
            _graph_zero = hidden.reshape(-1)[0] * 0.0
        except Exception:
            _graph_zero = torch.zeros(
                (), device=hidden.device, dtype=hidden.dtype, requires_grad=True
            )
        # ---- Loss placeholders (None unless computed below) ----
        loss = None
        lm_loss = None
        bev_loss = None
        ik_loss = None
        # Camera-query object losses (current frame / next frame).
        obj_loss_val = None
        box_loss_val = None
        obj_loss_next_val = None
        box_loss_next_val = None
        # Camera-query features (1, 4, H) of the current frame and of each
        # predicted next frame, filled by the IK branch for the object losses.
        q_cur = None
        q_next_list: List[torch.Tensor] = []
        # ---- Loss weights and IK options ----
        bev_w = float(getattr(self, "bev_loss_weight", 1.0) or 0.0)
        ik_w = float(getattr(self, "ik_loss_weight", 0.0) or 0.0)
        ik_stop_grad = bool(getattr(self, "ik_stop_grad", False))
        # Longitudinal / lateral weights of the IK diffusion loss
        # (--ik-longitudinal-weight 10, --ik-lateral-weight 5 by default).
        ik_w_lon = float(getattr(self, "ik_weight_longitudinal", 1.0) or 1.0)
        ik_w_lat = float(getattr(self, "ik_weight_lateral", 1.0) or 1.0)
        # All losses below require labels. Without labels (e.g. generate()) this
        # whole block is skipped.
        if labels is not None:
            # Diagnostics: expanded sequence length and the number of supervised
            # next-token targets before any BEV masking.
            try:
                dbg_mm_L = (
                    int(labels.shape[1]) if torch.is_tensor(labels) and labels.ndim == 2 else None
                )
                dbg_ce_raw = (
                    int(labels[..., 1:].ne(IGNORE_INDEX).sum().item())
                    if torch.is_tensor(labels)
                    else None
                )
            except Exception:
                dbg_mm_L = None
                dbg_ce_raw = None
            # ---- Language-model loss ----
            # "QA" labels: copy of the labels with the inside of every
            # <bev_start> ... <bev_end> span masked (the span markers stay).
            lm_labels_qa = labels.clone()
            try:
                BEVLatentActionWorldModelMixin._mask_spans_between_tokens_inplace(
                    lm_labels_qa,
                    start_id=int(self.bev_start_token_id),
                    end_id=int(self.bev_end_token_id),
                    ignore_index=IGNORE_INDEX,
                )
                dbg_bev_span_mask_applied = True
            except Exception:
                pass
            # Also mask any <BEV_STATE> token outside such spans: these positions
            # are trained by the BEV (MSE) loss, never by cross-entropy.
            lm_labels_qa[lm_labels_qa == int(self.bev_state_token_id)] = IGNORE_INDEX
            if logits is None:
                # No logits (LM weight 0 or no supervised tokens): graph-connected zero.
                lm_loss = _graph_zero
            else:
                # Causal shift: logits at position i predict the label at i + 1.
                # Mean token-level cross-entropy over the supervised QA targets.
                shift_labels = lm_labels_qa[..., 1:].contiguous()
                n_sup = int(shift_labels.ne(IGNORE_INDEX).sum().item())
                if n_sup > 0:
                    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)
                    shift_logits = logits[..., :-1, :].contiguous()
                    lm_loss = loss_fct(
                        shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                    )
                else:
                    lm_loss = _graph_zero
            # ---- Next visual state (BEV) prediction loss ----
            # Next-frame feature dicts ("bev_tokens"), one per horizon (None where
            # a frame has no visual input).
            next_list = list(next_uniad_pths or [])
            # Number of predicted horizons (len(bev_horizon_steps); 1 in the
            # recipes).
            num_h = int(self.num_bev_horizons)
            # Per-sample accumulators: BEV losses and the camera-query features
            # of the predicted next-frame tokens.
            per_sample_bev: List[torch.Tensor] = []
            next_qfeat_pred_all: List[List[torch.Tensor]] = []
            # Predicted next-state tokens are read where the (expanded) labels
            # hold <BEV_STATE>: the hidden state *at* each placeholder position
            # (no causal shift) is the prediction for that token. mask: (B, L_mm).
            tok_id = int(self.bev_state_token_id)
            mask = labels.eq(tok_id)
            # Targets: next-frame DINOv3 tokens projected into the LLM space with
            # the same mm_projector_bev as the input (no grad through targets).
            with torch.no_grad():
                target_scenes: List[Optional[torch.Tensor]] = [
                    None if p is None else self._project_scene_feature_from_vt_result(p)
                    for p in next_list
                ]
            # Pad with None / truncate so there is exactly one target slot per horizon.
            if len(target_scenes) < num_h:
                target_scenes = target_scenes + [None] * (num_h - len(target_scenes))
            target_scenes = target_scenes[:num_h]
            # Expected number of <BEV_STATE> tokens per horizon (--bev-pred-len 540).
            exp_k = int(self.bev_pred_len)
            for b in range(hidden.size(0)):
                # (K_total, H) hidden states at this sample's <BEV_STATE> positions.
                pred_all = hidden[b][mask[b]]
                # Split them into one chunk per horizon:
                #   no tokens            -> empty (0, H) chunks,
                #   exactly exp_k * num_h -> blocks of exp_k tokens,
                #   otherwise            -> equal splits, remainder in the last chunk
                #                            (e.g. after truncation).
                pred_chunks: List[torch.Tensor] = []
                if pred_all.numel() == 0:
                    pred_chunks = [
                        torch.zeros((0, hidden.size(-1)), device=hidden.device, dtype=hidden.dtype)
                        for _ in range(num_h)
                    ]
                elif pred_all.shape[0] == exp_k * num_h:
                    for hi in range(num_h):
                        pred_chunks.append(pred_all[hi * exp_k : (hi + 1) * exp_k])
                else:
                    step = max(1, pred_all.shape[0] // num_h)
                    for hi in range(num_h):
                        s = hi * step
                        e = (
                            pred_all.shape[0]
                            if hi == num_h - 1
                            else min(pred_all.shape[0], (hi + 1) * step)
                        )
                        pred_chunks.append(pred_all[s:e])
                bev_losses_h: List[torch.Tensor] = []
                qfeat_pred_h: List[torch.Tensor] = []
                for hi in range(num_h):
                    # Camera-query features (1, 4, H) of the predicted chunk
                    # (with grad), used by the next-frame object losses.
                    pred_seq = pred_chunks[hi]
                    qfeat_pred_h.append(self.ik_attn_pool(pred_seq))
                    tgt = target_scenes[hi]
                    if tgt is not None and torch.is_tensor(tgt):
                        # Select this sample from a batched target; cast to hidden's dtype.
                        if tgt.ndim == 3:
                            tgt = tgt[b] if b < tgt.shape[0] else tgt[0]
                        tgt = tgt.to(device=hidden.device, dtype=hidden.dtype)
                        tgt_len_raw = int(tgt.shape[0])
                        if pred_seq.numel() > 0:
                            # MSE between predicted and (length-aligned) target
                            # tokens, averaged over tokens and channels.
                            tgt_aligned = self._align_len(tgt, pred_seq.shape[0])
                            bev_losses_h.append(F.mse_loss(pred_seq, tgt_aligned))
                            # Diagnostics for the first horizon of the first
                            # sample: token count, mean L2 norms, mean cosine.
                            if hi == 0 and dbg_pred_norm is None:
                                with torch.no_grad():
                                    pf = pred_seq.detach().float()
                                    tf = tgt_aligned.detach().float()
                                    dbg_bev_k = int(pred_seq.shape[0])
                                    dbg_tgt_len = int(tgt_len_raw)
                                    dbg_pred_norm = pf.norm(dim=-1).mean()
                                    dbg_tgt_norm = tf.norm(dim=-1).mean()
                                    dbg_cos = F.cosine_similarity(pf, tf, dim=-1).mean()
                        else:
                            # No predicted tokens for this horizon -> zero loss.
                            bev_losses_h.append(
                                torch.zeros((), device=hidden.device, dtype=hidden.dtype)
                            )
                    else:
                        # Missing next frame: zero BEV loss for this horizon (it
                        # still counts in the mean).
                        bev_losses_h.append(
                            torch.zeros((), device=hidden.device, dtype=hidden.dtype)
                        )
                next_qfeat_pred_all.append(qfeat_pred_h)
                # Per-sample BEV loss = mean over horizons.
                per_sample_bev.append(torch.stack(bev_losses_h).mean())
            # Batch BEV loss = mean over samples.
            bev_loss = torch.stack(per_sample_bev).mean()
            # ---- IK (trajectory) head ----
            # Run the IK branch when either (a) the IK loss is on and a GT
            # trajectory is available, or (b) a sampled trajectory was requested
            # (evaluation, GRPO reference pass).
            need_ik = (ik_w > 0 and torch.is_tensor(gt_traj)) or bool(return_ik_pred_for_debug)
            if need_ik:
                # Current visual state: the same projected DINOv3 tokens that were
                # inserted at <SCENE>, recomputed here without grad (so the IK
                # loss never updates mm_projector_bev).
                with torch.no_grad():
                    cur_scene = self._project_scene_feature_from_vt_result(uniad_pth)
                # Drop a leading batch dim (sample 0) -> (N, H) in hidden's dtype.
                # These are the token-level current-frame conditioning of the
                # cross-attention IK head.
                cur_tokens = cur_scene[0] if cur_scene.ndim == 3 else cur_scene
                cur_tokens = cur_tokens.to(device=hidden.device, dtype=hidden.dtype)
                # Camera-query features of the current frame (current-frame
                # object losses) and of the predicted next frames of sample 0
                # (next-frame object losses). The latter come from the
                # predicted-state hidden states and are NOT detached by
                # ik_stop_grad below, so the next-frame object losses
                # backpropagate into the LLM hidden states.
                q_cur = self.ik_attn_pool(cur_tokens)
                q_next_list = list(next_qfeat_pred_all[0])
                # Token-level predicted next-frame conditioning: the <BEV_STATE>
                # hidden states of sample 0, split per horizon with the same
                # rule as the BEV loss above ((1, H) zero tokens if missing).
                next_tokens_list: List[torch.Tensor] = []
                pred_all = hidden[0][mask[0]]
                # No <BEV_STATE> tokens: one (1, H) zero token per horizon.
                if pred_all.numel() == 0:
                    next_tokens_list = [
                        torch.zeros((1, hidden.size(-1)), device=hidden.device, dtype=hidden.dtype)
                        for _ in range(num_h)
                    ]
                # Expected layout: consecutive blocks of exp_k (540) tokens.
                elif pred_all.shape[0] == exp_k * num_h:
                    for hi in range(num_h):
                        next_tokens_list.append(pred_all[hi * exp_k : (hi + 1) * exp_k])
                # Otherwise equal splits (remainder to the last horizon);
                # an empty split becomes a single zero token.
                else:
                    step = max(1, pred_all.shape[0] // num_h)
                    for hi in range(num_h):
                        s = hi * step
                        e = (
                            pred_all.shape[0]
                            if hi == num_h - 1
                            else min(pred_all.shape[0], (hi + 1) * step)
                        )
                        seq = pred_all[s:e]
                        if seq.numel() == 0:
                            seq = torch.zeros(
                                (1, hidden.size(-1)), device=hidden.device, dtype=hidden.dtype
                            )
                        next_tokens_list.append(seq)
                # Stop-grad: detach the conditioning so the IK loss trains only
                # the IK head, not the LLM. Active when
                # self.ik_stop_grad is True: curriculum stage 1, and every GRPO
                # step (the GRPO trainer forces it). In SFT stage 2 the
                # curriculum callback sets it to False, so gradients flow into
                # the LLM through next_tokens_list. cur_tokens never carries
                # LLM gradient (computed under no_grad above).
                if ik_stop_grad:
                    cur_tokens = cur_tokens.detach()
                    next_tokens_list = [t.detach() for t in next_tokens_list]
                # Hand the conditioning tokens to the GRPO trainer: detached
                # copies (sampling / reference policy) and the undetached
                # tensors (log-prob recomputation).
                self.ik_head._grpo_cur_tokens = cur_tokens.detach()
                self.ik_head._grpo_next_tokens_list = [t.detach() for t in next_tokens_list]
                self.ik_head._grpo_cur_tokens_grad = cur_tokens
                self.ik_head._grpo_next_tokens_list_grad = list(next_tokens_list)
                # Meters per normalised trajectory unit (--ik-traj-scale, 20.0).
                traj_scale = float(self.ik_traj_scale)
                # Sampled trajectory for evaluation / GRPO: DDIM with
                # ik_debug_sample_steps steps (50 by default, capped at the
                # training T), eta = 0 (deterministic given the random x_T),
                # fp32. Converted to meters and reshaped to (1, T, 2).
                if return_ik_pred_for_debug:
                    with torch.no_grad():
                        pred_flat = self.ik_head.sample(
                            cur_tokens,
                            next_tokens_list,
                            num_steps=int(self.ik_debug_sample_steps),
                        )
                        # Normalised -> meters; (1, 2T) -> (1, T, 2).
                        pred_flat_m = pred_flat * traj_scale
                        dbg_ik_pred_traj = pred_flat_m.view(pred_flat_m.shape[0], -1, 2).detach()
                        # Matching GT trajectory (1, T, 2) in meters for the eval L2/ADE.
                        if torch.is_tensor(gt_traj):
                            gtt = gt_traj.to(device=hidden.device, dtype=hidden.dtype)
                            if gtt.ndim == 2:
                                gtt = gtt.unsqueeze(0)
                            dbg_ik_gt_traj = gtt.detach()
                # ---- IK diffusion loss ----
                if ik_w > 0 and torch.is_tensor(gt_traj):
                    # GT trajectory (T, 2) meters -> (1, T, 2) -> flattened
                    # [x1, y1, ..., xT, yT] (1, 2T) -> normalised by traj_scale.
                    gt = gt_traj.to(device=hidden.device, dtype=hidden.dtype)
                    if gt.ndim == 2:
                        gt = gt.unsqueeze(0)
                    _gt_xad = gt.reshape(gt.shape[0], -1) / traj_scale
                    # Make the target exactly traj_dim wide: truncate extra
                    # waypoints or zero-pad missing ones.
                    _xad_td = self.ik_head.traj_dim
                    if _gt_xad.shape[-1] != _xad_td:
                        if _gt_xad.shape[-1] > _xad_td:
                            _gt_xad = _gt_xad[..., :_xad_td]
                        else:
                            _gt_xad = torch.nn.functional.pad(
                                _gt_xad, (0, _xad_td - _gt_xad.shape[-1])
                            )
                    # eps-MSE diffusion loss with mc = ik_diff_mc (8) Monte-Carlo
                    # (t, noise) draws and axis weights (longitudinal, lateral).
                    ik_loss = self.ik_head.diffusion_loss(
                        cur_tokens,
                        next_tokens_list,
                        _gt_xad,
                        mc=getattr(self, "ik_diff_mc", None),
                        weight_xy=(ik_w_lon, ik_w_lat),
                    )
                else:
                    # IK loss off (stage 1 / evaluation) or no GT trajectory
                    # (e.g. the sample has no valid future): zero IK loss.
                    ik_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
            else:
                # IK branch skipped entirely: zero IK loss.
                ik_loss = torch.zeros((), device=hidden.device, dtype=hidden.dtype)
            # ---- Total loss ----
            # loss = bev_w * bev_loss + qa_w * lm_loss + ik_w * ik_loss
            #        (+ object losses below).
            loss = bev_w * bev_loss
            if qa_w > 0:
                loss = loss + qa_w * lm_loss
            if ik_w > 0:
                loss = loss + ik_w * ik_loss
            # ---- Camera-query object auxiliary losses ----
            # Weights (NAVSIM recipe: 0.25 / 0.25 current, 0.15 / 0.15 next).
            _obj_w = float(getattr(self, "obj_loss_weight", 0.5) or 0.0)
            _box_w = float(getattr(self, "box_loss_weight", 0.5) or 0.0)
            _obj_wn = float(getattr(self, "obj_loss_weight_next", 0.3) or 0.0)
            _box_wn = float(getattr(self, "box_loss_weight_next", 0.3) or 0.0)
            obj_head = self.object_query_head
            # Current frame: q_cur (1, 4, H) -> boxes (1, 4, 6), logits (1, 4).
            # Presence: BCE-with-logits vs has_obj. Box: L1 averaged over the
            # 6 box dims, counted only for queries with an object and
            # normalised by the number of such queries (>= 1). Needs the IK
            # branch above (q_cur) and the collated targets.
            if (
                q_cur is not None
                and torch.is_tensor(query_box_targets)
                and torch.is_tensor(query_has_obj)
            ):
                box_pred_c, obj_pred_c = obj_head(q_cur)
                tgt_c = query_box_targets.to(device=q_cur.device, dtype=q_cur.dtype)
                has_c = query_has_obj.to(device=q_cur.device, dtype=q_cur.dtype)
                obj_loss_val = F.binary_cross_entropy_with_logits(obj_pred_c, has_c)
                box_err_c = (box_pred_c - tgt_c).abs().mean(dim=-1)
                box_loss_val = (box_err_c * has_c).sum() / has_c.sum().clamp_min(1.0)
                if _obj_w > 0:
                    loss = loss + _obj_w * obj_loss_val
                if _box_w > 0:
                    loss = loss + _box_w * box_loss_val
            # Future horizons: the same losses on the query features of the
            # *predicted* next-frame tokens vs. the future GT boxes
            # (targets (1, K, 4, 6) / (1, K, 4)), averaged over horizons.
            if (
                q_next_list
                and torch.is_tensor(query_box_targets_next)
                and torch.is_tensor(query_has_obj_next)
            ):
                K_h = min(len(q_next_list), query_box_targets_next.shape[1])
                obj_ks = []
                box_ks = []
                for k in range(K_h):
                    # Horizon k: query features (1, 4, H) vs targets [:, k].
                    q_k = q_next_list[k]
                    tgt_k = query_box_targets_next[:, k].to(device=q_k.device, dtype=q_k.dtype)
                    has_k = query_has_obj_next[:, k].to(device=q_k.device, dtype=q_k.dtype)
                    box_pred_k, obj_pred_k = obj_head(q_k)
                    obj_ks.append(F.binary_cross_entropy_with_logits(obj_pred_k, has_k))
                    err_k = (box_pred_k - tgt_k).abs().mean(dim=-1)
                    box_ks.append((err_k * has_k).sum() / has_k.sum().clamp_min(1.0))
                # Average over the horizons, then weight.
                if obj_ks:
                    obj_loss_next_val = torch.stack(obj_ks).mean()
                    box_loss_next_val = torch.stack(box_ks).mean()
                    if _obj_wn > 0:
                        loss = loss + _obj_wn * obj_loss_next_val
                    if _box_wn > 0:
                        loss = loss + _box_wn * box_loss_next_val
        # ---- Outputs ----
        # Tuple form (return_dict=False).
        if not return_dict:
            return (loss, logits)
        # Standard fields. hidden_states is always None in the returned output
        # (also when output_hidden_states was requested for the attention debug).
        out = FlexibleCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=None,
            attentions=outputs.attentions,
        )
        # Individual loss terms (for logging; the trainer reads lm_loss,
        # bev_loss and ik_loss).
        out["lm_loss"] = lm_loss
        out["bev_loss"] = bev_loss
        out["ik_loss"] = ik_loss
        # Camera-query object losses (None unless computed).
        out["obj_loss"] = obj_loss_val
        out["box_loss"] = box_loss_val
        out["obj_loss_next"] = obj_loss_next_val
        out["box_loss_next"] = box_loss_next_val
        # ---- wm_* diagnostics ----
        # Curriculum stage and whether the IK loss is currently active.
        out["wm_stage"] = int(getattr(self, "curriculum_stage", 1) or 1)
        out["wm_ik_enabled"] = bool(float(getattr(self, "ik_loss_weight", 0.0) or 0.0) > 0)
        # BEV prediction stats (first horizon of the first sample).
        out["wm_bev_k"] = dbg_bev_k
        out["wm_tgt_len"] = dbg_tgt_len
        out["wm_bev_pred_norm"] = dbg_pred_norm
        out["wm_bev_tgt_norm"] = dbg_tgt_norm
        out["wm_bev_cos"] = dbg_cos
        # Sampled IK trajectory and GT, (1, T, 2) in meters; consumed by the
        # evaluation loop (ADE / L2@k s) when return_ik_pred_for_debug is set.
        out["wm_ik_pred_traj"] = dbg_ik_pred_traj if return_ik_pred_for_debug else None
        out["wm_ik_gt_traj"] = dbg_ik_gt_traj if return_ik_pred_for_debug else None
        # Sequence lengths before / after placeholder expansion and CE target counts.
        out["wm_text_seq_len"] = dbg_text_L
        out["wm_mm_seq_len"] = (
            int(logits.shape[1]) if torch.is_tensor(logits) and logits.ndim == 3 else dbg_mm_L
        )
        out["wm_ce_tokens_raw"] = dbg_ce_raw
        out["wm_bev_span_mask_applied"] = dbg_bev_span_mask_applied
        # Expanded labels / attention mask (the expanded input ids are None
        # after multimodal preparation, so wm_mm_input_ids is normally None).
        if return_mm_labels_for_debug:
            out["wm_mm_input_ids"] = mm_input_ids_dbg
            out["wm_mm_labels"] = mm_labels_dbg
            out["wm_mm_attention_mask"] = mm_attn_dbg
        # Attention shares on visual / ego / text inputs (see the attention
        # diagnostic above); *_ratio values are percentages.
        if return_attn_debug:
            out["wm_attn_bev_weight"] = dbg_attn_bev_weight
            out["wm_attn_ego_weight"] = dbg_attn_ego_weight
            out["wm_attn_text_weight"] = dbg_attn_text_weight
            out["wm_attn_bev_ratio"] = dbg_attn_bev_ratio
            out["wm_attn_ego_ratio"] = dbg_attn_ego_ratio
            out["wm_attn_text_ratio"] = dbg_attn_text_ratio
            out["wm_attn_bev_per_tok"] = dbg_attn_bev_per_tok
            out["wm_attn_ego_per_tok"] = dbg_attn_ego_per_tok
            out["wm_attn_text_per_tok"] = dbg_attn_text_per_tok
        return out
