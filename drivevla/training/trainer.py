"""Training diagnostics, validation and task sampling.

This module holds the Hugging Face ``Trainer`` subclasses and callbacks that drive
every DriveIK training stage. ``drivevla/training/train.py::main`` builds the model,
datasets and ``TrainingArguments`` and then instantiates one of the trainers defined
here (or the GRPO trainer from ``grpo.py``, which inherits from both of them).

Contents
--------
* Logging helpers (``_wm_to_int``, ``_wm_add_scalar_or_stats``,
  ``_wm_collect_log_dict``): turn the many auxiliary scalars/tensors returned by
  ``BEVLatentActionWorldModelMixin.forward`` (``lm_loss``, ``bev_loss``, ``ik_loss``,
  ``wm_*`` diagnostics, ...) into a flat ``{name: number}`` dict for ``Trainer.log``.
* ``LatentActionWMTrainer``: a lean ``Trainer`` that only adds the IK-aware optimizer
  (separate LR for the IK modules), shape-tolerant optimizer resume and the
  auxiliary-loss logging. It is the base class of the NAVSIM GRPO trainer.
* ``LatentActionDebugTrainer``: the trainer used by the SFT recipe (and mixed
  into the GRPO trainer). On top of ``common.DebugTrainer``'s prompt/label previews
  it prints IK trajectory / attention diagnostics on debug steps, and it
  overrides ``evaluate`` with the DriveIK validation loop:
    - world-model (``wm``) split: forward pass + IK diffusion sampling, reporting
      next-visual-state loss (``bev_loss``), trajectory ADE and per-horizon L2 (m),
      and the share of attention that assistant tokens put on visual / ego / text
      input tokens (metric prefix ``eval_wm/``).
* ``CurriculumEpochSampler``: DDP-safe, seeded epoch/stage-dependent shuffle
  sampler that ``train.main`` installs as the training sampler.
* ``TwoStageCurriculumCallback``: switches the model to curriculum stage 2 (IK loss
  with full gradient flow into the LLM) at train begin and feeds the epoch number
  to the sampler.
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
import torch
from transformers import Trainer
from transformers.trainer_callback import TrainerCallback
from llava.model import LlavaQwenForCausalLM
from llava.utils import rank0_print
from . import common as wm
from .checkpoint import _create_optimizer_with_ik_lr, _load_optimizer_and_scheduler_with_remap


def _wm_to_int(x):
    """Best-effort scalar -> int.

    Args:
        x: ``None``, a ``bool``, a one-element tensor (any dtype/device), or anything
            accepted by ``int()``.

    Returns:
        ``int`` value of ``x``, or ``None`` when ``x`` is ``None``, a multi-element
        tensor, or cannot be converted. Never raises.
    """
    # Missing value -> nothing to log.
    if x is None:
        return None
    # bool is a subclass of int; convert explicitly to 0/1.
    if isinstance(x, bool):
        return int(x)
    # Tensors: only one-element tensors are converted (moved to CPU first);
    # vectors and conversion errors yield None.
    if torch.is_tensor(x):
        try:
            if x.numel() == 1:
                return int(x.detach().cpu().item())
        except Exception:
            return None
        return None
    # Plain Python / NumPy scalars, numeric strings, etc.
    try:
        return int(x)
    except Exception:
        return None


def _wm_add_scalar_or_stats(logs: dict, key: str, val):
    """Add a scalar to logs; if val is a vector tensor, add summary stats instead.

    Args:
        logs: Dict that is updated in place (the ``Trainer.log`` payload).
        key: Metric name, e.g. ``"bev_loss"``.
        val: ``None`` (ignored), ``bool`` (stored as 0/1), a tensor of any shape,
            or a Python number.

    Side effects:
        * one-element tensor / number -> ``logs[key] = float(val)``;
        * tensor with N > 1 elements -> ``logs[key/mean|std|min|max]`` over the
          flattened values (population std, 0.0 when N == 1);
        * empty tensors and values that cannot be converted are silently skipped.

    Returns:
        ``None``. Never raises.
    """
    # Nothing to record.
    if val is None:
        return
    # Booleans are logged as integers so W&B / TensorBoard can plot them.
    if isinstance(val, bool):
        logs[key] = int(val)
        return
    if torch.is_tensor(val):
        try:
            # Scalar tensor -> plain float.
            if val.numel() == 1:
                logs[key] = float(val.detach().float().cpu().item())
                return
            # Vector tensor (e.g. per-sample losses or diffusion timesteps):
            # flatten and log summary statistics instead of the raw values.
            v = val.detach().float().flatten()
            if v.numel() == 0:
                return
            logs[f"{key}/mean"] = float(v.mean().cpu().item())
            logs[f"{key}/std"] = float(v.std(unbiased=False).cpu().item()) if v.numel() > 1 else 0.0
            logs[f"{key}/min"] = float(v.min().cpu().item())
            logs[f"{key}/max"] = float(v.max().cpu().item())
            return
        except Exception:
            return
    # Non-tensor values: anything float() accepts; otherwise skip silently.
    try:
        logs[key] = float(val)
    except Exception:
        pass


def _wm_collect_log_dict(outputs: Any, inputs: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Collect all numeric WM/IK debug scalars for W&B logging.

    (The values go through ``Trainer.log``; in this release ``train.main`` sets
    ``report_to=["tensorboard"]``, so they end up in TensorBoard and the console.)

    Args:
        outputs: The model forward result (a ``dict`` or a ``ModelOutput``-like
            object) produced by ``BEVLatentActionWorldModelMixin.forward``. Missing
            keys are fine; every lookup falls back to ``None``.
        inputs: The batch dict passed to the model (optional). Only its
            ``wm_meta`` dict (built by ``DataCollatorForLatentActionWM``) is read.

    Returns:
        A flat ``{metric_name: int | float}`` dict ready for ``Trainer.log``. Vector
        tensors are expanded to ``/mean``, ``/std``, ``/min``, ``/max`` entries by
        ``_wm_add_scalar_or_stats``. ``traj_loss`` is an alias of ``ik_loss``.
    """
    logs: Dict[str, Any] = {}

    def _get(o, k):
        """Read key ``k`` from a dict-like (``.get``) or attribute-style object ``o``.

        Returns ``None`` if ``o`` is ``None``, the key/attribute is missing, or the
        lookup raises.
        """
        if o is None:
            return None
        # dicts and transformers ModelOutput both expose .get().
        if hasattr(o, "get"):
            try:
                return o.get(k, None)
            except Exception:
                return None
        # Fallback for plain objects / namespaces.
        return getattr(o, k, None)

    # Main loss components: LM cross-entropy, next-visual-state (BEV) loss, IK loss.
    # "traj_loss" duplicates ik_loss under a second name for dashboards.
    _wm_add_scalar_or_stats(logs, "lm_loss", _get(outputs, "lm_loss"))
    _wm_add_scalar_or_stats(logs, "bev_loss", _get(outputs, "bev_loss"))
    _wm_add_scalar_or_stats(logs, "ik_loss", _get(outputs, "ik_loss"))
    if "ik_loss" in logs:
        logs["traj_loss"] = logs["ik_loss"]
    # Camera-query object losses. Each one is only logged when the forward pass
    # returned it (None values are skipped).
    _wm_add_scalar_or_stats(logs, "obj_loss", _get(outputs, "obj_loss"))
    _wm_add_scalar_or_stats(logs, "box_loss", _get(outputs, "box_loss"))
    _wm_add_scalar_or_stats(logs, "obj_loss_next", _get(outputs, "obj_loss_next"))
    _wm_add_scalar_or_stats(logs, "box_loss_next", _get(outputs, "box_loss_next"))
    # Curriculum stage (1/2) the model ran with, and whether the IK loss was active
    # (ik_loss_weight > 0) for this forward pass.
    st = _wm_to_int(_get(outputs, "wm_stage"))
    if st is not None:
        logs["wm_stage"] = st
    ik_en = _get(outputs, "wm_ik_enabled")
    if ik_en is not None:
        logs["wm_ik_enabled"] = int(bool(ik_en))
    # Next-visual-state prediction diagnostics (horizon, target length, norms of
    # predicted/target features and their cosine similarity).
    for k in ["wm_bev_k", "wm_tgt_len", "wm_bev_pred_norm", "wm_bev_tgt_norm", "wm_bev_cos"]:
        _wm_add_scalar_or_stats(logs, k, _get(outputs, k))
    # Sequence-length / cross-entropy token-count diagnostics (text length vs.
    # multimodal length after feature expansion, supervised CE tokens before
    # masking the BEV span out of the LM loss).
    for k in [
        "wm_text_seq_len",
        "wm_mm_seq_len",
        "wm_ce_tokens_raw",
    ]:
        _wm_add_scalar_or_stats(logs, k, _get(outputs, k))
    # Whether the <bev_start>...<bev_end> span was masked out of the LM labels.
    bspan = _get(outputs, "wm_bev_span_mask_applied")
    if bspan is not None:
        logs["wm_bev_span_mask_applied"] = int(bool(bspan))
    # Batch-level metadata produced by the collator (not by the model).
    if isinstance(inputs, dict):
        meta = inputs.get("wm_meta", {})
        if isinstance(meta, dict):
            # Token length before truncation and the truncation limit.
            for mk, outk in [("orig_len", "wm_orig_len"), ("max_len", "wm_max_len")]:
                v = meta.get(mk, None)
                vi = _wm_to_int(v)
                if vi is not None:
                    logs[outk] = vi
            # Whether the sample was truncated.
            wt = meta.get("was_truncated", None)
            if wt is not None:
                logs["wm_was_truncated"] = int(bool(wt))
    return logs


class LatentActionWMTrainer(Trainer):
    """Trainer that logs auxiliary losses (lm/bev/ik) when available.

    Minimal ``transformers.Trainer`` subclass:

    * ``create_optimizer`` puts the IK modules (``ik_head``, ``ik_attn_pool``, ...)
      in their own parameter groups so they can use ``--ik-learning-rate``
      (see ``checkpoint._create_optimizer_with_ik_lr``);
    * ``_load_optimizer_and_scheduler`` resumes optimizer state while skipping
      entries whose shapes no longer match the model;
    * ``compute_loss`` runs the model once and logs the auxiliary losses /
      diagnostics returned by the forward pass.

    It is not used directly by ``train.main`` (which defaults to
    ``LatentActionDebugTrainer``); it is the base class of
    ``grpo.GRPOLatentActionWMTrainer``, whose ``compute_loss`` calls this one via
    ``super()`` to obtain the SFT loss.
    """

    def create_optimizer(self):
        """Build (or return the existing) optimizer with a separate IK learning rate."""
        return _create_optimizer_with_ik_lr(self)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resume optimizer/scheduler state from ``checkpoint`` with shape filtering."""
        return _load_optimizer_and_scheduler_with_remap(self, checkpoint)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Forward the batch and return the total loss, logging auxiliaries on rank 0.

        Args:
            model: The (possibly DDP-wrapped) ``LlavaQwenForCausalLM`` whose
                ``forward`` was replaced by ``BEVLatentActionWorldModelMixin.forward``.
            inputs: Collated batch dict from ``DataCollatorForLatentActionWM``
                (batch size 1), passed to the model as keyword arguments.
            return_outputs: If True, also return the raw model outputs.
            **kwargs: Extra arguments from newer ``Trainer.training_step``
                (e.g. ``num_items_in_batch``); accepted and ignored.

        Returns:
            ``loss`` (scalar tensor) or ``(loss, outputs)``.

        Side effects:
            Calls ``self.log`` at most once per ``global_step`` (on steps that are a
            multiple of ``logging_steps``) and records that step in
            ``self._wm_last_logged_step``.
        """
        # Single forward pass; the model returns the combined (weighted) training
        # loss under "loss" plus the individual components / diagnostics.
        outputs = model(**inputs)
        loss = outputs["loss"] if isinstance(outputs, dict) else outputs.loss
        # Current optimizer step (defaults to 0 if the trainer state is missing).
        try:
            gs = int(getattr(self.state, "global_step", 0) or 0)
        except Exception:
            gs = 0
        # Logging period, clamped to >= 1.
        try:
            log_every = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
        except Exception:
            log_every = 1
        # Lazily initialise the "last step we logged" marker.
        if not hasattr(self, "_wm_last_logged_step"):
            self._wm_last_logged_step = -1
        # Log only on the main process, on logging steps, and only once per global
        # step (compute_loss runs several times per step with gradient accumulation).
        should_log = (
            self.is_world_process_zero()
            and gs % log_every == 0
            and (gs != int(getattr(self, "_wm_last_logged_step", -1)))
        )
        if should_log:
            logs = _wm_collect_log_dict(outputs, inputs)
            _wm_add_scalar_or_stats(logs, "loss_total", loss)
            if len(logs) > 0:
                self.log(logs)
            self._wm_last_logged_step = int(gs)
        return (loss, outputs) if return_outputs else loss


class LatentActionDebugTrainer(wm.DebugTrainer):
    """DebugTrainer variant for latent-action WM.

    It is the default trainer class of ``train.main`` for the SFT recipe and
    the checkpoint-evaluation entry point, and it is also mixed into the NAVSIM GRPO
    trainer (``grpo.run``), which inherits its ``__init__`` (debug kwargs) and its
    ``evaluate`` override.

    On debug steps (every ``--debug-print-steps`` steps) it additionally:

    * asks the model forward to sample an IK trajectory and to return attention
      diagnostics (``return_ik_pred_for_debug`` / ``return_attn_debug``);
    * prints the predicted vs. GT trajectory (meters) and the assistant->BEV/EGO/TEXT
      attention split.

    ``evaluate`` is replaced by a rank-0-only validation loop (``_evaluate_wm``)
    whose sample budget is ``_EVAL_MAX_WM_SAMPLES``.
    """

    def create_optimizer(self):
        """Build (or return the existing) optimizer with a separate IK learning rate."""
        return _create_optimizer_with_ik_lr(self)

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Resume optimizer/scheduler state from ``checkpoint`` with shape filtering."""
        return _load_optimizer_and_scheduler_with_remap(self, checkpoint)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """Like wm.DebugTrainer.compute_loss, but also requests IK predictions on debug steps
        and prints a small IK trajectory preview (pred/gt) when available.

        Args:
            model: The (possibly DDP-wrapped) DriveIK model.
            inputs: Collated batch dict (batch size 1) including ``wm_meta``. On debug
                steps this dict is modified in place to add the
                ``return_*_debug`` request flags consumed by the model forward.
            return_outputs: If True, return ``(loss, outputs)``.
            num_items_in_batch: Passed through to the parent ``compute_loss``
                (newer ``transformers`` versions supply it).
            **kwargs: Passed through to the parent ``compute_loss``.

        Returns:
            ``loss`` (scalar tensor) or ``(loss, outputs)``.

        Side effects:
            Rank-0 console prints on debug steps and ``self.log`` of the auxiliary
            losses once per logging step. All diagnostic code is wrapped in
            ``try/except`` so that a printing failure never interrupts training.
        """
        # Current optimizer step.
        try:
            gs = int(getattr(self.state, "global_step", 0) or 0)
        except Exception:
            gs = 0
        # Collator metadata (prompt pieces, ids, truncation flag, ...).
        meta = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
        was_trunc = bool(meta.get("was_truncated", False))
        # Decide whether this is a debug step: a step selected by
        # DebugTrainer._should_print (every --debug-print-steps steps).
        request_debug = False
        try:
            if (
                int(getattr(self, "_dbg_steps", 0) or 0) > 0
                and hasattr(self, "_should_print")
                and self._should_print(gs, was_trunc)
            ):
                request_debug = True
        except Exception:
            request_debug = False
        # On debug steps, ask the forward pass to (a) sample an IK trajectory for
        # printing and (b) compute the attention split. The forward pops these
        # flags; they only add diagnostic outputs (the extra IK sampling runs
        # under torch.no_grad()).
        if request_debug and isinstance(inputs, dict):
            inputs["return_ik_pred_for_debug"] = True
            inputs["return_attn_debug"] = True
        # Delegate the actual loss computation (and DebugTrainer's own previews) to
        # the parent class; always keep the outputs for the diagnostics below.
        loss, outputs = super().compute_loss(
            model, inputs, return_outputs=True, num_items_in_batch=num_items_in_batch, **kwargs
        )
        # (1) IK diagnostics on debug steps (rank 0 only): loss summary line and
        # predicted vs. GT waypoints.
        try:
            if request_debug and self.is_world_process_zero():
                # Pull the IK-related outputs; every lookup tolerates a missing key
                # or an outputs object without .get().
                ik_loss = outputs.get("ik_loss", None) if hasattr(outputs, "get") else None
                ik_pred = outputs.get("wm_ik_pred_traj", None) if hasattr(outputs, "get") else None
                ik_gt = outputs.get("wm_ik_gt_traj", None) if hasattr(outputs, "get") else None
                stage = outputs.get("wm_stage", None) if hasattr(outputs, "get") else None
                # One-line summary of the step: stage and IK loss.
                rank0_print(f"[DEBUG_EXTRA] step={gs} stage={stage} ik_loss={ik_loss}")
                # Predicted / GT trajectories are (B, N_wp, 2) tensors in meters
                # (the model rescales the sampled normalised trajectory by
                # ik_traj_scale). Print the first sample's waypoints rounded to mm.
                if torch.is_tensor(ik_pred):
                    p = ik_pred.detach().float().cpu()
                    formatted = [
                        [round(v, 3) for v in wp]
                        for wp in (p[0].tolist() if p.ndim == 3 else p.tolist())
                    ]
                    rank0_print("[IK_PRED] " + str(formatted))
                if torch.is_tensor(ik_gt):
                    g = ik_gt.detach().float().cpu()
                    formatted = [
                        [round(v, 3) for v in wp]
                        for wp in (g[0].tolist() if g.ndim == 3 else g.tolist())
                    ]
                    rank0_print("[IK_GT]   " + str(formatted))
        except Exception as e:
            # Any failure in the IK diagnostics is reported and ignored.
            if self.is_world_process_zero():
                rank0_print(f"[WARN][IK_DEBUG] Failed: {type(e).__name__}: {e}")
        # (2) Attention split on debug steps (rank 0 only). The model forward (with
        # return_attn_debug=True) recomputes last-layer attention for the first
        # sample and reports, for assistant (label != IGNORE) query positions:
        #   *_weight  : attention mass on BEV / ego / text input tokens, averaged
        #               over assistant tokens;
        #   *_ratio   : the same split as a percentage of the total (sums to 100);
        #   *_per_tok : mass per input token of that category.
        try:
            if request_debug and self.is_world_process_zero():
                is_training = model.training
                if hasattr(outputs, "get"):
                    attn_bev_w = outputs.get("wm_attn_bev_weight", None)
                    attn_ego_w = outputs.get("wm_attn_ego_weight", None)
                    attn_text_w = outputs.get("wm_attn_text_weight", None)
                    attn_bev_r = outputs.get("wm_attn_bev_ratio", None)
                    attn_ego_r = outputs.get("wm_attn_ego_ratio", None)
                    attn_text_r = outputs.get("wm_attn_text_ratio", None)
                    attn_bev_pt = outputs.get("wm_attn_bev_per_tok", None)
                    attn_ego_pt = outputs.get("wm_attn_ego_per_tok", None)
                    attn_text_pt = outputs.get("wm_attn_text_per_tok", None)
                    # Print only if at least one category was computed. Each part
                    # reads "<mean mass per assistant token> (<percent>%)".
                    if attn_bev_w is not None or attn_ego_w is not None or attn_text_w is not None:
                        mode_str = "TRAIN" if is_training else "EVAL"
                        bev_str = (
                            f"{attn_bev_w:.4f} ({attn_bev_r:.1f}%)"
                            if attn_bev_w is not None
                            else "N/A"
                        )
                        ego_str = (
                            f"{attn_ego_w:.4f} ({attn_ego_r:.1f}%)"
                            if attn_ego_w is not None
                            else "N/A"
                        )
                        text_str = (
                            f"{attn_text_w:.4f} ({attn_text_r:.1f}%)"
                            if attn_text_w is not None
                            else "N/A"
                        )
                        rank0_print(
                            f"[ATTN_WEIGHT] [{mode_str}] step={gs} assistant→BEV={bev_str} | assistant→EGO={ego_str} | assistant→TEXT={text_str}"
                        )
                        # Second line with the per-token normalised attention mass.
                        if attn_bev_pt is not None:
                            bev_pt_str = f"{attn_bev_pt:.6f}" if attn_bev_pt is not None else "N/A"
                            ego_pt_str = f"{attn_ego_pt:.6f}" if attn_ego_pt is not None else "N/A"
                            text_pt_str = (
                                f"{attn_text_pt:.6f}" if attn_text_pt is not None else "N/A"
                            )
                            rank0_print(
                                f"[ATTN_WEIGHT/tok] [{mode_str}] step={gs} BEV/tok={bev_pt_str} | EGO/tok={ego_pt_str} | TEXT/tok={text_pt_str}"
                            )
        except Exception as _attn_e:
            if self.is_world_process_zero():
                rank0_print(f"[WARN][ATTN_DEBUG] Failed: {type(_attn_e).__name__}: {_attn_e}")
        # (3) Log the auxiliary losses / diagnostics, at most once per global step
        # on logging steps (same de-duplication as LatentActionWMTrainer). Errors
        # are swallowed so logging can never stop training.
        try:
            if not hasattr(self, "_wm_last_logged_step"):
                self._wm_last_logged_step = -1
            gs_log = int(getattr(self.state, "global_step", 0) or 0)
            log_every = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
            do_log = (
                self.is_world_process_zero()
                and gs_log % log_every == 0
                and (gs_log != int(getattr(self, "_wm_last_logged_step", -1)))
            )
            if do_log:
                logs = _wm_collect_log_dict(outputs, inputs if isinstance(inputs, dict) else None)
                _wm_add_scalar_or_stats(logs, "loss_total", loss)
                if len(logs) > 0:
                    self.log(logs)
                self._wm_last_logged_step = int(gs_log)
        except Exception:
            pass
        return (loss, outputs) if return_outputs else loss

    # Validation sample budget (class default). train.main overwrites it on the
    # trainer instance from --eval-max-wm-samples, using float("inf") for a value
    # <= 0 so that the whole split is evaluated.
    _EVAL_MAX_WM_SAMPLES = 500
    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval", **kwargs):
        """Override evaluate to compute, on the world-model validation split:
        forward + IK head → bev_loss, trajectory ADE and per-horizon L2 (meters),
        attention split over BEV / ego / text tokens.

        Evaluation runs on the main process only, one sample at a time, on the
        unwrapped model (no DDP). Other ranks wait at a barrier and return ``{}``.
        Called by ``Trainer`` every ``eval_steps`` during training and directly by
        ``scripts/evaluate_checkpoint.py`` (which replaces ``Trainer.train``).

        Args:
            eval_dataset: Dataset to evaluate. Defaults to ``self.eval_dataset``
                (the WM validation dataset built by ``train.main``).
            ignore_keys: Accepted for ``Trainer`` API compatibility; unused.
            metric_key_prefix: Metric prefix; metrics are named
                ``f"{metric_key_prefix}_wm/<name>"`` (e.g. ``eval_wm/ade``).
            **kwargs: Ignored.

        Returns:
            ``{metric_name: float | int}`` on rank 0, ``{}`` on other ranks or when
            there is no evaluation dataset.

        Side effects:
            ``self.log(metrics)`` on rank 0; switches the model to ``eval()`` and back
            to ``train()``; calls ``torch.distributed.barrier()`` when distributed.
        """
        # Non-main ranks do no work; they just synchronise with rank 0 at the
        # barrier that rank 0 reaches after finishing its evaluation.
        if not self.is_world_process_zero():
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            return {}
        # Resolve the dataset to evaluate.
        eval_ds = eval_dataset if eval_dataset is not None else self.eval_dataset
        if eval_ds is None:
            return {}
        # Eval mode (disables dropout) and unwrap DDP so the forward does not try
        # to synchronise with the other (idle) ranks.
        self.model.eval()
        unwrapped = self.model.module if hasattr(self.model, "module") else self.model
        all_metrics = self._evaluate_wm(unwrapped, eval_ds, f"{metric_key_prefix}_wm")
        self.log(all_metrics)
        # Back to training mode and release the waiting ranks.
        self.model.train()
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        return all_metrics

    @torch.no_grad()
    def _evaluate_wm(self, model, dataset, prefix):
        """WM eval: BEV loss + IK trajectory ADE / per-horizon L2 (1s/2s/3s).

        For each selected validation sample the full model forward is run with
        ``return_ik_pred_for_debug=True``, which makes the IK diffusion head sample a
        future trajectory (DDIM, ``model.ik_debug_sample_steps`` steps, from a fresh
        random x_T) conditioned only on the current and predicted future visual
        states, and with ``return_attn_debug=True`` for the attention split.

        Sample selection: ``min(len(dataset), self._EVAL_MAX_WM_SAMPLES)`` samples
        at evenly spaced indices ``int(i * len / total)`` (default budget 500;
        ``train.main`` maps ``--eval-max-wm-samples 0`` to ``float("inf")``, i.e.
        the full split).

        Trajectory metrics (UniAD-style, meters). Predicted and GT trajectories are
        ``(B, N_wp, 2)`` tensors of (x, y) waypoints at 0.5 s spacing, with
        ``N_wp = model.ik_num_waypoints`` (8 for NAVSIM):

        * ``ade``: mean point-wise L2 over all waypoints of all evaluated samples;
        * ``l2_{s}s`` for s = 1 .. N_wp // 2: mean L2 of the *single* waypoint at
          s seconds (index ``2s - 1``). This is not ST-P3's metric, which averages
          the errors of all waypoints up to each horizon (and is therefore lower).

        Args:
            model: Unwrapped DriveIK model.
            dataset: World-model validation ``BEVLatentActionWorldModelDataset``.
            prefix: Metric key prefix (e.g. ``"eval_wm"``).

        Returns:
            Dict with ``{prefix}/bev_loss`` (mean next-visual-state loss over
            successful forwards), ``{prefix}/ade``, ``{prefix}/l2_{s}s``,
            ``{prefix}/attn_{bev,ego,text}_pct`` (percent of assistant-token
            attention), ``{prefix}/attn_{bev,ego,text}_per_tok`` (only if any is
            > 0), ``{prefix}/runtime`` (seconds) and ``{prefix}/num_samples``
            (number of trajectories scored). Keys whose counters stayed at zero are
            omitted.
        """
        import time as _time

        t0 = _time.time()
        collator = self.data_collator
        # Trajectory metrics need an attached IK head.
        has_ik = hasattr(model, "ik_head") and model.ik_head is not None
        # Running sums for the next-visual-state (BEV) loss.
        bev_loss_sum = 0.0
        loss_count = 0
        # Number of predicted waypoints and the per-waypoint-index L2 sums (meters);
        # traj_count counts scored trajectories (batch rows).
        _eval_nw = int(getattr(model, "ik_num_waypoints", 6) or 6)
        wp_l2_sums = [0.0] * _eval_nw
        traj_count = 0
        # Running sums for the attention split (percentages and per-token mass).
        attn_bev_ratio_sum = 0.0
        attn_ego_ratio_sum = 0.0
        attn_text_ratio_sum = 0.0
        attn_bev_per_tok_sum = 0.0
        attn_ego_per_tok_sum = 0.0
        attn_text_per_tok_sum = 0.0
        attn_count = 0
        # Pick up to _EVAL_MAX_WM_SAMPLES evenly spaced indices (all when the budget
        # is >= len(dataset), e.g. float("inf")).
        total = min(len(dataset), self._EVAL_MAX_WM_SAMPLES)
        if total < len(dataset):
            step = len(dataset) / total
            indices = [int(i * step) for i in range(total)]
        else:
            indices = list(range(total))
        rank0_print(f"[EVAL] Starting {prefix} (wm): {len(indices)}/{len(dataset)} samples")
        for idx in indices:
            # Load one sample; samples that fail to load are skipped silently.
            try:
                sample = dataset[idx]
            except Exception:
                continue
            # Collate a batch of one (the collator only supports batch size 1),
            # move tensors to the device and drop the non-tensor metadata dict,
            # which the forward does not need here.
            batch = collator([sample])
            batch = self._move_to_device(batch)
            batch.pop("wm_meta", None)
            # Forward with IK trajectory sampling and attention diagnostics; a
            # failing sample is reported and skipped.
            try:
                batch["return_ik_pred_for_debug"] = True
                batch["return_attn_debug"] = True
                outputs = model(**batch)
            except Exception as e:
                rank0_print(f"[EVAL] forward failed idx={idx}: {type(e).__name__}: {e}")
                continue
            # Next-visual-state loss. loss_count counts every successful forward,
            # so a forward without a bev_loss tensor contributes 0 to the mean.
            _bev = outputs.get("bev_loss", None) if isinstance(outputs, dict) else None
            loss_count += 1
            if _bev is not None and torch.is_tensor(_bev):
                bev_loss_sum += float(_bev)
            # Attention split: accumulate only when all three ratios are present.
            # Per-token values are accumulated together, keyed on the BEV value.
            if isinstance(outputs, dict):
                _a_bev = outputs.get("wm_attn_bev_ratio", None)
                _a_ego = outputs.get("wm_attn_ego_ratio", None)
                _a_text = outputs.get("wm_attn_text_ratio", None)
                _a_bev_pt = outputs.get("wm_attn_bev_per_tok", None)
                _a_ego_pt = outputs.get("wm_attn_ego_per_tok", None)
                _a_text_pt = outputs.get("wm_attn_text_per_tok", None)
                if _a_bev is not None and _a_ego is not None and (_a_text is not None):
                    attn_bev_ratio_sum += float(_a_bev)
                    attn_ego_ratio_sum += float(_a_ego)
                    attn_text_ratio_sum += float(_a_text)
                    if _a_bev_pt is not None:
                        attn_bev_per_tok_sum += float(_a_bev_pt)
                        attn_ego_per_tok_sum += float(_a_ego_pt)
                        attn_text_per_tok_sum += float(_a_text_pt)
                    attn_count += 1
            # Trajectory errors from the IK head's sampled trajectory.
            if has_ik:
                # (B, N_wp, 2) predicted / GT trajectories in meters (None when the
                # forward skipped the IK branch, e.g. no GT trajectory).
                pred_traj = (
                    outputs.get("wm_ik_pred_traj", None) if isinstance(outputs, dict) else None
                )
                gt_traj_out = (
                    outputs.get("wm_ik_gt_traj", None) if isinstance(outputs, dict) else None
                )
                if torch.is_tensor(pred_traj) and torch.is_tensor(gt_traj_out):
                    p = pred_traj.float().cpu()
                    g = gt_traj_out.float().cpu()
                    # Score only well-formed 3-D outputs whose waypoint count
                    # matches ik_num_waypoints.
                    if p.ndim == 3 and g.ndim == 3 and (p.shape[1] == _eval_nw):
                        _nw_eff = min(p.shape[1], g.shape[1])
                        # Point-wise Euclidean error per waypoint: (B, _nw_eff).
                        l2 = torch.norm(p[:, :_nw_eff] - g[:, :_nw_eff], dim=-1)
                        # Accumulate the error of each waypoint index separately,
                        # summed over the batch rows.
                        for wi in range(_nw_eff):
                            wp_l2_sums[wi] += float(l2[:, wi].sum())
                        traj_count += int(p.shape[0])
        metrics = {}
        # Mean next-visual-state loss over all successful forwards.
        if loss_count > 0:
            metrics[f"{prefix}/bev_loss"] = round(bev_loss_sum / loss_count, 6)
        if traj_count > 0:
            # ADE: mean point-wise L2 over all waypoints of all scored trajectories.
            ade = sum(wp_l2_sums) / (_eval_nw * traj_count)
            metrics[f"{prefix}/ade"] = round(ade, 4)
            # l2_{s}s: L2 of the single waypoint at s seconds. With 0.5 s spacing
            # that is waypoint index 2s-1 (1 s -> index 1, 2 s -> 3, 3 s -> 5,
            # 4 s -> 7). UniAD-style, not ST-P3's cumulative average.
            for s in range(1, _eval_nw // 2 + 1):
                wi = 2 * s - 1
                if wi < _eval_nw:
                    metrics[f"{prefix}/l2_{s}s"] = round(wp_l2_sums[wi] / traj_count, 4)
        # Mean attention split (percent) and per-token attention mass.
        if attn_count > 0:
            avg_bev = attn_bev_ratio_sum / attn_count
            avg_ego = attn_ego_ratio_sum / attn_count
            avg_text = attn_text_ratio_sum / attn_count
            metrics[f"{prefix}/attn_bev_pct"] = round(avg_bev, 2)
            metrics[f"{prefix}/attn_ego_pct"] = round(avg_ego, 2)
            metrics[f"{prefix}/attn_text_pct"] = round(avg_text, 2)
            avg_bev_pt = attn_bev_per_tok_sum / attn_count
            avg_ego_pt = attn_ego_per_tok_sum / attn_count
            avg_text_pt = attn_text_per_tok_sum / attn_count
            # Per-token values are reported only if any of them is non-zero.
            if avg_bev_pt > 0 or avg_ego_pt > 0 or avg_text_pt > 0:
                metrics[f"{prefix}/attn_bev_per_tok"] = round(avg_bev_pt, 8)
                metrics[f"{prefix}/attn_ego_per_tok"] = round(avg_ego_pt, 8)
                metrics[f"{prefix}/attn_text_per_tok"] = round(avg_text_pt, 8)
        # Wall-clock time and number of scored trajectories. scripts/
        # evaluate_checkpoint.py aborts when "*wm/num_samples" is 0.
        elapsed = _time.time() - t0
        metrics[f"{prefix}/runtime"] = round(elapsed, 1)
        metrics[f"{prefix}/num_samples"] = traj_count
        # Console summary (metrics dict, then the attention split in words).
        rank0_print(f"[EVAL] {prefix}: {metrics}")
        if attn_count > 0:
            rank0_print(
                f"[EVAL] {prefix} attention distribution ({attn_count} samples): BEV={avg_bev:.1f}% | EGO={avg_ego:.1f}% | TEXT={avg_text:.1f}%"
            )
            if avg_bev_pt > 0 or avg_ego_pt > 0 or avg_text_pt > 0:
                rank0_print(
                    f"[EVAL] {prefix} attn per-token ({attn_count} samples): BEV/tok={avg_bev_pt:.6f} | EGO/tok={avg_ego_pt:.6f} | TEXT/tok={avg_text_pt:.6f}"
                )
        return metrics

    def _move_to_device(self, batch):
        """Move tensor values in a batch dict to model device.

        Args:
            batch: Collated batch. Non-dict inputs are returned unchanged.

        Returns:
            A new dict where every tensor is moved to ``self.args.device``; nested
            dicts are processed recursively, except ``wm_meta`` (pure metadata),
            and all other values (lists, strings, None, ...) are copied as is.
        """
        if not isinstance(batch, dict):
            return batch
        device = self.args.device
        out = {}
        for k, v in batch.items():
            # Tensors -> device.
            if torch.is_tensor(v):
                out[k] = v.to(device)
            # Nested dicts (other than wm_meta) -> recurse.
            elif isinstance(v, dict) and k != "wm_meta":
                out[k] = self._move_to_device(v)
            # Everything else (lists of paths, strings, metadata) unchanged.
            else:
                out[k] = v
        return out


class CurriculumEpochSampler(torch.utils.data.Sampler):
    """Seeded, epoch- and stage-dependent shuffle sampler that is safe for DDP.

    * Stage is read from ``model.curriculum_stage`` (1 or 2); it only changes the
      RNG seed (both stages draw from every dataset index).
    * Per-epoch length is fixed to ``len(dataset)`` for scheduler stability.

    NOTE: This intentionally does NOT inherit DistributedSampler.
    HF Accelerate detects DistributedSampler subclasses and skips its own
    BatchSamplerShard, which leads to all GPUs seeing identical data when
    num_replicas=1.  By inheriting plain Sampler, Accelerate applies
    BatchSamplerShard normally → each GPU gets a unique shard.

    ``train.main`` installs this sampler (via ``trainer._get_train_sampler``),
    and ``TwoStageCurriculumCallback`` calls ``set_epoch`` at every epoch start.
    """

    def __init__(
        self,
        dataset,
        *,
        model,
        seed: int = 42,
        shuffle: bool = True,
    ):
        """Store the dataset, model and seed.

        Args:
            dataset: Training dataset.
            model: The (unwrapped) model; its ``curriculum_stage`` attribute is
                read at iteration time.
            seed: Base seed of the shuffling generator.
            shuffle: Shuffle within each pass over the dataset.
        """
        # torch >= 2.10 removed Sampler.__init__(data_source); the argument was ignored before.
        super().__init__()
        self.dataset = dataset
        self.model = model
        self.seed = int(seed)
        self.epoch = 0
        self.shuffle = bool(shuffle)
        self._pool_all: List[int] = list(range(len(dataset)))
        rank0_print(f"[SAMPLER] CurriculumEpochSampler | pool={len(self._pool_all)}")

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch used to seed the next ``__iter__`` (different order per epoch)."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        """Always ``len(dataset)``, so the number of steps per epoch never depends on the stage."""
        return len(self.dataset)

    def __iter__(self):
        """Yield ``len(dataset)`` indices.

        The generator is seeded (when iteration starts, i.e. at the beginning of
        each epoch's DataLoader pass) with
        ``seed + 1000 * epoch + 17 * stage`` so the order is reproducible and
        differs between epochs and stages. The pool is concatenated (reshuffled
        for each repetition when ``shuffle``) until ``len(dataset)`` indices are
        collected, then truncated.
        """
        pool = self._pool_all
        target_len = len(self.dataset)
        # Deterministic, epoch- and stage-dependent RNG.
        g = torch.Generator()
        g.manual_seed(
            int(self.seed)
            + 1000 * int(self.epoch)
            + 17 * int(getattr(self.model, "curriculum_stage", 2) or 2)
        )
        indices: List[int] = []
        # No shuffling (or a trivial pool): repeat the pool in order.
        if not self.shuffle or len(pool) <= 1:
            while len(indices) < target_len:
                indices.extend(pool)
            indices = indices[:target_len]
        # Shuffling: concatenate fresh permutations of the pool.
        else:
            while len(indices) < target_len:
                perm = torch.randperm(len(pool), generator=g).tolist()
                indices.extend([pool[i] for i in perm])
            indices = indices[:target_len]
        return iter(indices)


class TwoStageCurriculumCallback(TrainerCallback):
    """Curriculum switch into stage 2 (full gradient flow).

    ``train.main`` sets the model up in stage 1 (IK loss weight 0; this is also the
    state checkpoint evaluation runs with, since no callback runs there). Stage 1
    has a length of zero during training: this callback switches the model to
    stage 2 (IK loss with full gradient flow) at train begin, also when resuming,
    and it stays there.

    We do NOT gate the Dataset/prompt by stage (to avoid DataLoader worker
    desynchronization). Only the model's ``curriculum_stage`` / ``ik_loss_weight`` /
    ``ik_stop_grad`` are changed (``curriculum_stage`` also enters the seed of the
    attached ``CurriculumEpochSampler``). Because the callback writes
    ``ik_stop_grad = False``, it overrides the value that ``train.main``
    initialised from ``--ik-stop-grad``. The callback also forwards the epoch number
    to the sampler at every epoch start.
    """

    def __init__(
        self,
        *,
        model: LlavaQwenForCausalLM,
        ik_loss_weight: float = 1.0,
    ):
        """Store the model and the IK loss weight.

        Args:
            model: The unwrapped model whose ``curriculum_stage``,
                ``ik_loss_weight`` and ``ik_stop_grad`` attributes are written.
            ik_loss_weight: IK loss weight written to the model (``--ik-loss-weight``).
        """
        super().__init__()
        self.model = model
        self.ik_loss_weight = float(ik_loss_weight)
        # Optional CurriculumEpochSampler (set by train.main) that receives
        # set_epoch() calls.
        self._sampler = None
        # Current stage (None until on_train_begin).
        self._stage: Optional[int] = None

    @staticmethod
    def _safe_float(x) -> Optional[float]:
        """``float(x)``, or ``None`` if ``x`` is ``None`` or not convertible."""
        try:
            if x is None:
                return None
            return float(x)
        except Exception:
            return None

    def _apply_stage2(self, *, step: int, epoch: Optional[float]) -> None:
        """Write the stage-2 settings onto the model (no-op once applied).

        Sets ``model.curriculum_stage = 2``, ``model.ik_loss_weight`` and
        ``model.ik_stop_grad = False``, and prints a ``[CURRICULUM]`` line on rank 0.
        """
        if self._stage == 2:
            return
        self._stage = 2
        # Stage flag read by the model forward (reported as wm_stage) and by
        # CurriculumEpochSampler.
        try:
            self.model.curriculum_stage = 2
        except Exception:
            pass
        # IK loss with gradients flowing into the LLM.
        self.model.ik_loss_weight = float(self.ik_loss_weight)
        self.model.ik_stop_grad = False
        rank0_print(
            f"[CURRICULUM] step={int(step)} epoch={(epoch if epoch is not None else 'NA')} -> stage=2 | ik_loss_weight={float(getattr(self.model, 'ik_loss_weight', 0.0))} | ik_stop_grad={bool(getattr(self.model, 'ik_stop_grad', False))}"
        )

    def on_train_begin(self, args, state, control, **kwargs):
        """Switch the model to stage 2 (also when resuming)."""
        step = int(getattr(state, "global_step", 0) or 0)
        epoch_f = self._safe_float(getattr(state, "epoch", None))
        self._apply_stage2(step=step, epoch=epoch_f)
        return control

    def on_epoch_begin(self, args, state, control, **kwargs):
        """Forward the epoch number to the epoch sampler.

        Errors (e.g. ``state.epoch`` being ``None``) are swallowed.
        """
        try:
            epoch_f = self._safe_float(getattr(state, "epoch", None))
            if self._sampler is not None:
                self._sampler.set_epoch(int(epoch_f))
        except Exception:
            pass
        return control
