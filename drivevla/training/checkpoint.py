"""Checkpoint loading, optimizer groups and resume compatibility.

Two groups of helpers live here:

1. Optimizer construction / resume, installed as Trainer method overrides by
   ``drivevla/training/trainer.py`` (``LatentActionWMTrainer`` and
   ``LatentActionDebugTrainer``):

   * :func:`_create_optimizer_with_ik_lr` (``Trainer.create_optimizer``) builds
     up to four optimizer parameter groups -- IK modules vs. the rest, each
     split into weight-decay / no-decay -- so the randomly initialised IK
     modules can use ``--ik-learning-rate``. When resuming, it rebuilds the
     group layout that the checkpoint's ``optimizer.pt`` was saved with; if no
     layout matches, it remaps the saved per-parameter state onto the new
     layout and loads it itself.
   * :func:`_load_optimizer_and_scheduler_with_remap`
     (``Trainer._load_optimizer_and_scheduler``) then skips the optimizer
     reload if that remap already happened, and otherwise loads
     ``optimizer.pt`` through :func:`_filter_optimizer_state_shapes`.

   Shape tolerance exists mainly for the NAVSIM ego projector: train.py
   widens the first Linear of ``ego_projector`` by one zero-initialised input
   column per launch for the step-conditioning channel (raw can_bus width 18 -> 19 in
   NAVSIM SFT, and 19 -> 20 when NAVSIM RFT re-reads the 19-wide SFT projector
   with ``--ego-can-bus-dim 19``). The Adam moments ``exp_avg`` /
   ``exp_avg_sq`` saved for the old ``(out, in)`` weight are then one column
   short; they are zero-padded on the last dimension to the new width. Any
   other shape mismatch drops that parameter's state so it restarts fresh.

2. Restoring dynamically attached modules from a checkpoint directory.
   ``LlavaQwenForCausalLM.from_pretrained`` only fills modules that exist when
   the class is instantiated; train.py attaches the IK head / pools and the
   ``mm_projector_*`` modules afterwards, so their tensors are read back
   explicitly (filtered by key prefix and shape) by
   :func:`_restore_ik_head_weights_from_ckpt` and
   :func:`_restore_projector_weights_from_ckpt`, with the directory found by
   :func:`_resolve_ckpt_dir_for_weight_loading`.

Messages go through ``llava.utils.rank0_print``.
"""

from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import Trainer
from llava.model import LlavaQwenForCausalLM
from llava.utils import rank0_print
from . import common as wm


def _create_optimizer_with_ik_lr(trainer: Trainer) -> torch.optim.Optimizer:
    """Create an optimizer with separate LR for IK modules.

    (The optimizer class itself comes from ``TrainingArguments``; AdamW by
    default.)

    * Base / LLM params         : trainer.args.learning_rate
    * IK params (ik_head/pool)  : trainer.args.ik_learning_rate (if set) else base LR

    This is useful because IK heads are randomly initialized and typically need a
    much larger LR (e.g., 1e-4 ~ 1e-3) compared to the LLM fine-tuning LR (e.g., 2e-5).

    Used as ``create_optimizer`` by the Trainer subclasses in ``trainer.py``.

    "IK params" are all parameters of the modules ``ik_head`` and
    ``ik_attn_pool`` of the (unwrapped) model; everything else that requires
    grad (LLM, projectors, ``object_query_head``, ...) uses the base LR. Each side is split into a weight-decay group
    (``args.weight_decay``) and a no-decay group (0.0); empty groups are
    omitted, so there are 1-4 groups, always in the order
    IK-decay, IK-no-decay, base-decay, base-no-decay.

    Resume handling (``trainer._resume_ckpt``, set by train.py, or
    ``trainer._resume_from_checkpoint``): PyTorch optimizer state is keyed by
    the position of each parameter in the flattened group list, so the groups
    must be rebuilt exactly as they were saved. The group sizes stored in the
    checkpoint's ``optimizer.pt`` are compared with the layouts produced by
    several decay-name policies (see ``full_cand_policies``):

    * a layout with identical sizes is used as-is, and the Trainer's
      (overridden) ``_load_optimizer_and_scheduler`` step later loads the
      state;
    * otherwise the state is remapped by parameter name onto the new layout,
      zero-padding Adam moments whose last dimension grew (e.g. the widened
      ``ego_projector`` input), loaded here, and
      ``trainer._opt_state_already_loaded`` is set so that
      :func:`_load_optimizer_and_scheduler_with_remap` does not load
      ``optimizer.pt`` a second time.

    Args:
        trainer: The ``transformers.Trainer`` being set up. Uses
            ``trainer.model`` (possibly DDP-wrapped), ``trainer.args``
            (``learning_rate``, ``ik_learning_rate``, ``weight_decay``, optimizer settings) and optional resume
            attributes.

    Returns:
        ``trainer.optimizer`` (created here unless it already existed).

    Side effects:
        Sets ``trainer.optimizer``; replaces its ``load_state_dict`` with a
        shape-checking wrapper; on the remap path may set/delete
        ``trainer._remapped_optimizer_state`` and set
        ``trainer._opt_state_already_loaded``. Logs the chosen group layout.

    Raises:
        ValueError: if no trainable parameter ends up in any group.
    """
    # HF may call create_optimizer more than once; keep an existing optimizer.
    if getattr(trainer, "optimizer", None) is not None:
        return trainer.optimizer
    # opt_model is what the optimizer sees (its parameter names may carry a
    # DDP "module." prefix); raw_model is the unwrapped model that owns the
    # IK sub-modules as attributes.
    opt_model = trainer.model
    raw_model = opt_model.module if hasattr(opt_model, "module") else opt_model
    # Collect the identities (id()) of every parameter belonging to an IK-side
    # module; id() is used because parameter names differ between the
    # wrapped and unwrapped model.
    ik_param_ids: set[int] = set()
    for attr in ("ik_head", "ik_attn_pool"):
        m = getattr(raw_model, attr, None)
        if m is None:
            continue
        for p in m.parameters(recurse=True):
            ik_param_ids.add(id(p))
    # Resolve the IK learning rate: --ik-learning-rate (copied onto the
    # TrainingArguments by train.py), else the base LR.
    ik_lr = getattr(trainer.args, "ik_learning_rate", None)
    if ik_lr is None:
        ik_lr = trainer.args.learning_rate
    ik_lr = float(ik_lr)
    base_lr = float(trainer.args.learning_rate)
    # Optimizer class and its kwargs (betas, eps, ...) as HF would choose them
    # from TrainingArguments (``optim``; AdamW by default).
    optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(trainer.args)

    def _strip_module_prefix(n: str) -> str:
        """Drop a leading DDP ``"module."`` (7 chars) from a parameter name."""
        return n[7:] if isinstance(n, str) and n.startswith("module.") else n

    # --- Read the group layout saved in the resume checkpoint, if any -------
    # expected_group_sizes = number of parameters in each saved param group
    # (None when not resuming or when optimizer.pt is missing / unreadable).
    # Note: the whole optimizer.pt is loaded to CPU just to read these sizes.
    resume_ckpt = getattr(trainer, "_resume_ckpt", None) or getattr(
        trainer, "_resume_from_checkpoint", None
    )
    expected_group_sizes = None
    if resume_ckpt:
        opt_pt = os.path.join(str(resume_ckpt), "optimizer.pt")
        if os.path.isfile(opt_pt):
            try:
                sd0 = torch.load(opt_pt, map_location="cpu")
                if isinstance(sd0, dict) and isinstance(sd0.get("param_groups"), list):
                    expected_group_sizes = [len(g.get("params", [])) for g in sd0["param_groups"]]
            except Exception:
                expected_group_sizes = None
    from transformers.trainer_pt_utils import get_parameter_names

    # --- Weight-decay name policies -----------------------------------------
    # Different Transformers versions decide differently which parameters get
    # weight decay, so the checkpoint may have been saved with either rule.
    # Both return names without the "module." prefix.

    def _decay_names_legacy(model: torch.nn.Module) -> set[str]:
        """Old HF rule: decay every parameter outside LayerNorm modules whose
        name does not contain "bias"."""
        dn = set(get_parameter_names(model, [torch.nn.LayerNorm]))
        dn = {n for n in dn if "bias" not in n}
        return {_strip_module_prefix(n) for n in dn}

    def _decay_names_hf(model: torch.nn.Module) -> Optional[set[str]]:
        """Current HF rule via ``Trainer.get_decay_parameter_names`` (None if
        this Transformers version has no such method or it fails)."""
        if hasattr(trainer, "get_decay_parameter_names"):
            try:
                dn = set(trainer.get_decay_parameter_names(model))
                return {_strip_module_prefix(n) for n in dn}
            except Exception:
                return None
        return None

    # Candidate policies in priority order: the HF rule first (when
    # available), then the legacy rule. cand_policies[0] is the default for
    # fresh runs and for the remap path.
    cand_policies: List[Tuple[str, set[str]]] = []
    hf_dn = _decay_names_hf(opt_model)
    if hf_dn is not None:
        cand_policies.append(("hf_get_decay_parameter_names", hf_dn))
    cand_policies.append(("legacy_layernorm_only", _decay_names_legacy(opt_model)))
    weight_decay = float(getattr(trainer.args, "weight_decay", 0.0) or 0.0)

    def _build_groups(
        decay_parameter_names: set[str], *, exclude_heads: bool = True
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Build the optimizer param groups for one decay policy.

        Args:
            decay_parameter_names: Names (without "module.") that get weight
                decay.
            exclude_heads: If True, skip parameters whose name contains
                ``"traj_head."`` or ``"point_head."``; the ``*_with_heads``
                candidates keep them, to reproduce older layouts.

        Returns:
            ``(groups, sizes)``: list of ``{"params", "weight_decay", "lr"}``
            dicts in the order IK-decay, IK-no-decay, base-decay,
            base-no-decay (empty ones omitted), and the number of params in
            each group.
        """

        def _pick_params(*, want_ik: bool, want_decay: bool) -> List[torch.nn.Parameter]:
            """Trainable params (in named_parameters order) on the requested
            IK / decay side."""
            params: List[torch.nn.Parameter] = []
            for n, p in opt_model.named_parameters():
                # Frozen parameters never enter the optimizer.
                if not p.requires_grad:
                    continue
                # Optionally leave out legacy trajectory / point heads.
                if exclude_heads and ("traj_head." in n or "point_head." in n):
                    continue
                # IK side vs. base side.
                is_ik = id(p) in ik_param_ids
                if is_ik != want_ik:
                    continue
                # Decay vs. no-decay, by prefix-stripped name.
                n2 = _strip_module_prefix(n)
                is_decay = n2 in decay_parameter_names
                if is_decay != want_decay:
                    continue
                params.append(p)
            return params

        # IK groups first (IK LR), then base groups (base LR); no-decay groups
        # use weight_decay=0.0.
        groups: List[Dict[str, Any]] = []
        ik_decay = _pick_params(want_ik=True, want_decay=True)
        ik_nodecay = _pick_params(want_ik=True, want_decay=False)
        if len(ik_decay) > 0:
            groups.append({"params": ik_decay, "weight_decay": weight_decay, "lr": ik_lr})
        if len(ik_nodecay) > 0:
            groups.append({"params": ik_nodecay, "weight_decay": 0.0, "lr": ik_lr})
        base_decay = _pick_params(want_ik=False, want_decay=True)
        base_nodecay = _pick_params(want_ik=False, want_decay=False)
        if len(base_decay) > 0:
            groups.append({"params": base_decay, "weight_decay": weight_decay, "lr": base_lr})
        if len(base_nodecay) > 0:
            groups.append({"params": base_nodecay, "weight_decay": 0.0, "lr": base_lr})
        return (groups, [len(g["params"]) for g in groups])

    def _build_groups_with_names(
        decay_parameter_names: set[str], *, exclude_heads: bool = True
    ) -> Tuple[List[List[str]], List[int]]:
        """Like _build_groups but returns param names per group instead of param tensors.

        The names are the raw ``named_parameters()`` names (DDP prefix kept),
        in the same group order and per-group order as :func:`_build_groups`,
        so flattening them gives the optimizer's state index of each name.

        Returns:
            ``(names_per_group, sizes)``.
        """

        def _pick_names(*, want_ik: bool, want_decay: bool) -> List[str]:
            """Same filter as ``_pick_params``, returning names."""
            names: List[str] = []
            for n, p in opt_model.named_parameters():
                if not p.requires_grad:
                    continue
                if exclude_heads and ("traj_head." in n or "point_head." in n):
                    continue
                is_ik = id(p) in ik_param_ids
                if is_ik != want_ik:
                    continue
                n2 = _strip_module_prefix(n)
                is_decay = n2 in decay_parameter_names
                if is_decay != want_decay:
                    continue
                names.append(n)
            return names

        # Same four-group order as _build_groups, empty groups omitted.
        all_group_names: List[List[str]] = []
        for want_ik, want_decay in [(True, True), (True, False), (False, True), (False, False)]:
            ns = _pick_names(want_ik=want_ik, want_decay=want_decay)
            if len(ns) > 0:
                all_group_names.append(ns)
        sizes = [len(g) for g in all_group_names]
        return (all_group_names, sizes)

    # Every decay policy is tried with the legacy heads excluded (the current
    # layout) and included ("<policy>_with_heads").
    full_cand_policies: List[Tuple[str, set[str], bool]] = []
    for pol_name, dn in cand_policies:
        full_cand_policies.append((pol_name, dn, True))
        full_cand_policies.append((pol_name + "_with_heads", dn, False))
    chosen_name = None
    optimizer_grouped_parameters = None
    chosen_sizes = None
    if expected_group_sizes is not None:
        # --- Resume: look for a layout identical to the saved one -----------
        # The first candidate whose group sizes equal the checkpoint's is
        # used; optimizer.pt is then loaded later by the normal resume path.
        for pol_name, dn, excl in full_cand_policies:
            groups, sizes = _build_groups(dn, exclude_heads=excl)
            if sizes == expected_group_sizes:
                chosen_name = pol_name
                optimizer_grouped_parameters = groups
                chosen_sizes = sizes
                break
        if optimizer_grouped_parameters is None:
            # --- Resume, no layout matched: remap state by parameter name ---
            # Log every candidate's sizes for diagnosis.
            diag = []
            for pol_name, dn, excl in full_cand_policies:
                _g, _s = _build_groups(dn, exclude_heads=excl)
                diag.append((pol_name, _s))
            if trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][RESUME_REMAP] checkpoint_group_sizes={expected_group_sizes} | candidates={diag}. Will remap optimizer state by parameter name."
                )
            # Build the new optimizer with the default policy (heads excluded).
            chosen_name, dn = cand_policies[0]
            optimizer_grouped_parameters, chosen_sizes = _build_groups(dn)
            chosen_name = chosen_name + "_remapped"
            # New state index of every parameter name (flattened group order).
            cur_group_names, _ = _build_groups_with_names(dn)
            cur_flat_names: List[str] = []
            for gn in cur_group_names:
                cur_flat_names.extend(gn)
            new_name_to_idx = {n: i for (i, n) in enumerate(cur_flat_names)}
            # Old state index of every name. No candidate layout reproduces the
            # saved group sizes (checked above), so assume the old state
            # indices 0..total_old-1 follow the current named_parameters()
            # order of trainable parameters (ignoring grouping). This is only
            # correct if the old optimizer was flat in that order; mismatched
            # shapes are caught later, same-shape mismatches are not.
            old_name_to_idx: Optional[Dict[str, int]] = None
            total_old = sum(expected_group_sizes)
            all_trainable = [n for (n, p) in opt_model.named_parameters() if p.requires_grad]
            if len(all_trainable) >= total_old:
                old_name_to_idx = {n: i for (i, n) in enumerate(all_trainable[:total_old])}
                if trainer.is_world_process_zero():
                    rank0_print(
                        f"[OPT][RESUME_REMAP] no policy matched checkpoint sizes, using positional fallback (old={total_old}, cur_trainable={len(all_trainable)})"
                    )
            if old_name_to_idx is not None:
                # Re-key the saved per-parameter state from old to new
                # indices; names without saved state are left to lazy-init.
                opt_pt = os.path.join(str(resume_ckpt), "optimizer.pt")
                try:
                    old_sd = torch.load(opt_pt, map_location="cpu")
                    old_state = old_sd.get("state", {})
                    new_state: Dict[int, Any] = {}
                    matched = 0
                    for name, new_idx in new_name_to_idx.items():
                        old_idx = old_name_to_idx.get(name)
                        if old_idx is not None and old_idx in old_state:
                            new_state[new_idx] = old_state[old_idx]
                            matched += 1
                    if trainer.is_world_process_zero():
                        rank0_print(
                            f"[OPT][RESUME_REMAP] remapped {matched}/{len(new_name_to_idx)} param states (old had {len(old_state)} states)"
                        )
                    # Stash for the block below that runs once the optimizer
                    # exists.
                    trainer._remapped_optimizer_state = new_state
                except Exception as e:
                    if trainer.is_world_process_zero():
                        rank0_print(
                            f"[OPT][RESUME_REMAP] failed to load/remap optimizer state: {e}"
                        )
            elif trainer.is_world_process_zero():
                rank0_print(
                    "[OPT][RESUME_REMAP] could not reconstruct old param ordering, optimizer state will be fresh"
                )
    else:
        # --- Fresh run (or no readable optimizer.pt): default policy --------
        chosen_name, dn = cand_policies[0]
        optimizer_grouped_parameters, chosen_sizes = _build_groups(dn)
    if trainer.is_world_process_zero():
        rank0_print(
            f"[OPT][GROUPS] decay_policy={chosen_name} | expected_ckpt_sizes={expected_group_sizes} | built_sizes={chosen_sizes}"
        )
    if len(optimizer_grouped_parameters) == 0:
        raise ValueError("No trainable parameters found when building the optimizer.")
    # Instantiate the optimizer; per-group "lr" / "weight_decay" override the
    # defaults in optimizer_kwargs.
    trainer.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
    _orig_load_sd = trainer.optimizer.load_state_dict

    def _load_state_dict_with_shape_guard(state_dict, *args, **kwargs):
        """Load optimizer state, then clear per-param state with wrong shapes.

        Calls the optimizer's original ``load_state_dict`` and afterwards
        checks every parameter's ``exp_avg`` / ``exp_avg_sq`` (Adam moments)
        against the parameter's shape. Mismatching entries are replaced by an
        empty dict, so the optimizer lazily re-initialises them (including
        ``step``) on the next update instead of failing inside the update.
        Installed on the optimizer instance, so it also covers the HF resume
        path and the remap block below.

        Returns:
            Whatever the original ``load_state_dict`` returns (``None``).
        """
        ret = _orig_load_sd(state_dict, *args, **kwargs)
        try:
            opt = trainer.optimizer
            reset = 0
            for group in opt.param_groups:
                for p in group["params"]:
                    # Parameters without state (not stepped yet) are fine.
                    pst = opt.state.get(p, None)
                    if not isinstance(pst, dict) or len(pst) == 0:
                        continue
                    # Any moment tensor whose shape differs marks the state bad.
                    bad = False
                    for k in ("exp_avg", "exp_avg_sq"):
                        v = pst.get(k, None)
                        if torch.is_tensor(v) and tuple(v.shape) != tuple(p.shape):
                            bad = True
                            break
                    if bad:
                        opt.state[p] = {}
                        reset += 1
            if reset > 0 and trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][SHAPE_GUARD] reset {reset} optimizer state(s) with mismatched per-element tensor shapes — those parameters will lazy-init fresh state on the next step."
                )
        except Exception as _e:
            # The check is best-effort; the load itself already succeeded.
            if trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][SHAPE_GUARD] post-load validation skipped: {type(_e).__name__}: {_e}"
                )
        return ret

    trainer.optimizer.load_state_dict = _load_state_dict_with_shape_guard
    # --- Apply the remapped state (remap path only) -------------------------
    remapped_state = getattr(trainer, "_remapped_optimizer_state", None)
    if remapped_state is not None:
        try:
            # Start from the new optimizer's own param_groups (new layout and
            # hyper-parameters) and fill "state" with the remapped entries.
            cur_sd = trainer.optimizer.state_dict()
            new_sd = {"state": {}, "param_groups": cur_sd["param_groups"]}
            cur_flat_params: List[torch.nn.Parameter] = []
            for pg in trainer.optimizer.param_groups:
                cur_flat_params.extend(pg["params"])
            adapted = 0
            dropped = 0
            for new_idx, st in remapped_state.items():
                if 0 <= int(new_idx) < len(cur_flat_params):
                    cur_p = cur_flat_params[int(new_idx)]
                    cur_shape = tuple(cur_p.shape)
                    new_st = dict(st)
                    drop_this = False
                    for k in ("exp_avg", "exp_avg_sq"):
                        v = new_st.get(k, None)
                        if not torch.is_tensor(v):
                            continue
                        v_shape = tuple(v.shape)
                        if v_shape == cur_shape:
                            continue
                        # Widened parameter: same rank, same leading dims and
                        # a smaller (non-zero) last dim -- e.g. the ego
                        # projector weight (out, 18|19) -> (out, 19|20). The
                        # new columns sit at the end (train.py appends the
                        # zero-initialised column last), so append zero
                        # moments there.
                        if (
                            v.dim() >= 1
                            and len(cur_shape) == v.dim()
                            and (v_shape[:-1] == cur_shape[:-1])
                            and (0 < int(v_shape[-1]) < int(cur_shape[-1]))
                        ):
                            pad_cols = int(cur_shape[-1]) - int(v_shape[-1])
                            pad_shape = list(v.shape)
                            pad_shape[-1] = pad_cols
                            pad = v.new_zeros(*pad_shape)
                            new_st[k] = torch.cat([v, pad], dim=-1)
                            adapted += 1
                        else:
                            # Any other mismatch: drop this parameter's state.
                            drop_this = True
                            break
                    if drop_this:
                        dropped += 1
                        continue
                    st = new_st
                # In-range entries now hold the (possibly padded) copy;
                # out-of-range indices are copied through unchanged.
                new_sd["state"][new_idx] = st
            if adapted > 0 and trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][RESUME_REMAP] zero-padded {adapted} optimizer tensor(s) to match resized parameters (e.g. ego_projector +1 column)."
                )
            if dropped > 0 and trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][RESUME_REMAP] dropped {dropped} optimizer state(s) with irreconcilable shape mismatch — those params start with fresh state."
                )
            # Goes through the shape guard installed above.
            trainer.optimizer.load_state_dict(new_sd)
            if trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][RESUME_REMAP] successfully loaded remapped optimizer state ({len(remapped_state)} params)"
                )
            # Tell _load_optimizer_and_scheduler_with_remap not to load
            # optimizer.pt again (it will still restore scheduler.pt).
            trainer._opt_state_already_loaded = True
        except Exception as e:
            if trainer.is_world_process_zero():
                rank0_print(
                    f"[OPT][RESUME_REMAP] failed to apply remapped state: {e}. Continuing with fresh optimizer."
                )
        finally:
            # The stash is consumed exactly once.
            del trainer._remapped_optimizer_state
    # --- Summary log (best-effort) ------------------------------------------
    try:
        if trainer.is_world_process_zero():
            n_ik = sum(
                (
                    p.numel()
                    for p in raw_model.parameters()
                    if p.requires_grad and id(p) in ik_param_ids
                )
            )
            n_all = sum((p.numel() for p in opt_model.parameters() if p.requires_grad))
            rank0_print(
                f"[OPT] base_lr={base_lr:.2e} ik_lr={ik_lr:.2e} | trainable_params={n_all:,} | ik_params={n_ik:,} | groups={len(trainer.optimizer.param_groups)}"
            )
    except Exception:
        pass
    return trainer.optimizer


def _filter_optimizer_state_shapes(optimizer, saved_state: dict) -> tuple[dict, list[int]]:
    """Remove optimizer state entries whose shape doesn't match the current parameter.

    Returns (filtered_state_dict, list_of_skipped_indices).

    Entries whose last dimension merely grew (same rank and leading dims,
    smaller non-zero last dim -- the widened ``ego_projector`` input) are kept
    with ``exp_avg`` / ``exp_avg_sq`` zero-padded at the end of the last
    dimension instead of being removed.

    Args:
        optimizer: The freshly built optimizer; its flattened
            ``param_groups[*]["params"]`` give the current parameters.
        saved_state: A ``torch.optim.Optimizer.state_dict()`` loaded from
            ``optimizer.pt`` (``{"state": {idx: {...}}, "param_groups": [...]}``).

    Returns:
        ``(filtered, skipped)``: a shallow copy of ``saved_state`` whose
        ``"state"`` keeps only shape-compatible (or padded) entries, and the
        state keys that were dropped. ``param_groups`` are passed through
        unchanged.

    Note:
        The i-th *entry* of ``saved_state["state"]`` (in dict order) is
        compared with the i-th current parameter, not the parameter at
        position ``param_id``; the two coincide only when every parameter has
        saved state and the keys are in order.
    """
    # Current parameters in optimizer (flattened group) order.
    all_params = []
    for pg in optimizer.param_groups:
        all_params.extend(pg["params"])
    skipped = []
    filtered_state = {}
    for idx, (param_id, param_state) in enumerate(saved_state["state"].items()):
        # More saved entries than current parameters: drop the extras.
        if idx >= len(all_params):
            skipped.append(param_id)
            continue
        cur_param = all_params[idx]
        # Use the first Adam moment (or the second, if only that exists) as
        # the shape reference.
        ref = param_state.get("exp_avg", param_state.get("exp_avg_sq", None))
        if ref is not None and tuple(ref.shape) != tuple(cur_param.shape):
            cur_shape = tuple(cur_param.shape)
            v_shape = tuple(ref.shape)
            # Paddable = same rank, same leading dims, last dim grew.
            can_pad = (
                ref.dim() >= 1
                and len(cur_shape) == ref.dim()
                and (v_shape[:-1] == cur_shape[:-1])
                and (0 < int(v_shape[-1]) < int(cur_shape[-1]))
            )
            if can_pad:
                # Zero-pad both moments on the last dim to the new width.
                new_st = dict(param_state)
                pad_cols = int(cur_shape[-1]) - int(v_shape[-1])
                for k in ("exp_avg", "exp_avg_sq"):
                    v = new_st.get(k, None)
                    if not torch.is_tensor(v):
                        continue
                    pad_shape = list(v.shape)
                    pad_shape[-1] = pad_cols
                    pad = v.new_zeros(*pad_shape)
                    new_st[k] = torch.cat([v, pad], dim=-1)
                filtered_state[param_id] = new_st
            else:
                # Irreconcilable: drop, the parameter will start fresh.
                skipped.append(param_id)
        else:
            # Shapes match (or no moment tensors): keep as-is.
            filtered_state[param_id] = param_state
    # Same dict layout as the input, with the filtered state.
    result = dict(saved_state)
    result["state"] = filtered_state
    return (result, skipped)


def _load_optimizer_and_scheduler_with_remap(trainer_self, checkpoint):
    """Override for _load_optimizer_and_scheduler that skips optimizer loading
    when we already loaded a remapped state in create_optimizer.
    Also filters optimizer state entries whose shape no longer matches the
    current model parameters (e.g. after an architecture change).

    Installed as ``_load_optimizer_and_scheduler`` by the Trainer subclasses in
    ``trainer.py``; HF calls it once at the start of training with the resume
    checkpoint directory (or ``None`` for a fresh run).

    Behaviour:

    1. ``trainer_self._opt_state_already_loaded`` set (remap path of
       :func:`_create_optimizer_with_ik_lr`): clear the flag, restore only
       ``scheduler.pt`` and return.
    2. ``checkpoint/optimizer.pt`` exists: load it, pass it through
       :func:`_filter_optimizer_state_shapes`, load it into the optimizer
       (which also runs the shape guard installed at creation), restore
       ``scheduler.pt`` if present, and return.
    3. Otherwise, or if step 2 raised: delegate to the parent class's
       ``_load_optimizer_and_scheduler``.

    Args:
        trainer_self: The Trainer instance.
        checkpoint: Checkpoint directory path, or ``None``.

    Note:
        Step 3 uses ``super(type(trainer_self), trainer_self)``, i.e. it
        resolves relative to the instance's concrete class. That reaches
        ``transformers.Trainer``'s implementation only when the concrete
        class is the one that installs this override; a subclass that does
        not override the method again would recurse into this function
        (the GRPO trainer in ``grpo.py`` overrides it).
    """
    # Case 1: optimizer state already restored by the remap path.
    if getattr(trainer_self, "_opt_state_already_loaded", False):
        del trainer_self._opt_state_already_loaded
        if checkpoint is not None:
            # Restore the LR scheduler (warnings from the scheduler's
            # load_state_dict are recorded and discarded).
            scheduler_path = os.path.join(checkpoint, "scheduler.pt")
            if os.path.isfile(scheduler_path):
                import warnings as _warnings

                with _warnings.catch_warnings(record=True):
                    trainer_self.lr_scheduler.load_state_dict(
                        torch.load(scheduler_path, weights_only=False)
                    )
                if trainer_self.is_world_process_zero():
                    rank0_print(
                        "[OPT][RESUME_REMAP] skipped optimizer.pt reload (already remapped), loaded scheduler.pt"
                    )
        return
    # Case 2: shape-tolerant load of optimizer.pt (+ scheduler.pt).
    if checkpoint is not None:
        opt_path = os.path.join(checkpoint, "optimizer.pt")
        if os.path.isfile(opt_path):
            try:
                saved = torch.load(opt_path, map_location="cpu", weights_only=False)
                filtered, skipped = _filter_optimizer_state_shapes(trainer_self.optimizer, saved)
                if skipped:
                    rank0_print(
                        f"[OPT][RESUME] Skipped {len(skipped)} optimizer state(s) with mismatched shapes (param indices: {skipped}). Those parameters will start with fresh optimizer state."
                    )
                # torch maps saved state onto the current params by position;
                # the shape guard then resets anything still inconsistent.
                trainer_self.optimizer.load_state_dict(filtered)
                scheduler_path = os.path.join(checkpoint, "scheduler.pt")
                if os.path.isfile(scheduler_path) and trainer_self.lr_scheduler is not None:
                    import warnings as _warnings

                    with _warnings.catch_warnings(record=True):
                        trainer_self.lr_scheduler.load_state_dict(
                            torch.load(scheduler_path, map_location="cpu", weights_only=False)
                        )
                return
            except Exception as e:
                rank0_print(
                    f"[OPT][RESUME] Shape-tolerant load failed ({e}), falling back to default."
                )
    # Case 3: no checkpoint / no optimizer.pt / failure above -> HF default.
    return super(type(trainer_self), trainer_self)._load_optimizer_and_scheduler(checkpoint)


def _read_ckpt_ik_head_type(ckpt_dir: str) -> Optional[str]:
    """Read ik_head_type recorded in a checkpoint's ik_mode.json (best-effort).

    train.py compares the returned class name with the type of the newly
    built ``model.ik_head``; on a mismatch the IK head is freshly initialised
    instead of warm-started (see ``skip_ik_head`` in
    :func:`_restore_ik_head_weights_from_ckpt`).

    Args:
        ckpt_dir: Checkpoint directory (may be empty / None).

    Returns:
        The recorded ``ik_head_type`` string (e.g. a head class name), or
        ``None`` if the directory, the file or the field is missing, empty or
        unreadable.
    """
    if not ckpt_dir:
        return None
    p = os.path.join(ckpt_dir, "ik_mode.json")
    if os.path.isfile(p):
        try:
            with open(p) as f:
                # Empty / missing field -> None.
                return str(json.load(f).get("ik_head_type") or "") or None
        except Exception:
            return None
    return None


def _restore_ik_head_weights_from_ckpt(
    model: LlavaQwenForCausalLM, ckpt_dir: str, *, skip_ik_head: bool = False
) -> None:
    """Restore IK-related module weights from a checkpoint.

    HF's from_pretrained can ignore unknown keys if the model class doesn't
    instantiate the modules. We attach `model.ik_head` (and optionally
    `model.ik_attn_pool`) dynamically, so we must explicitly restore their
    weights when resuming.

    skip_ik_head: drop `ik_head.*` keys (keep ik_attn_pool / object_query_head).
        Used for fresh-init when the head TYPE changed (e.g. DDPM→flow), where
        loading DDPM eps weights into a flow head is a poor init.

    Restored key prefixes: ``ik_head.`` (unless ``skip_ik_head``),
    ``ik_attn_pool.`` and ``object_query_head.``. Only tensors whose key exists in the current
    ``model.state_dict()`` with the same shape are loaded
    (``strict=False``); the rest are counted and reported.

    Args:
        model: The model after train.py attached the IK modules.
        ckpt_dir: Local checkpoint directory (from
            :func:`_resolve_ckpt_dir_for_weight_loading`); falsy = no-op.
        skip_ik_head: See above.

    Returns:
        None. Never raises: failures are logged as warnings.
    """
    # Nothing to do without a directory or without IK modules on the model.
    if not ckpt_dir:
        return
    has_ik = hasattr(model, "ik_head") and getattr(model, "ik_head") is not None
    has_pool = hasattr(model, "ik_attn_pool") and getattr(model, "ik_attn_pool") is not None
    if not has_ik and (not has_pool):
        return
    try:
        # Read only the IK-related tensors from the checkpoint files.
        # Precedence: ``A and B or C or ...`` == ``(A and B) or C or ...``,
        # so skip_ik_head only affects the ik_head.* keys.
        sd = wm._load_state_dict_keys_from_checkpoint(
            ckpt_dir,
            keep_fn=lambda k: k.startswith("ik_head.")
            and (not skip_ik_head)
            or k.startswith("ik_attn_pool.")
            or k.startswith("object_query_head."),
        )
        if not sd:
            return
        # Keep tensors whose key exists now with an identical shape; count
        # the ones skipped because the key is missing or the shape differs.
        cur = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped_missing = 0
        skipped_shape = 0
        for k, v in sd.items():
            if k not in cur:
                skipped_missing += 1
                continue
            try:
                if tuple(cur[k].shape) != tuple(v.shape):
                    skipped_shape += 1
                    continue
            except Exception:
                skipped_shape += 1
                continue
            filtered[k] = v
        # Checkpoint had IK tensors but none fit this architecture.
        if not filtered:
            rank0_print(
                f"[LOAD][WARN] Found IK tensors in {ckpt_dir} (keys={len(sd)}) but none matched current model (skipped_missing={skipped_missing} skipped_shape={skipped_shape}). Check --ik-decoder-dim / --ik-decoder-heads / --ik-attn-heads compatibility."
            )
            return
        # Partial load; missing_keys will list every non-IK key of the model,
        # so only the counts are logged.
        incompatible = model.load_state_dict(filtered, strict=False)
        miss = getattr(incompatible, "missing_keys", [])
        unexp = getattr(incompatible, "unexpected_keys", [])
        rank0_print(
            f"[LOAD] Restored IK weights from {ckpt_dir} | loaded={len(filtered)}/{len(sd)} tensors | skipped_missing={skipped_missing} skipped_shape={skipped_shape} | missing={len(miss)} unexpected={len(unexp)}"
        )
    except Exception as e:
        rank0_print(
            f"[WARN] Failed to restore IK head weights from {ckpt_dir}: {type(e).__name__}: {e}"
        )


def _dir_has_hf_weights(p: Path) -> bool:
    """Whether ``p`` is a directory holding Hugging Face model weights.

    Recognises single-file (``model.safetensors`` / ``pytorch_model.bin``) and
    sharded (``*.index.json``) checkpoints.

    Args:
        p: Candidate directory.

    Returns:
        True if any of those four files exists in ``p``.
    """
    if not p.exists() or not p.is_dir():
        return False
    # Single-file checkpoints.
    if (p / "model.safetensors").exists() or (p / "pytorch_model.bin").exists():
        return True
    # Sharded checkpoints.
    if (p / "model.safetensors.index.json").exists() or (
        p / "pytorch_model.bin.index.json"
    ).exists():
        return True
    return False


def _resolve_ckpt_dir_for_weight_loading(model_path: str) -> Optional[str]:
    """Resolve a local dir to load extra module weights from.

    Accepts:
      - a checkpoint dir itself (contains model.safetensors / pytorch_model.bin / sharded index)
      - an output_dir that contains checkpoint-* subdirs (we pick the largest step)

    Returns None for non-local paths (e.g., HF hub ids).

    Args:
        model_path: ``--model-path`` value (local dir or hub id); ``~`` is
            expanded.

    Returns:
        ``model_path`` itself if it directly contains HF weights; otherwise
        the ``checkpoint-<step>`` subdirectory with the largest numeric step
        (or, if no suffix is numeric, the most recently modified one),
        provided that subdirectory contains HF weights; else ``None``.
    """
    # Hub ids / missing paths: nothing local to restore from.
    if not model_path:
        return None
    mp = Path(str(model_path)).expanduser()
    if not mp.exists() or not mp.is_dir():
        return None
    # The path is itself a checkpoint directory.
    if _dir_has_hf_weights(mp):
        return str(mp)
    # Otherwise treat it as an output_dir: collect checkpoint-* subdirs and
    # parse the step number from the suffix (None if not an integer).
    cands = []
    for sub in mp.glob("checkpoint-*"):
        if not sub.is_dir():
            continue
        step = None
        try:
            step = int(str(sub.name).split("checkpoint-", 1)[1])
        except Exception:
            step = None
        cands.append((step, sub))
    if not cands:
        return None
    # Prefer the highest numeric step; fall back to the newest mtime.
    cands_num = [c for c in cands if isinstance(c[0], int)]
    if cands_num:
        sub = sorted(cands_num, key=lambda x: int(x[0]), reverse=True)[0][1]
    else:
        sub = sorted(cands, key=lambda x: x[1].stat().st_mtime, reverse=True)[0][1]
    # Only the chosen subdir is checked; older ones are not tried.
    return str(sub) if _dir_has_hf_weights(sub) else None


def _restore_projector_weights_from_ckpt(model: LlavaQwenForCausalLM, ckpt_dir: str) -> None:
    """Restore dynamically-created mm_projector_{scene,track,map,bev} weights.

    We filter by key+shape to avoid crashes and to avoid re-loading unrelated
    base-model projector keys (e.g., a single `mm_projector` from vanilla LLaVA).

    train.py calls this right after ``ensure_mm_projectors`` has
    created the projectors that map the DINOv3 camera tokens into the LLM
    hidden size. Keys are selected by substring (``"mm_projector_scene"``
    etc. anywhere in the key) and loaded with ``strict=False`` only when the
    key exists in the current model with the same shape.

    Args:
        model: The LLaVA-Qwen model; ``model.get_model()`` holds the projectors.
        ckpt_dir: Local checkpoint directory; falsy = no-op.

    Returns:
        None. Never raises: failures are logged as warnings.
    """
    if not ckpt_dir:
        return
    # Skip if none of the dynamic projectors has been attached as a direct
    # child module of the inner model.
    mm = model.get_model()
    has_any = any(
        (
            name in getattr(mm, "_modules", {})
            for name in [
                "mm_projector_scene",
                "mm_projector_track",
                "mm_projector_map",
                "mm_projector_bev",
            ]
        )
    )
    if not has_any:
        return
    try:
        # Read only projector tensors from the checkpoint files.
        sd = wm._load_state_dict_keys_from_checkpoint(
            ckpt_dir,
            keep_fn=lambda k: "mm_projector_scene" in k
            or "mm_projector_track" in k
            or "mm_projector_map" in k
            or ("mm_projector_bev" in k),
        )
        if not sd:
            return
        # Same key + shape filter as for the IK modules.
        cur = model.state_dict()
        filtered: Dict[str, torch.Tensor] = {}
        skipped_missing = 0
        skipped_shape = 0
        for k, v in sd.items():
            if k not in cur:
                skipped_missing += 1
                continue
            try:
                if tuple(cur[k].shape) != tuple(v.shape):
                    skipped_shape += 1
                    continue
            except Exception:
                skipped_shape += 1
                continue
            filtered[k] = v
        if not filtered:
            rank0_print(
                f"[LOAD][WARN] Found projector tensors in {ckpt_dir} (keys={len(sd)}) but none matched current model (skipped_missing={skipped_missing} skipped_shape={skipped_shape}). This usually means cache_kind/mm_hidden_size mismatch or different projector config."
            )
            return
        # Partial load; only counts of missing / unexpected keys are logged.
        incompatible = model.load_state_dict(filtered, strict=False)
        miss = getattr(incompatible, "missing_keys", [])
        unexp = getattr(incompatible, "unexpected_keys", [])
        rank0_print(
            f"[LOAD] Restored projector weights from {ckpt_dir} | loaded={len(filtered)}/{len(sd)} tensors | skipped_missing={skipped_missing} skipped_shape={skipped_shape} | missing={len(miss)} unexpected={len(unexp)}"
        )
    except Exception as e:
        rank0_print(
            f"[WARN] Failed to restore projector weights from {ckpt_dir}: {type(e).__name__}: {e}"
        )
