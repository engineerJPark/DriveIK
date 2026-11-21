"""NAVSIM diffusion GRPO training with inline PDM scoring.

This module implements the NAVSIM RFT stage (``DriveIK_NAVSIM_rft``, launched by
``train_navsim/DriveIK_NAVSIM_rft.sh`` -> ``scripts/train_bev_latent_action_world_model_grpo.py``
-> :func:`run`). It starts from the NAVSIM SFT checkpoint and keeps the normal
SFT objective (the loss returned by the model's forward: LM, next-visual-state,
IK diffusion and the enabled auxiliary losses), adding a reinforcement
fine-tuning term for the Inverse-Kinematics (IK) diffusion head.

Per training step (``GRPOLatentActionWMTrainer.compute_loss``):

1. SFT forward. The model forward stores, on the IK head, the current visual
   tokens and the LLM's predicted future visual tokens that condition the head
   (``_grpo_cur_tokens`` / ``_grpo_next_tokens_list`` and their ``*_grad``
   twins).
2. Group sampling. The IK head draws G stochastic-DDIM trajectories for the
   same scene (``sample_with_logprob_data``) and records the whole denoising
   chain (x_t, x_{t-1}, sigma_t per step).
3. Rewards. A hand-made proxy reward is computed first
   (:func:`compute_group_rewards`); when the inline PDM scorer is configured
   (the released recipe), each trajectory is instead scored by the NAVSIM PDM
   simulator against the scene's metric cache (``drivevla/training/pdm.py``)
   and the EPDMS-style aggregate *replaces* the proxy. Scenes without a
   metric-cache entry skip the GRPO term for that step.
4. Log-probs. ``compute_logprob_for_chain`` re-evaluates, with gradients, the
   log-likelihood log pi_theta(tau) of each recorded chain under the current IK
   head, and the same chains are evaluated without gradients under a frozen
   reference copy of the SFT model (log pi_ref(tau)). The reference is always
   built; if it cannot be (ZeRO-3 or a setup error) log pi_ref is taken as 0.
5. Loss. The best- and worst-reward samples form one preference pair; the
   loss is ``-logsigmoid(grpo_beta * [(log pi_theta - log pi_ref)(best) -
   (log pi_theta - log pi_ref)(worst)])`` (a DPO-style pairwise loss).
   Groups whose rewards have (near) zero variance contribute no GRPO loss.
6. Total = SFT loss + ``grpo_loss_weight`` * GRPO loss.

Because ``compute_loss`` forces ``ik_stop_grad = True`` before the forward, the
IK-head inputs are detached and the GRPO gradient only reaches the IK head's
parameters.

The module also keeps the proxy-reward helpers (progress, comfort). They are
computed every step but, when PDM scoring is active, only feed
logging/diagnostics (and serve as the fallback when the scorer raises).
"""

from __future__ import annotations
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from llava.utils import rank0_print
from types import SimpleNamespace
from .trainer import LatentActionWMTrainer, LatentActionDebugTrainer
from .train import main as train


def _load_ckpt_state_dict(ckpt: str) -> Dict[str, torch.Tensor]:
    """Load the weights of an HF-saved model checkpoint as a dict of CPU tensors.

    Tries, in order: safetensors (single / sharded) → pytorch_model.bin
    (single / sharded) → a direct single-file path. Used to load the frozen SFT
    weights for the KL / reference anchor independently of the policy
    weights (--model-path).

    Args:
        ckpt: checkpoint directory (HF ``save_pretrained`` layout) or the path
            of a single ``.safetensors`` / torch-pickle weight file.

    Returns:
        ``{parameter_name: tensor}`` with every tensor on CPU.

    Raises:
        FileNotFoundError: if ``ckpt`` does not exist or a directory contains
            none of the recognised weight files.
    """
    import json

    if os.path.isdir(ckpt):
        # 1) Single-file safetensors.
        st = os.path.join(ckpt, "model.safetensors")
        if os.path.isfile(st):
            from safetensors.torch import load_file

            return load_file(st, device="cpu")
        # 2) Sharded safetensors: the index maps each tensor name to a shard;
        #    load every distinct shard once and merge.
        st_idx = os.path.join(ckpt, "model.safetensors.index.json")
        if os.path.isfile(st_idx):
            from safetensors.torch import load_file

            with open(st_idx) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            sd: Dict[str, torch.Tensor] = {}
            for s in shards:
                sd.update(load_file(os.path.join(ckpt, s), device="cpu"))
            return sd
        # 3) Single-file torch pickle.
        bin1 = os.path.join(ckpt, "pytorch_model.bin")
        if os.path.isfile(bin1):
            return torch.load(bin1, map_location="cpu", weights_only=False)
        # 4) Sharded torch pickles, same merge as for safetensors.
        bin_idx = os.path.join(ckpt, "pytorch_model.bin.index.json")
        if os.path.isfile(bin_idx):
            with open(bin_idx) as f:
                shards = sorted(set(json.load(f)["weight_map"].values()))
            sd = {}
            for s in shards:
                sd.update(torch.load(os.path.join(ckpt, s), map_location="cpu", weights_only=False))
            return sd
        raise FileNotFoundError(f"no model weights under dir {ckpt}")
    # A direct file path: pick the loader by extension.
    if os.path.isfile(ckpt):
        if ckpt.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(ckpt, device="cpu")
        return torch.load(ckpt, map_location="cpu", weights_only=False)
    raise FileNotFoundError(f"ckpt path not found: {ckpt}")


# ─────────────────────────────────────────────────────────────────────────────
# Proxy reward components (per trajectory, scalar tensors).
# Both take a trajectory ``traj_m`` of shape (T, 2) in meters. The progress
# term assumes the axis convention stated in its docstring ("y is forward";
# x lateral). When the inline PDM scorer is active these values are
# only used for logging / diagnostics, because the PDM reward replaces them.
# ─────────────────────────────────────────────────────────────────────────────


def _reward_ego_progress(traj_m: torch.Tensor, ep_max: float = 4.0) -> torch.Tensor:
    """clamp(y_final / ep_max, 0, 1). y is the forward axis.

    Args:
        traj_m: (T, 2) waypoints [m]; only the final waypoint's y is used.
        ep_max: distance [m] that counts as full progress (floored at 1e-6).

    Returns:
        0-dim tensor in [0, 1].
    """
    return torch.clamp(traj_m[-1, 1] / max(ep_max, 1e-06), 0.0, 1.0)


def _reward_comfort(traj_m: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """exp(-accel_norm/σ). Acceleration from the second-order finite difference.

    ``accel`` is the per-step second difference of the positions [m per
    step^2], not divided by dt^2. The mean of its norm is mapped to (0, 1].

    Returns:
        0-dim tensor; 1.0 if there are fewer than 3 waypoints.
    """
    # A second difference needs at least three points.
    if traj_m.shape[0] < 3:
        return torch.ones((), device=traj_m.device, dtype=traj_m.dtype)
    accel = traj_m[2:] - 2 * traj_m[1:-1] + traj_m[:-2]
    return torch.exp(-accel.norm(dim=-1).mean() / max(sigma, 1e-06))


@torch.no_grad()
def compute_sample_reward(
    traj_m: torch.Tensor,
    cfg: "GRPOConfig",
    return_components: bool = False,
):
    """Combined proxy reward of a single trajectory.

    R = w_ep·EP + w_comfort·comfort.

    Args:
        traj_m: (T, 2) sampled trajectory [m].
        cfg: GRPOConfig with the weights / scales.
        return_components: also return the unweighted components.

    Returns:
        If return_components=True: (R_total, components_dict), where
        components_dict[name] = scalar tensor (the raw component value *before*
        weighting) for the names ep / comfort.
        Otherwise only R_total (0-dim tensor).
    """
    dev, dtype = (traj_m.device, traj_m.dtype)
    r = torch.zeros((), device=dev, dtype=dtype)
    comp: Dict[str, torch.Tensor] = {}
    # Progress along the (assumed) forward y axis.
    v = _reward_ego_progress(traj_m, ep_max=cfg.ep_max)
    r = r + cfg.w_ep * v
    comp["ep"] = v
    # Smoothness (small second differences).
    v = _reward_comfort(traj_m, sigma=cfg.sigma_comfort)
    r = r + cfg.w_comfort * v
    comp["comfort"] = v
    if return_components:
        return (r, comp)
    return r


@torch.no_grad()
def compute_group_rewards(
    samples: List[torch.Tensor],
    traj_scale: float,
    cfg: "GRPOConfig",
    num_waypoints: int = 6,
    return_breakdown: bool = False,
) -> torch.Tensor:
    """Proxy reward vector for G samples. shape: (G,).

    Args:
        samples: list of G tensors, each (1, traj_dim) with traj_dim =
            2 * num_waypoints, in the IK head's normalized units
            (meters / traj_scale).
        traj_scale: meters per normalized unit (``ik_traj_scale``).
        cfg: forwarded to ``compute_sample_reward``.
        num_waypoints: T, number of (x, y) waypoints per sample.
        return_breakdown: also return the per-component values.

    Returns:
        rewards: (G,) tensor on the samples' device/dtype (empty CPU tensor
        when ``samples`` is empty).
        If return_breakdown=True, returns (rewards, breakdown_dict) instead,
        where breakdown_dict[component_name] = (G,) tensor of raw reward values
        before weighting, for the names l2 / ep / comfort. The recipe has no
        L2-to-GT term, so "l2" stays 0; the key is kept so the
        ``grpo/diag/r_l2_*`` log keys keep existing.
    """
    # Empty group: nothing to score.
    if not samples:
        if return_breakdown:
            return (torch.zeros(0), {})
        return torch.zeros(0)
    device = samples[0].device
    dtype = samples[0].dtype
    G = len(samples)
    # Pre-allocate the breakdown so every component key exists (zeros).
    breakdown: Dict[str, torch.Tensor] = {}
    if return_breakdown:
        for k in ("l2", "ep", "comfort"):
            breakdown[k] = torch.zeros(G, device=device, dtype=dtype)
    # Score each sample independently after de-normalizing it to meters.
    rewards = torch.zeros(G, device=device, dtype=dtype)
    for g, s in enumerate(samples):
        traj = s.reshape(num_waypoints, 2) * traj_scale
        if return_breakdown:
            # Diagnostic path: also record each raw component for sample g.
            r, comp = compute_sample_reward(traj, cfg, return_components=True)
            rewards[g] = r
            for k, v in comp.items():
                breakdown[k][g] = v
        else:
            rewards[g] = compute_sample_reward(traj, cfg)
    if return_breakdown:
        return (rewards, breakdown)
    return rewards


def _scene_token_from_inputs(inputs: Any) -> Optional[str]:
    """Extract the NAVSIM scene token from the inputs dict (used to look up the metric_cache for PDM scoring).

    The OpenDriveVLA navsim dataset stores the raw 16-char hex token in
    wm_meta['sample_id'] (the data's 'token'/'id' fields have the form
    '<hex>_wm', so the suffix must be stripped). The collator in
    ``drivevla/training/data.py`` also puts the same sample id in the
    top-level ``_scene_token`` key, which is checked first.

    Lookup order: ``inputs['_scene_token']`` → ``inputs['wm_meta'][k]``
    (dict, or first element of a list/tuple of dicts) for k in sample_id,
    sample_token, scene_token, token → top-level ``inputs[k]`` for k in
    sample_id, scene_token, token, sample_token, frame_token (first element if
    it is a list/tuple). Each candidate is stringified and a trailing ``_wm``
    suffix is removed.

    Args:
        inputs: the batch dict passed to ``compute_loss``.

    Returns:
        The token string, or None if ``inputs`` is not a dict or no candidate
        was found. On a miss the code is meant to print the available keys once
        (guarded by the module global ``_SCENE_TOKEN_MISS_LOGGED``; see the
        NOTE at the end of the function body).
    """
    global _SCENE_TOKEN_MISS_LOGGED
    if not isinstance(inputs, dict):
        return None

    def _strip_suffix(t: str) -> str:
        """Drop the task suffix ("<token>_wm" -> "<token>"); otherwise return t."""
        if isinstance(t, str) and "_" in t:
            base, suf = t.rsplit("_", 1)
            if suf == "wm":
                return base
        return t

    # 1) Explicit top-level token set by the collator (non-empty string only).
    v = inputs.get("_scene_token", None)
    if isinstance(v, str) and v:
        return _strip_suffix(v)
    # 2) wm_meta as a single dict: first non-None candidate key wins.
    meta = inputs.get("wm_meta", None)
    if isinstance(meta, dict):
        for k in ("sample_id", "sample_token", "scene_token", "token"):
            v = meta.get(k, None)
            if v is not None:
                return _strip_suffix(str(v))
    # 3) wm_meta as a list/tuple of per-sample dicts: use the first sample.
    if isinstance(meta, (list, tuple)) and len(meta) > 0 and isinstance(meta[0], dict):
        for k in ("sample_id", "sample_token", "scene_token", "token"):
            v = meta[0].get(k, None)
            if v is not None:
                return _strip_suffix(str(v))
    # 4) Plain top-level keys (scalar, or the first element of a sequence).
    for k in ("sample_id", "scene_token", "token", "sample_token", "frame_token"):
        v = inputs.get(k, None)
        if v is None:
            continue
        if isinstance(v, (list, tuple)) and len(v) > 0:
            return _strip_suffix(str(v[0]))
        return _strip_suffix(str(v))
    # Nothing found: print the available keys once per process to help debug.
    # NOTE: ``_SCENE_TOKEN_MISS_LOGGED`` is not assigned at module level in
    # this file, so reaching this line raises NameError on the first miss.
    if not _SCENE_TOKEN_MISS_LOGGED:
        _SCENE_TOKEN_MISS_LOGGED = True
        try:
            keys = sorted(list(inputs.keys()))
        except Exception:
            keys = []
        meta_keys = []
        if isinstance(meta, dict):
            try:
                meta_keys = sorted(list(meta.keys()))
            except Exception:
                pass
        print(
            f"[GRPO][PDM][miss] _scene_token_from_inputs returned None — inputs top-level keys={keys}  wm_meta keys={meta_keys}"
        )
    return None


class GRPOConfig:
    """Typed snapshot of the ``--grpo-*`` command-line options.

    Built from the namespace that :func:`run` creates, which contains only the
    argparse attributes whose names start with ``grpo_`` (see
    ``GRPO_ARGUMENTS`` in ``drivevla/training/config.py``). Each attribute
    falls back to the default given here when the option is absent. The
    pairing (best vs worst), the objective and the PDM trajectory frame are
    fixed in the trainer and are not read here.
    """

    def __init__(self, args):
        """Read every GRPO option from ``args`` (any attribute container)."""
        # Group sampling / loss mixing.
        #   group_size   G trajectories sampled per scene
        #   loss_weight  multiplier of the GRPO loss in the total loss
        self.group_size = int(getattr(args, "grpo_group_size", 8))
        self.loss_weight = float(getattr(args, "grpo_loss_weight", 0.5))
        # Proxy reward weights and scales (see compute_sample_reward).
        self.w_ep = float(getattr(args, "grpo_reward_ep_weight", 0.5))
        self.w_comfort = float(getattr(args, "grpo_reward_comfort_weight", 0.2))
        self.ep_max = float(getattr(args, "grpo_ep_max", 4.0))
        self.sigma_comfort = float(getattr(args, "grpo_comfort_sigma", 1.0))
        # Stochastic DDIM sampling of the IK head and log-prob evaluation:
        #   diff_eta           DDIM eta (1.0 = DDPM-like noise)
        #   diff_num_steps     length K of the sampled denoising chain
        #   sigma_min_sample   floor on the per-step sampling sigma_t
        #   sigma_min_logprob  floor on sigma_t in the log-prob denominator
        #   gamma_denoising    per-step log-prob weight gamma^(K-1-i)
        #   xt_clip_value      +/- clip of the prior x_T and injected noise
        self.diff_eta = float(getattr(args, "grpo_diff_eta", 1.0))
        self.diff_num_steps = int(getattr(args, "grpo_diff_num_steps", 20))
        self.sigma_min_sample = float(getattr(args, "grpo_sigma_min_sample", 0.0))
        self.sigma_min_logprob = float(getattr(args, "grpo_sigma_min_logprob", 0.0))
        self.gamma_denoising = float(getattr(args, "grpo_gamma_denoising", 1.0))
        self.xt_clip_value = float(getattr(args, "grpo_xt_clip", 0.0))
        # Extra grpo/diag/* logging during the first diag_steps global steps.
        self.diag_steps = int(getattr(args, "grpo_diagnostic_steps", 0))
        # Reference model: the frozen reference policy is always built (see
        # setup_grpo); sft_ckpt is the checkpoint whose weights it loads.
        self.sft_ckpt = str(getattr(args, "grpo_sft_ckpt", "") or "")
        # Inline PDM scorer: metric cache dir (the scorer is created whenever it
        # is set).
        self.pdm_inline_metric_cache_dir = str(
            getattr(args, "grpo_pdm_inline_metric_cache_dir", "") or ""
        )
        # Preference-loss temperature beta in -logsigmoid(beta * delta).
        self.grpo_beta = float(getattr(args, "grpo_beta", 0.1))


class GRPOLatentActionWMTrainer(LatentActionWMTrainer):
    """Diffusion GRPO with raw NAVSIM EPDMS preferences and a frozen SFT reference.

    Extends the world-model trainer: ``compute_loss`` first runs the usual SFT
    forward/loss, then adds ``loss_weight`` x a pairwise preference loss on
    the IK diffusion head (see the module docstring for the full step).

    Instance state:
        _grpo_cfg:          GRPOConfig built from ``grpo_args``.
        _grpo_last_log_step: last global step whose grpo/* dict was logged.
        _ref_model:         frozen reference copy (None when disabled/failed).
        _ref_setup_done:    guards against building the reference twice.
        _pdm_inline:        InlinePDMScorer, or None (PDM disabled/failed).
    Attributes created lazily: ``_pdm_miss_count`` (metric-cache misses),
    ``_grpo_exc_count`` (GRPO exceptions).
    """

    def __init__(self, *args, grpo_args, **kwargs):
        """Create the trainer; ``grpo_args`` holds the ``grpo_*`` options.

        The GRPO state must exist before ``super().__init__`` (which builds
        the HF Trainer); ``setup_grpo`` needs the finished trainer (it reads
        ``self.model``) and therefore runs last.
        """
        self._grpo_cfg = GRPOConfig(grpo_args)
        self._grpo_last_log_step = -1
        self._ref_model: Optional[nn.Module] = None
        self._ref_setup_done = False
        self._pdm_inline = None
        super().__init__(*args, **kwargs)
        self.setup_grpo()

    def _load_optimizer_and_scheduler(self, checkpoint):
        """Use the stock Hugging Face optimizer/scheduler resume logic.

        Bypasses the remap-aware ``_load_optimizer_and_scheduler_with_remap``
        override that ``LatentActionWMTrainer`` installs (from checkpoint.py)
        and calls ``transformers.Trainer``'s implementation directly.
        """
        import transformers

        return transformers.Trainer._load_optimizer_and_scheduler(self, checkpoint)

    def setup_grpo(self) -> None:
        """Build the GRPO state; called once the Trainer has been constructed.

        Builds the frozen reference model and the inline PDM scorer (if a
        metric-cache dir is given). Does nothing if the model has no
        ``ik_head``. Safe to call again: the reference is only built once.
        """
        # No model / no IK head -> GRPO cannot run; leave everything disabled.
        model = getattr(self, "model", None)
        if model is None:
            return
        base = _unwrap(model)
        ik_head = getattr(base, "ik_head", None)
        if ik_head is None:
            rank0_print("[GRPO] ik_head 없음 — GRPO 비활성화.")
            return
        cfg = self._grpo_cfg
        rank0_print(f"[GRPO] setup. group_size={cfg.group_size}")
        # Frozen reference policy for the log-ratio (built once).
        if not self._ref_setup_done:
            self._setup_reference_model(base)
            self._ref_setup_done = True
        # Inline PDM scorer (reward source), scored on every GRPO step.
        if cfg.pdm_inline_metric_cache_dir:
            try:
                from .pdm import InlinePDMScorer

                # Construction is cheap; the devkit is imported lazily on the
                # first has_token()/score_batch() call.
                self._pdm_inline = InlinePDMScorer(
                    metric_cache_dir=cfg.pdm_inline_metric_cache_dir
                )
                rank0_print(
                    f"[GRPO][PDM][inline] InlinePDMScorer 등록: cache={cfg.pdm_inline_metric_cache_dir}"
                )
            except Exception as ex:
                # PDM is optional at this point: on failure training continues
                # with the proxy rewards only (compute_loss checks for None).
                rank0_print(f"[GRPO][PDM][inline] InlinePDMScorer 등록 실패: {ex}")
                self._pdm_inline = None

    def _setup_reference_model(self, base_model: nn.Module) -> None:
        """KL Option B: full deepcopy of LLM + BEV + IK as frozen reference.

        Acts as the anchor to the SFT distribution and keeps the policy from
        drifting away through reward hacking. Under DDP this roughly doubles
        GPU memory (7B in fp16 → +14GB). Under DeepSpeed ZeRO-3 a sharded copy
        cannot be made, so a warning is printed and the reference is disabled.

        Steps: deepcopy the (unwrapped) policy → freeze + eval → overwrite with
        the ``--grpo-sft-ckpt`` weights (tensors whose name and shape match;
        everything else keeps the copied policy-init values) → cast to bf16 →
        move to ``cuda:$LOCAL_RANK``. The reference is a plain attribute of the
        trainer: it is not wrapped by DDP, not optimized and not saved in
        checkpoints (it is rebuilt from ``sft_ckpt`` on every launch).

        Args:
            base_model: the unwrapped policy model to copy.

        Side effects:
            Sets ``self._ref_model`` (None on any failure, which disables the
            reference: the GRPO loss then uses log pi_ref = 0).
        """
        try:
            # ZeRO-3 partitions parameters (they carry a ``ds_id``); a
            # deepcopy would not produce a usable full model.
            if any((hasattr(p, "ds_id") for p in base_model.parameters())):
                rank0_print(
                    "[GRPO][KL] DeepSpeed ZeRO-3 환경에서는 reference model 비활성화 (sharded params, deepcopy 미지원). LoRA 기반 ref 가 필요합니다."
                )
                return
            import copy

            # Independent frozen copy (no grads, dropout off).
            ref = copy.deepcopy(base_model)
            for p in ref.parameters():
                p.requires_grad = False
            ref.eval()
            # Load the frozen SFT weights over the copy (without
            # --grpo-sft-ckpt the reference stays the policy init).
            sft_ckpt = self._grpo_cfg.sft_ckpt
            if sft_ckpt:
                try:
                    # Partial, shape-checked load: copy only tensors present
                    # in both state dicts with identical shapes. Shape
                    # mismatches (e.g. the ego projector, which is widened at
                    # the start of GRPO) and ref-only keys keep the deepcopied
                    # values; ckpt-only keys are ignored.
                    sd = _load_ckpt_state_dict(sft_ckpt)
                    ref_sd = ref.state_dict()
                    matched: Dict[str, torch.Tensor] = {}
                    skipped_shape: List[str] = []
                    for k, v in sd.items():
                        if k not in ref_sd:
                            continue
                        if tuple(ref_sd[k].shape) == tuple(v.shape):
                            matched[k] = v
                        else:
                            skipped_shape.append(k)
                    missing = [k for k in ref_sd if k not in sd]
                    unexpected = [k for k in sd if k not in ref_sd]
                    ref.load_state_dict(matched, strict=False)
                    rank0_print(
                        f"[GRPO][KL] ref ← FROZEN SFT ckpt: {sft_ckpt}\n           loaded {len(matched)}/{len(ref_sd)} tensors; skip(shape)={len(skipped_shape)}{skipped_shape[:4]}; ref-only(kept init)={len(missing)}; ckpt-only(ignored)={len(unexpected)}"
                    )
                except Exception as load_ex:
                    # Fall back to the deepcopy of the policy init (i.e. the
                    # weights loaded from --model-path).
                    rank0_print(
                        f"[GRPO][KL] FROZEN SFT ckpt 로드 실패 — deepcopy(model-path) 로 폴백: {type(load_ex).__name__}: {load_ex}"
                    )
            # Keep the reference in bf16 (a failed cast is logged and ignored).
            # Its log-probs are therefore computed at a different precision
            # than the policy's IK head, which may run in fp32.
            try:
                ref = ref.to(dtype=torch.bfloat16)
            except Exception as cast_ex:
                rank0_print(f"[GRPO][KL] ref bf16 cast 실패(무시 가능): {cast_ex}")
            # Placement: the whole reference goes to this rank's GPU.
            try:
                if torch.cuda.is_available():
                    local_rank = int(os.environ.get("LOCAL_RANK", 0))
                    ref_dev = torch.device(f"cuda:{local_rank}")
                    ref = ref.to(ref_dev)
                    rank0_print(f"[GRPO][KL] ref model moved to {ref_dev}")
            except Exception as dev_ex:
                rank0_print(f"[GRPO][KL] ref device move 실패(무시 가능): {dev_ex}")
            # Publish the reference and report its size (billions of params).
            # It is never refreshed: it stays the frozen SFT model.
            n_params = sum((p.numel() for p in ref.parameters())) / 1000000000.0
            self._ref_model = ref
            rank0_print(f"[GRPO][KL] Reference model 생성 완료. params={n_params:.2f}B")
        except Exception as ex:
            # Any failure disables the reference (GRPO then runs without it).
            rank0_print(
                f"[GRPO][KL] Reference model 생성 실패 — KL 비활성화: {type(ex).__name__}: {ex}"
            )
            self._ref_model = None

    def _maybe_apply_inline_pdm(
        self,
        samples: List[torch.Tensor],
        rewards: torch.Tensor,
        traj_scale: float,
        num_waypoints: int,
        inputs: Any,
        info: Dict[str, float],
    ) -> torch.Tensor:
        """Same-process PDM scoring → replaces the proxy rewards directly.

        Call frequency: the scorer is created when a metric-cache dir is
        given, and ``compute_loss`` then calls this method on every GRPO step.

        Args:
            samples: G tensors (1, traj_dim) in normalized IK units.
            rewards: (G,) proxy rewards (used for logging, device/dtype, and as
                the fallback return value on an exception).
            traj_scale: meters per normalized unit.
            num_waypoints: T, waypoints per sample.
            inputs: the batch dict (for the scene token and GT).
            info: dict that receives the ``grpo/pdm_inline/*`` log values
                (mutated in place).

        Returns:
            (G,) tensor of EPDMS-style PDM rewards (see
            ``InlinePDMScorer.aggregate_pdm_score``; samples that could not be
            scored get 0.0). None means "PDM-only: skip GRPO for this step" and
            is returned when the scene token is missing or not in the metric
            cache (``skip_reason`` = 1) or no sample could be scored
            (``skip_reason`` = 2). On an unexpected exception the *proxy*
            ``rewards`` are returned unchanged.

        Logged keys (in ``info``):
            grpo/pdm_inline/skip_reason   1 = token not in cache, 2 = all scores None
            grpo/pdm_inline/latency_s     wall time of score_batch [s]
            grpo/pdm_inline/pdm_mean/std  mean / unbiased std of the G PDM rewards
            grpo/pdm_inline/proxy_mean    mean of the proxy rewards (same step)
            grpo/pdm_inline/n_valid       number of samples that were scored
            grpo/pdm_inline/sub_{k}_mean/std/min  per sub-metric k (nc, dac,
                ddc, tlc, ep, ttc, lk, hc, ec) over the valid samples
                (numpy population std)
            grpo/pdm_inline/multiplicative_min/mean  of NC*DAC*DDC*TLC
        """
        # Not configured -> None (skip). (compute_loss only calls this method
        # when the scorer exists.)
        if self._pdm_inline is None:
            return None
        gs = int(getattr(self.state, "global_step", 0) or 0)
        # Resolve the scene token and check that its metric cache exists. Note
        # that this lookup is outside the try block below, so an exception
        # here propagates to compute_loss.
        scene_token = _scene_token_from_inputs(inputs)
        token_in_cache = bool(scene_token) and self._pdm_inline.has_token(scene_token)
        if not token_in_cache:
            info["grpo/pdm_inline/skip_reason"] = 1.0
            import sys as _sys

            # Count misses; print a detailed report (to stderr and stdout, on
            # every rank) for the first three only.
            _miss_n = int(getattr(self, "_pdm_miss_count", 0)) + 1
            self._pdm_miss_count = _miss_n
            if _miss_n <= 3:
                try:
                    _rank = int(os.environ.get("RANK", "0"))
                except Exception:
                    _rank = 0
                _tok_preview = repr(scene_token)[:80] if scene_token else "<empty>"
                _loader_n = (
                    len(self._pdm_inline._loader.tokens)
                    if hasattr(self._pdm_inline, "_loader") and self._pdm_inline._loader
                    else "?"
                )
                _input_keys = (
                    sorted(list(inputs.keys()))[:20] if isinstance(inputs, dict) else "N/A"
                )
                _wm = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
                _wm_keys = sorted(list(_wm.keys()))[:20] if isinstance(_wm, dict) else "N/A"
                _wm_sid = _wm.get("sample_id", "<missing>") if isinstance(_wm, dict) else "N/A"
                _top_st = (
                    inputs.get("_scene_token", "<missing>") if isinstance(inputs, dict) else "N/A"
                )
                _msg = f"[GRPO][PDM][miss#{_miss_n}] rank={_rank} step={gs}\n  scene_token resolved = {_tok_preview}\n  loader has {_loader_n} tokens (cache_dir={self._grpo_cfg.pdm_inline_metric_cache_dir})\n  inputs.keys() = {_input_keys}\n  inputs['_scene_token'] = {_top_st!r}\n  inputs['wm_meta'].keys() = {_wm_keys}\n  inputs['wm_meta']['sample_id'] = {_wm_sid!r}"
                print(_msg, file=_sys.stderr, flush=True)
                print(_msg, flush=True)
            return None
        try:
            # De-normalize the G samples to meters: (G, T, 2) float32 numpy.
            # They are passed to the NAVSIM scorer unchanged, i.e. read as
            # ego-frame (x, y) poses, with headings derived by the scorer.
            samples_m_torch = (
                torch.stack([s.reshape(num_waypoints, 2) for s in samples], dim=0) * traj_scale
            )
            samples_m_np = samples_m_torch.detach().float().cpu().numpy()
            # Score the whole group against the scene's metric cache (timed).
            t0 = time.time()
            scores = self._pdm_inline.score_batch(samples_m_np, scene_token)
            dt = time.time() - t0
            # No sample could be scored -> skip GRPO for this step.
            valid = [s for s in scores if s is not None]
            if not valid:
                info["grpo/pdm_inline/skip_reason"] = 2.0
                return None
            # Scalar EPDMS-style reward per sample (NaN for unscored samples),
            # then NaN -> 0.0 (``v == v`` is False only for NaN), so an
            # unscored sample ranks as the worst in the group. The tensor takes
            # the proxy rewards' dtype, i.e. the sample dtype (bf16 when the
            # model runs in bf16, as in the recipe).
            pdm_scores = [
                self._pdm_inline.aggregate_pdm_score(s) if s is not None else float("nan")
                for s in scores
            ]
            pdm_arr = torch.tensor(
                [v if v == v else 0.0 for v in pdm_scores],
                device=rewards.device,
                dtype=rewards.dtype,
            )
            # Group-level reward statistics.
            info["grpo/pdm_inline/latency_s"] = dt
            info["grpo/pdm_inline/pdm_mean"] = float(pdm_arr.mean())
            info["grpo/pdm_inline/pdm_std"] = float(pdm_arr.std())
            info["grpo/pdm_inline/proxy_mean"] = float(rewards.float().mean())
            info["grpo/pdm_inline/n_valid"] = float(len(valid))
            # Per-sub-metric statistics over the valid samples, plus the
            # product of the multiplicative (hard-constraint) metrics. Logging
            # failures are reported but do not affect the rewards.
            try:
                _comp_keys = ("nc", "dac", "ddc", "tlc", "ep", "ttc", "lk", "hc", "ec")
                for k in _comp_keys:
                    vals = [s[k] for s in valid if s is not None and k in s]
                    if vals:
                        import numpy as _np

                        info[f"grpo/pdm_inline/sub_{k}_mean"] = float(_np.mean(vals))
                        info[f"grpo/pdm_inline/sub_{k}_std"] = float(_np.std(vals))
                        info[f"grpo/pdm_inline/sub_{k}_min"] = float(_np.min(vals))
                bin_scores = [
                    s["nc"] * s["dac"] * s["ddc"] * s["tlc"] for s in valid if s is not None
                ]
                if bin_scores:
                    info["grpo/pdm_inline/multiplicative_min"] = float(min(bin_scores))
                    info["grpo/pdm_inline/multiplicative_mean"] = float(
                        sum(bin_scores) / len(bin_scores)
                    )
            except Exception as ex:
                rank0_print(f"[GRPO][PDM] sub-score logging 실패: {ex}")
            # PDM-only reward: the proxy is discarded.
            rewards = pdm_arr
            return rewards
        except Exception as ex:
            # Unexpected failure: keep training with the proxy rewards for
            # this step (unlike the skip cases above, which return None).
            rank0_print(
                f"[GRPO][PDM][inline] _maybe_apply_inline_pdm 실패 step={gs}: {type(ex).__name__}: {ex}"
            )
            return rewards

    def _compute_grpo_loss(
        self,
        log_probs: torch.Tensor,
        rewards: torch.Tensor,
        inputs: Any,
        chain: List[Dict[str, Any]],
        base_for_ref_refresh: Any,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Diffusion-GRPO loss: pair-wise preference from PDM rewards.

        L = -log σ(β · [(log π_θ(τ_w) - log π_ref(τ_w)) - (log π_θ(τ_l) - log π_ref(τ_l))])

        τ_w is the highest-reward sample of the group and τ_l the lowest; log π
        is the (γ-weighted) sum of per-step Gaussian log-likelihoods of the
        recorded denoising chain (``compute_logprob_for_chain``); β is
        ``cfg.grpo_beta``. Minimizing L raises the winner's log-ratio relative to
        the loser's; the reference term anchors both to the frozen SFT model.

        Pair selection: best vs worst (one pair per group).

        Edge cases:
          - fewer than 2 rewards, or rewards.std() < 1e-6 → all samples are
            equivalent (e.g. a tie across the whole group) → no signal →
            return a zero loss (no graph) and log ``grpo/skipped_no_var``; the
            reference is not evaluated in that case.
          - ref_model None / ref forward failure → log_probs_ref=0 substitute
            (GRPO degrades to the policy log-ratio only, still functional but
            weaker) and ``grpo/ref_missing`` = 1.

        Args:
            log_probs: (G,) float32 log π_θ(τ_g) of each sampled chain, with
                gradients into the IK head.
            rewards: (G,) rewards (PDM rewards in the released recipe).
            inputs: batch dict; re-used for the reference forward.
            chain: the recorded sampling chain (list of per-step dicts).
            base_for_ref_refresh: the policy model (unused).

        Returns:
            (grpo_loss, info): 0-dim loss tensor and the ``grpo/*`` scalars:
              grpo/skipped_no_var   1 if the group had no reward variance
              grpo/ref_missing      1 if log π_ref was replaced by zeros
              grpo/n_pairs          number of preference pairs (1)
              grpo/log_ratio_w_mean mean log π_θ - log π_ref of the winner(s)
              grpo/log_ratio_l_mean same for the loser(s)
              grpo/delta_mean/std   winner minus loser log-ratio (std 0 for 1 pair)
              grpo/loss             value of the preference loss
              grpo/accuracy         fraction of pairs with delta > 0, i.e. the
                                    policy already prefers the winner more
                                    than the reference does
              grpo/beta             β used
              grpo/reward_gap_mean  reward(winner) - reward(loser)
        """
        cfg = self._grpo_cfg
        info: Dict[str, float] = {}
        device = log_probs.device
        dtype = log_probs.dtype
        zero_loss = torch.zeros((), device=device, dtype=dtype)
        # No reward variance (ties / single sample) -> no preference signal.
        rewards_f = rewards.float()
        if rewards_f.numel() < 2 or rewards_f.std() < 1e-06:
            info["grpo/skipped_no_var"] = 1.0
            return (zero_loss, info)
        # Rank the group by reward and take best (winner) vs worst (loser).
        sorted_idx = torch.argsort(rewards_f, descending=True)
        winners_idx = sorted_idx[:1]
        losers_idx = sorted_idx[-1:]
        # Reference log-probs of the *same* chains (no grad); zeros if the
        # reference is unavailable.
        log_probs_ref = self._compute_ref_logprobs_for_chain(inputs, chain)
        if log_probs_ref is None:
            log_probs_ref = torch.zeros_like(log_probs.detach())
            info["grpo/ref_missing"] = 1.0
        log_probs_ref = log_probs_ref.detach().to(device=device, dtype=dtype)
        # Implicit-reward difference between winner and loser, then the
        # logistic (Bradley-Terry) preference loss scaled by beta.
        log_ratio = log_probs - log_probs_ref
        lr_w = log_ratio.index_select(0, winners_idx)
        lr_l = log_ratio.index_select(0, losers_idx)
        delta = lr_w - lr_l
        beta = float(cfg.grpo_beta)
        grpo_loss = -F.logsigmoid(beta * delta).mean()
        # Scalars for logging (see the docstring for their meaning).
        with torch.no_grad():
            info["grpo/n_pairs"] = float(winners_idx.numel())
            info["grpo/log_ratio_w_mean"] = float(lr_w.mean())
            info["grpo/log_ratio_l_mean"] = float(lr_l.mean())
            info["grpo/delta_mean"] = float(delta.mean())
            info["grpo/delta_std"] = float(delta.std()) if delta.numel() > 1 else 0.0
            info["grpo/loss"] = float(grpo_loss.detach())
            info["grpo/accuracy"] = float((delta > 0).float().mean())
            info["grpo/beta"] = beta
            info["grpo/reward_gap_mean"] = float(
                (
                    rewards_f.index_select(0, winners_idx) - rewards_f.index_select(0, losers_idx)
                ).mean()
            )
        return (grpo_loss, info)

    @torch.no_grad()
    def _compute_ref_logprobs_for_chain(
        self, inputs: Any, chain: List[Dict[str, Any]]
    ) -> Optional[torch.Tensor]:
        """Compute log π_ref(τ) of the same chain with the frozen reference model.

        The GRPO loss needs log π_θ - log π_ref for both the winner and the
        loser, so the reference model's chain log_probs are obtained here
        without gradients.

        The reference runs its *own* full forward on the batch (its LLM
        predicts its own future visual tokens), and its IK head then scores the
        policy's recorded chain (x_t → x_{t-1} transitions) conditioned on the
        reference's current / predicted visual tokens. Sampling-time options
        (sigma floor, gamma) are the same as for the policy.

        Args:
            inputs: batch dict (supervision fields are stripped first).
            chain: list of per-step dicts from ``sample_with_logprob_data``
                (``x_t``, ``x_prev`` (G, traj_dim) fp32, ``t_idx``,
                ``t_prev``, ``sigma_t``).

        Returns:
            log_probs_ref: (G,) float32 tensor — log p of the same chain under
            the ref policy. None if there is no reference, the reference
            forward produced no IK tokens, or anything raised (the error is
            printed on rank 0).
        """
        # No reference available (ZeRO-3 or setup failure).
        if self._ref_model is None:
            return None
        try:
            # Target device: where the chain tensors live; otherwise the
            # reference's own parameter device (or cuda/cpu as a last resort).
            try:
                target_device = chain[0]["x_t"].device if chain else None
            except Exception:
                target_device = None
            if target_device is None:
                try:
                    target_device = next(
                        _unwrap(self._ref_model.module).parameters()
                        if hasattr(self._ref_model, "module")
                        else self._ref_model.parameters()
                    ).device
                except StopIteration:
                    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # Move the reference there if needed (the move persists).
            try:
                ref_device = next(self._ref_model.parameters()).device
                if ref_device != target_device:
                    self._ref_model = self._ref_model.to(target_device)
            except StopIteration:
                pass
            # Forward without supervision targets; return_ik_pred_for_debug
            # forces the model to run its IK branch even without gt_traj, which
            # is what stores _grpo_cur_tokens / _grpo_next_tokens_list on the
            # reference's IK head. (labels are kept, so the reference also
            # computes its LM loss; the output itself is discarded.)
            light_inputs = self._strip_supervision_fields(inputs)
            if isinstance(light_inputs, dict):
                light_inputs = {**light_inputs, "return_ik_pred_for_debug": True}
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                _ = self._ref_model(**light_inputs)
            # Conditioning tokens captured by the reference forward.
            ref_ik = _unwrap(self._ref_model).ik_head
            cur_tokens_ref = getattr(ref_ik, "_grpo_cur_tokens", None)
            next_toks_ref = getattr(ref_ik, "_grpo_next_tokens_list", None) or []
            if cur_tokens_ref is None:
                return None
            # Log-likelihood of the policy's chain under the reference head.
            log_probs_ref = ref_ik.compute_logprob_for_chain(
                cur_tokens_ref,
                next_toks_ref,
                chain,
                sigma_min_logprob=self._grpo_cfg.sigma_min_logprob,
                gamma_denoising=self._grpo_cfg.gamma_denoising,
            )
            return log_probs_ref
        except Exception as ex:
            # Caller substitutes zeros and logs grpo/ref_missing.
            rank0_print(f"[GRPO] ref log_probs 계산 실패: {type(ex).__name__}: {ex}")
            return None

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """SFT loss + GRPO loss.

        Steps (see also the module docstring):
          1. force ``ik_stop_grad`` and run the parent SFT forward/loss;
          2. if the IK head captured its conditioning tokens, sample G
             trajectories with their denoising chain;
          3. compute proxy rewards (with a per-component breakdown during the
             diagnostic window) and, if the PDM scorer exists, replace them by
             PDM rewards — or return the SFT loss alone when PDM skips the
             scene;
          4. recompute the chain log-probs with gradients and build the
             pairwise preference loss against the reference;
          5. total = sft_loss + loss_weight * grpo_loss; log grpo/* scalars and
             optionally print a debug block.
        Any exception inside steps 2-4 is caught: the step falls back to
        SFT-only (unless ``ODVLA_STRICT_REWARD`` is set, which re-raises).

        Args:
            model: the (possibly DDP-wrapped) policy model.
            inputs: collated batch dict (per-device batch size 1 in the recipe).
            return_outputs: also return the model outputs.
            **kwargs: forwarded to the parent ``compute_loss``.

        Returns:
            ``total`` (0-dim tensor), or ``(total, outputs)`` when
            ``return_outputs``.

        Extra keys this method may add to the logged dict:
            grpo/pdm/skipped           1 when PDM returned None (scene
                                            skipped, SFT-only step)
            grpo/reward_exception           1 when the GRPO branch raised
            grpo/reward_exception_total     running count of such exceptions
            grpo/diag/*                     see _collect_diag_metrics
        """
        # Detach the IK head's inputs inside the model forward, so the IK
        # losses (SFT and GRPO) do not backpropagate into the LLM; this
        # overrides whatever the curriculum callback set.
        base = _unwrap(model)
        try:
            base.ik_stop_grad = True
        except Exception:
            pass
        # Regular SFT forward + loss (LatentActionWMTrainer.compute_loss). Its
        # forward also stores the IK conditioning tokens used below.
        sft_loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kwargs)
        grpo_loss = torch.zeros((), device=sft_loss.device, dtype=sft_loss.dtype)
        grpo_info: Dict[str, float] = {}
        cfg = self._grpo_cfg
        if cfg.loss_weight > 0:
            base = _unwrap(model)
            ik_head = getattr(base, "ik_head", None)
            # Ready check: the SFT forward must have stored _grpo_cur_tokens on
            # the diffusion IK head (these attributes are never cleared, so a
            # forward that skips the IK branch leaves the previous step's
            # tokens in place).
            if getattr(ik_head, "_grpo_cur_tokens", None) is not None:
                try:
                    # Normalization scale [m per unit] and waypoint count T.
                    traj_scale = float(getattr(base, "ik_traj_scale", 1.0) or 1.0)
                    num_wps = int(getattr(base, "ik_num_waypoints", 6) or 6)
                    # Group sampling: G stochastic-DDIM trajectories for this
                    # scene plus the recorded chain (no grad).
                    samples, chain = self._collect_xattn_diffusion_samples_with_chain(
                        ik_head,
                        cfg.group_size,
                        eta=cfg.diff_eta,
                        num_steps=cfg.diff_num_steps,
                        sigma_min=cfg.sigma_min_sample,
                        xt_clip_value=cfg.xt_clip_value,
                    )
                    # A preference pair needs at least two samples and the
                    # chain; otherwise this step stays SFT-only (grpo_loss=0).
                    if len(samples) >= 2 and chain is not None:
                        # Rewards are computed on the final (noisy) chain
                        # samples.
                        reward_traj = samples
                        # Proxy rewards; during the first diag_steps steps
                        # also the per-component breakdown + grpo/diag/*.
                        _diag = self._grpo_diagnostic_active()
                        if _diag:
                            rewards, _r_breakdown = compute_group_rewards(
                                reward_traj,
                                traj_scale,
                                cfg,
                                num_wps,
                                return_breakdown=True,
                            )
                            self._collect_diag_metrics(
                                samples, rewards, _r_breakdown, chain, traj_scale, grpo_info
                            )
                        else:
                            rewards = compute_group_rewards(reward_traj, traj_scale, cfg, num_wps)
                        # PDM rewards replace the proxy. None = PDM-only skip
                        # (scene not in the metric cache / nothing scored):
                        # log the collected keys right away (one self.log call
                        # per key, every rank, not throttled by
                        # _try_log_grpo) and return the SFT loss alone.
                        if self._pdm_inline is not None:
                            rewards = self._maybe_apply_inline_pdm(
                                reward_traj, rewards, traj_scale, num_wps, inputs, grpo_info
                            )
                            if rewards is None:
                                grpo_info["grpo/pdm/skipped"] = 1.0
                                for k, v in grpo_info.items():
                                    self.log({k: v}) if hasattr(self, "log") else None
                                return (sft_loss, outputs) if return_outputs else sft_loss
                        # Conditioning tokens for the differentiable log-prob:
                        # prefer the *_grad versions stored by the forward
                        # (detached anyway under ik_stop_grad, so gradients
                        # flow into the IK head parameters only).
                        _grad_t = getattr(ik_head, "_grpo_cur_tokens_grad", None)
                        cur_tokens_grad = (
                            _grad_t if _grad_t is not None else ik_head._grpo_cur_tokens
                        )
                        _grad_l = getattr(ik_head, "_grpo_next_tokens_list_grad", None)
                        if _grad_l is None:
                            _grad_l = ik_head._grpo_next_tokens_list
                        next_toks_grad = _grad_l if _grad_l is not None else []
                        # log π_θ(τ_g) of each recorded chain, with gradients
                        # (the reference enters only through the log-ratio in
                        # _compute_grpo_loss).
                        log_probs = ik_head.compute_logprob_for_chain(
                            cur_tokens_grad,
                            next_toks_grad,
                            chain,
                            sigma_min_logprob=cfg.sigma_min_logprob,
                            gamma_denoising=cfg.gamma_denoising,
                        )
                        # Pairwise preference loss (best vs worst) against the
                        # frozen reference; also yields the grpo/* scalars.
                        grpo_loss, _grpo_info = self._compute_grpo_loss(
                            log_probs=log_probs,
                            rewards=rewards,
                            inputs=inputs,
                            chain=chain,
                            base_for_ref_refresh=base,
                        )
                        grpo_info.update(_grpo_info)
                except Exception as ex:
                    # Any GRPO failure degrades this step to SFT-only: print the
                    # traceback (stderr, every rank) and count the event.
                    # ODVLA_STRICT_REWARD=1/true re-raises instead.
                    import traceback, sys as _sys

                    _msg = (
                        f"[GRPO][RANK {os.environ.get('RANK', '?')}] 손실 계산 오류 → 이 step 은 SFT-only 로 퇴화: {type(ex).__name__}: {ex}\n"
                        + traceback.format_exc()
                    )
                    print(_msg, file=_sys.stderr, flush=True)
                    grpo_info["grpo/reward_exception"] = 1.0
                    self._grpo_exc_count = int(getattr(self, "_grpo_exc_count", 0)) + 1
                    grpo_info["grpo/reward_exception_total"] = float(self._grpo_exc_count)
                    if str(os.environ.get("ODVLA_STRICT_REWARD", "0")).lower() in ("1", "true"):
                        raise
        # Combine: total = sft_loss + loss_weight * grpo_loss.
        total = sft_loss + cfg.loss_weight * grpo_loss
        # Throttled logging of the grpo/* dict (rank 0, every logging_steps).
        self._try_log_grpo(grpo_info)
        # Optional debug block (rank 0, every --debug-print-steps steps, via
        # DebugTrainer._should_print): prompt, template and GRPO summary.
        # Failures here never affect training.
        try:
            if int(getattr(self, "_dbg_steps", 0) or 0) > 0 and self.is_world_process_zero():
                _gs = int(getattr(self.state, "global_step", 0) or 0)
                _meta = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
                _was_trunc = bool(_meta.get("was_truncated", False))
                if self._should_print(_gs, _was_trunc):
                    # Prompt pieces from the sample metadata; long texts are
                    # trimmed to at most 1500 characters.
                    _sys_p = str(_meta.get("system", "") or "")
                    _q = str(_meta.get("question", "") or "")
                    _tmpl = str(_meta.get("conv_template", "") or "")
                    _sid = _meta.get("sample_id", None)
                    _max = int(min(getattr(self, "_dbg_max_chars", 4000), 1500))

                    def _trim(s):
                        """Truncate ``s`` to ``_max`` chars, noting how many were cut."""
                        s = str(s or "")
                        return s if len(s) <= _max else s[:_max] + f"\n…<+{len(s) - _max} chars>"

                    # Summary lines: losses, configuration, selected grpo/*
                    # metrics.
                    _g = grpo_info
                    _grpo_lines = []
                    _grpo_lines.append(
                        f"sft_loss={float(sft_loss.detach()):.6f} grpo_loss={float(grpo_loss.detach()):.6f} total={float(total.detach()):.6f}"
                    )
                    _grpo_lines.append(
                        f"group_size={cfg.group_size} loss_weight={cfg.loss_weight}"
                    )
                    # Only these keys are echoed.
                    _wanted = (
                        "grpo/reward_gap_mean",
                        "grpo/accuracy",
                        "grpo/skipped_no_var",
                        "grpo/pdm_inline/skip_reason",
                        "grpo/pdm/skipped",
                    )
                    _stats = {
                        k: float(v)
                        for (k, v) in _g.items()
                        if k in _wanted and isinstance(v, (int, float))
                    }
                    if _stats:
                        _grpo_lines.append(
                            "metrics: " + " ".join((f"{k}={v:.4g}" for (k, v) in _stats.items()))
                        )
                    # One framed block: header, template, system prompt,
                    # question, GRPO summary.
                    rank0_print(
                        "\n"
                        + "=" * 90
                        + f"\n[GRPO_DEBUG] step={_gs} sample_id={_sid}"
                        + "\n"
                        + "-" * 90
                        + "\n[CONV_TEMPLATE]\n"
                        + _tmpl
                        + "\n\n[SYSTEM]\n"
                        + _trim(_sys_p)
                        + "\n\n[QUESTION]\n"
                        + _trim(_q)
                        + "\n\n[GRPO]\n"
                        + "\n".join(_grpo_lines)
                        + "\n"
                        + "=" * 90
                        + "\n"
                    )
        except Exception as _gdbg_e:
            # Debug printing is best effort; even the warning may fail silently.
            try:
                rank0_print(
                    f"[WARN][GRPO_DEBUG] print failed: {type(_gdbg_e).__name__}: {_gdbg_e}"
                )
            except Exception:
                pass
        return (total, outputs) if return_outputs else total

    def _collect_xattn_diffusion_samples_with_chain(
        self,
        ik_head: nn.Module,
        G: int,
        eta: float = 1.0,
        num_steps: int = 20,
        sigma_min: float = 0.0,
        xt_clip_value: float = 0.0,
    ) -> Tuple[List[torch.Tensor], Optional[List[Dict[str, Any]]]]:
        """For DDPO: collect G samples and their sampling chain with stochastic DDIM.

        All G samples share the scene's conditioning (the current visual tokens
        and the LLM-predicted future visual tokens stored on the IK head by the
        SFT forward); diversity comes only from the random prior x_T ~ N(0, I)
        and the noise injected at every step (sigma_t, floored at
        ``sigma_min``). Runs without gradients, in fp32.

        Args:
            ik_head: the CrossAttentionDiffusionIK head.
            G: group size (number of trajectories to sample).
            eta: DDIM eta (stochasticity of each step).
            num_steps: number of denoising steps K in the chain.
            sigma_min: floor on sigma_t during sampling.
            xt_clip_value: +/- clip of x_T and of the injected noise (0 = off).

        Returns:
            samples:    G × (1, traj_dim) — the noisy final samples, in
                        normalized units (meters / ik_traj_scale), cast back to
                        the token dtype (bf16 in the recipe); the default for
                        log-prob / exploration / rewards
            chain:      List[Dict] — the chain exactly as returned by
                        sample_with_logprob_data (passed unchanged to
                        compute_logprob_for_chain); one dict per step with
                        x_t, x_prev (G, traj_dim) fp32, t_idx, t_prev, sigma_t,
                        x0_clean
            ([], None) if no tokens are available or sampling raised.
        """
        # Conditioning captured during the SFT forward.
        cur_tokens = getattr(ik_head, "_grpo_cur_tokens", None)
        next_toks = getattr(ik_head, "_grpo_next_tokens_list", None) or []
        if cur_tokens is None:
            return ([], None)
        # Draw the G chains in one batched call (the head expands the tokens
        # to G copies internally).
        try:
            with torch.no_grad():
                trajs_batched, chain = ik_head.sample_with_logprob_data(
                    cur_tokens,
                    next_toks,
                    num_samples=int(G),
                    num_steps=int(num_steps),
                    eta=float(eta),
                    sigma_min=float(sigma_min),
                    xt_clip_value=float(xt_clip_value),
                )
        except Exception as ex:
            rank0_print(f"[GRPO] sample_with_logprob_data 실패: {type(ex).__name__}: {ex}")
            return ([], None)
        # Split the (G, traj_dim) batch into G independent (1, traj_dim) copies.
        samples = [
            trajs_batched[g : g + 1].detach().clone() for g in range(int(trajs_batched.shape[0]))
        ]
        return (samples, chain)

    @staticmethod
    def _strip_supervision_fields(inputs: Any) -> Any:
        """Prepare the inputs for an auxiliary forward (originally the dropout-sampling forward; here the reference forward).

        Because ``return_ik_pred_for_debug=True`` is added separately by the
        caller, the IK head still runs without gt_traj. gt_traj is removed as
        well to save computation. Removed keys: GT trajectory, object-query box
        targets / masks, and ``wm_meta``.
        Everything else (token ids, labels, visual features, ego status, ...)
        is kept.

        Args:
            inputs: batch dict (non-dicts are returned unchanged).

        Returns:
            A new shallow-copied dict without the supervision keys.
        """
        if not isinstance(inputs, dict):
            return inputs
        skip = {
            "gt_traj",
            "query_box_targets",
            "query_has_obj",
            "query_box_targets_next",
            "query_has_obj_next",
            "wm_meta",
        }
        return {k: v for (k, v) in inputs.items() if k not in skip}

    def _grpo_diagnostic_active(self) -> bool:
        """True while ``global_step < cfg.diag_steps`` (diagnostic window).

        ``--grpo-diagnostic-steps`` (1000 in the recipe) enables the reward
        breakdown and the grpo/diag/* metrics for the first N steps; 0 or a
        missing config disables them.
        """
        cfg = self._grpo_cfg
        if cfg is None:
            return False
        steps = int(getattr(cfg, "diag_steps", 0))
        if steps <= 0:
            return False
        gs = int(getattr(self.state, "global_step", 0) or 0)
        return gs < steps

    @torch.no_grad()
    def _collect_diag_metrics(
        self,
        samples: List[torch.Tensor],
        rewards: torch.Tensor,
        breakdown: Dict[str, torch.Tensor],
        chain: List[Dict[str, Any]],
        traj_scale: float,
        info: Dict[str, float],
    ) -> None:
        """Add group-diversity / reward / noise diagnostics to ``info`` (in place).

        Logged keys:
            grpo/diag/traj_std_per_dim   std over the G samples of each
                                         trajectory coordinate [m], averaged
                                         over coordinates
            grpo/diag/traj_pairwise_l2_m mean pairwise Euclidean distance [m]
                                         between the flattened (traj_dim,)
                                         trajectories
            grpo/diag/r_{k}_mean / _std  mean / std over the group of each raw
                                         proxy component k (l2, ep, comfort;
                                         0 if not computed)
            grpo/diag/sigma_t_min/max/mean  sampling sigma_t over the chain
            grpo/diag/chain_len          number of chain steps K
            grpo/diag/xT_std             std over samples of the prior x_T
                                         (normalized units), averaged over dims

        Args:
            samples: G tensors (1, traj_dim) in normalized units.
            rewards: (G,) proxy rewards (unused here).
            breakdown: per-component (G,) proxy values.
            chain: the sampling chain.
            traj_scale: meters per normalized unit.
            info: log dict to update.
        """
        if not samples:
            return
        try:
            # (G, traj_dim) samples in meters.
            G = len(samples)
            S = torch.stack([s.reshape(-1) for s in samples], dim=0).float() * traj_scale
            info["grpo/diag/traj_std_per_dim"] = float(S.std(dim=0).mean())
            # Mean over the upper triangle (i < j) of the pairwise distances.
            if G >= 2:
                pdists = torch.cdist(S, S, p=2)
                triu = torch.triu_indices(G, G, offset=1)
                if triu.shape[1] > 0:
                    info["grpo/diag/traj_pairwise_l2_m"] = float(pdists[triu[0], triu[1]].mean())
            # Spread of each proxy-reward component within the group.
            for k, v in (breakdown or {}).items():
                if v is None or v.numel() == 0:
                    continue
                vf = v.float()
                info[f"grpo/diag/r_{k}_mean"] = float(vf.mean())
                info[f"grpo/diag/r_{k}_std"] = float(vf.std())
            # Per-step sampling noise levels recorded in the chain.
            sigmas = [float(c.get("sigma_t", 0.0)) for c in chain]
            if sigmas:
                info["grpo/diag/sigma_t_min"] = float(min(sigmas))
                info["grpo/diag/sigma_t_max"] = float(max(sigmas))
                info["grpo/diag/sigma_t_mean"] = float(sum(sigmas) / len(sigmas))
                info["grpo/diag/chain_len"] = float(len(sigmas))
            # Spread of the starting noise x_T (first chain entry).
            if chain:
                xT = chain[0]["x_t"].float()
                info["grpo/diag/xT_std"] = float(xT.std(dim=0).mean())
        except Exception as ex:
            rank0_print(f"[GRPO][diag] 수집 실패: {type(ex).__name__}: {ex}")

    def _try_log_grpo(self, info: Dict[str, float]) -> None:
        """Log the grpo/* dict, throttled like the trainer's own WM logging.

        Logs only on the world-process-zero rank, only when
        ``global_step % logging_steps == 0`` (50 in the recipe), and at most
        once per global step (the first micro-batch that reaches it), so the
        values are a single micro-batch snapshot, not an average. Logging
        errors are swallowed.

        Args:
            info: ``{key: float}``; nothing is logged when empty.
        """
        if not info:
            return
        try:
            gs = int(getattr(self.state, "global_step", 0) or 0)
            log_every = max(1, int(getattr(self.args, "logging_steps", 1) or 1))
            if (
                self.is_world_process_zero()
                and gs % log_every == 0
                and (gs != self._grpo_last_log_step)
            ):
                self.log(info)
                self._grpo_last_log_step = gs
        except Exception:
            pass


def _unwrap(model: nn.Module) -> nn.Module:
    """Return ``model.module`` for DDP/DeepSpeed-style wrappers, else ``model``."""
    return model.module if hasattr(model, "module") else model


def run(args):
    """Entry point of the NAVSIM GRPO stage (called by the GRPO training script).

    Runs the normal training pipeline (``drivevla.training.train.main``: data,
    model, IK head, checkpoint loading, HF Trainer) with a GRPO-enabled
    trainer class.

    Args:
        args: the parsed argparse namespace (base training options plus the
            "NAVSIM diffusion GRPO" group from ``add_grpo_args``).
    """
    # Hand the trainer only the options whose names start with "grpo_".
    grpo_args = SimpleNamespace(
        **{k: v for (k, v) in vars(args).items() if k.startswith("grpo_")}
    )

    class GRPOTrainer(GRPOLatentActionWMTrainer, LatentActionDebugTrainer):
        """GRPO trainer combined with the debug-print trainer.

        MRO: GRPOLatentActionWMTrainer → LatentActionWMTrainer →
        LatentActionDebugTrainer → DebugTrainer → transformers.Trainer.
        LatentActionDebugTrainer/DebugTrainer accept the debug_print_* kwargs
        that ``train.main`` passes to the trainer constructor and provide
        ``_dbg_steps`` / ``_should_print``. ``compute_loss`` resolves to the
        GRPO version, whose ``super()`` call reaches
        ``LatentActionWMTrainer.compute_loss`` (that one does not call
        ``super()``, so LatentActionDebugTrainer.compute_loss is not used).
        """

        def __init__(self, *a, **kw):
            """Inject the captured ``grpo_args`` into the GRPO trainer."""
            super().__init__(*a, grpo_args=grpo_args, **kw)

    # Build data / model / trainer and train with this trainer class.
    train(args, trainer_cls=GRPOTrainer)
