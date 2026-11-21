"""Cross-attention diffusion IK and camera query heads.

This module holds the small neural heads that sit on top of the LLaVA-style
Qwen2.5 world model (see ``drivevla/training/model.py`` for the forward pass and
``drivevla/training/train.py`` for where the heads are attached to the model):

* :class:`CrossAttentionDiffusionIK` -- the "Inverse Kinematics" (IK) network of
  DriveIK. It is a conditional DDPM/DDIM model over the flattened future
  trajectory ``[x1, y1, x2, y2, ..., xT, yT]`` (``traj_dim = 2 * T``; T = 8
  waypoints for NAVSIM, 0.5 s apart). The trajectory is in the
  ego frame (x = lateral, y = forward, meters) and is normalised by
  ``ik_traj_scale`` (default 20 m) before it reaches this head. The denoiser
  sees *only* visual states: the current-frame DINOv3 tokens (projected into the
  LLM hidden space by ``mm_projector_bev``) and the LLM's predicted next-frame
  visual tokens (hidden states at the ``<BEV_STATE>`` positions). It never sees
  ego status or text. Besides SFT training (:meth:`~CrossAttentionDiffusionIK.diffusion_loss`)
  and sampling (:meth:`~CrossAttentionDiffusionIK.sample`), it exposes the
  stochastic-chain sampler and per-step log-probabilities used by the NAVSIM
  GRPO stage (``drivevla/training/grpo.py``).
* :class:`CameraConditionedQueryPool` / :class:`ObjectQueryHead` -- the 4-query
  attention pool over the 3 front cameras (``model.ik_attn_pool``) and the
  per-query auxiliary object-box heads supervised in ``model.py``.

Environment variable:
    ODVLA_IK_HEAD_FP32: when truthy ("1", "true", ...), ``train.py`` casts the IK
        head parameters to fp32 and the denoiser / token projection below run
        with autocast disabled (see :func:`_ik_fp32_ctx`).
"""

from __future__ import annotations
import contextlib
import math
import os
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Module-level flag read once at import time: any value other than "", "0",
# "false", "no" or "off" (case-insensitive, surrounding whitespace ignored)
# enables fp32 IK-head computation. ``train.py`` imports this flag to decide
# whether to call ``ik_head.float()`` after building the head.
_ODVLA_IK_HEAD_FP32 = str(os.environ.get("ODVLA_IK_HEAD_FP32", "0")).strip().lower() not in (
    "",
    "0",
    "false",
    "no",
    "off",
)


def _ik_fp32_ctx(device_type: str = "cuda"):
    """autocast(enabled=False) when ODVLA_IK_HEAD_FP32=1, else a no-op context.

    Args:
        device_type: device type string passed to ``torch.autocast`` ("cuda" or
            "cpu"). Only used when the flag is on.

    Returns:
        A context manager. With the flag on, ops inside it run in the dtype of
        their inputs (no fp16/bf16 autocast); with the flag off it is a
        ``contextlib.nullcontext`` and any surrounding autocast stays active.
    """
    # Flag on: disable autocast so the IK head's fp32 weights are used in fp32.
    if _ODVLA_IK_HEAD_FP32:
        return torch.autocast(device_type=device_type, enabled=False)
    # Flag off: leave the caller's autocast state untouched.
    return contextlib.nullcontext()


class _DiffusionSchedule:
    """Stateless helpers shared by diffusion heads (beta schedule, time embedding).

    Both functions are re-exported as static methods of
    :class:`CrossAttentionDiffusionIK` (``_cosine_beta_schedule`` and
    ``_timestep_embedding``).
    """

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, *, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from "Improved Denoising Diffusion Probabilistic Models".

        For very small T (e.g. 10), a linear beta schedule often fails to push
        x_T close to N(0, I), which breaks sampling when starting from pure noise;
        this is the only schedule used by :class:`CrossAttentionDiffusionIK`.

        Implements ``alpha_bar(t) = cos^2(((t / T) + s) / (1 + s) * pi / 2)``,
        normalised so that ``alpha_bar(0) = 1``, and
        ``beta_t = 1 - alpha_bar(t) / alpha_bar(t - 1)`` for ``t = 1..T``.

        Args:
            timesteps: number of diffusion steps T (> 0). The released recipes
                use T = 50 (``--ik-num-steps 50``).
            s: small offset (Nichol & Dhariwal use 0.008) that keeps beta_t from
                being too small near t = 0.

        Returns:
            (T,) float32 tensor of betas, clamped to [1e-4, 0.999].
        """
        # Evaluate alpha_bar on the T + 1 grid points 0, 1, ..., T.
        T = int(timesteps)
        assert T > 0
        steps = T + 1
        x = torch.linspace(0, T, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos((x / T + float(s)) / (1.0 + float(s)) * math.pi * 0.5) ** 2
        # Normalise so alpha_bar(0) == 1 exactly.
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        # Per-step betas from ratios of consecutive alpha_bar values.
        betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        # Lower clamp keeps every step slightly noisy; upper clamp avoids
        # beta == 1 (alpha_bar collapsing to exactly 0) at the last step.
        return betas.clamp(0.0001, 0.999)

    @staticmethod
    def _timestep_embedding(
        timesteps: torch.Tensor, dim: int, max_period: int = 10000
    ) -> torch.Tensor:
        """Create sinusoidal timestep embeddings.

        Frequencies are ``exp(-log(max_period) * i / half)`` for
        ``i = 0..half-1`` with ``half = dim // 2``; the output is
        ``[cos(t * f), sin(t * f)]`` (cosines first), zero-padded by one column
        when ``dim`` is odd.

        Args:
            timesteps: (B,) int64 tensor of diffusion steps.
            dim: embedding dimension.
            max_period: controls the minimum frequency of the embeddings.

        Returns:
            (B, dim) float32 tensor.
        """
        # Accept any shape by flattening to (B,).
        if timesteps.ndim != 1:
            timesteps = timesteps.view(-1)
        # Geometric frequency ladder from 1 down to ~1/max_period.
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32, device=timesteps.device)
            / half
        )
        # Outer product (B, 1) x (1, half) -> (B, half), then [cos | sin] -> (B, 2*half).
        args = timesteps.to(dtype=torch.float32).unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        # Odd dim: append one zero column so the width is exactly `dim`.
        if dim % 2 == 1:
            emb = torch.cat(
                [emb, torch.zeros((emb.shape[0], 1), device=emb.device, dtype=emb.dtype)], dim=-1
            )
        return emb


class CrossAttentionDiffusionIK(nn.Module):
    """Cross-attention + Diffusion IK head.

    Unlike DiffusionIK (an earlier head, not part of this release), which is
    conditioned on a flat cond vector, this module cross-attends over the token
    sequences of the current frame and the next frame, so it denoises while
    keeping their spatial (per-token) structure.

    In the DriveIK forward pass (``model.py``) the two sequences are:

    * ``cur_tokens``: (N, H) current-frame DINOv3 camera tokens projected into the
      LLM hidden space by ``mm_projector_bev`` (N = 540 = 3 front cameras x 180
      patches in the released recipes; H = LLM hidden size, 896 for
      Qwen2.5-0.5B).
    * ``next_tokens_list``: one (N, H) tensor per predicted horizon, i.e. the
      LLM's last hidden states at the ``<BEV_STATE>`` placeholder positions
      (the predicted future visual state). The released recipes use a single
      horizon (8 steps = 4 s ahead).

    Usage:

    * Training: ``diffusion_loss(cur_tokens, next_tokens_list, x0)``
    * Inference: ``sample(cur_tokens, next_tokens_list, num_steps=50)`` -> trajectory
    * GRPO: every ``sample()`` call starts from a fresh ``x_T ~ N(0, I)``, so it is
      naturally stochastic; this is the same approach as DiffGRPO (ReCogDrive,
      arXiv:2506.08052). The GRPO trainer uses :meth:`sample_with_logprob_data`
      and :meth:`compute_logprob_for_chain`.

    Denoising network (one query token = the whole trajectory):
      x_t (B, traj_dim) + t_embed
          -> cross-attn over cur_tokens (N, H)
          -> cross-attn over next_tokens (N, H)
          -> FFN
          -> eps_pred (B, traj_dim)
    Each stage is a post-norm residual block (``LayerNorm(h + f(h))``).

    Trajectory convention: ``x0`` is the flattened trajectory
    ``[x1, y1, ..., xT, yT]`` divided by ``ik_traj_scale`` (meters / 20 by
    default), i.e. it lives in *normalised* units; callers multiply samples by
    the same scale to get meters.

    Attributes set from outside (by ``model.py`` forward, read by ``grpo.py``):
        _grpo_cur_tokens / _grpo_next_tokens_list: detached copies of the last
            forward's conditioning tokens, used to draw GRPO samples and to
            evaluate the reference policy.
        _grpo_cur_tokens_grad / _grpo_next_tokens_list_grad: the same tensors
            without ``detach()`` (they carry grad only if the caller did not
            stop gradients), used to recompute log-probs with gradient.
    """

    # Re-export the shared schedule helpers as static methods of this class.
    _cosine_beta_schedule = staticmethod(_DiffusionSchedule._cosine_beta_schedule)
    _timestep_embedding = staticmethod(_DiffusionSchedule._timestep_embedding)

    def __init__(
        self,
        hidden_size: int,
        traj_dim: int = 12,
        d_model: int = 256,
        num_heads: int = 8,
        num_steps: int = 100,
        dropout: float = 0.0,
    ):
        """Build the denoiser and the (non-persistent) diffusion schedule buffers.

        Args:
            hidden_size: width H of the conditioning tokens (LLM hidden size).
            traj_dim: flattened trajectory size, ``2 * num_waypoints``
                (16 in the released NAVSIM recipes).
            d_model: internal width of the denoiser (``--ik-decoder-dim``, 256).
            num_heads: attention heads of both cross-attention layers
                (``--ik-decoder-heads``, 8). Must divide ``d_model``.
            num_steps: number of training diffusion steps T
                (``--ik-num-steps``; 50 in the released recipes). The betas
                follow the cosine schedule.
            dropout: attention dropout of the cross-attention layers.
        """
        super().__init__()
        # Basic sizes.
        self.traj_dim = int(traj_dim)
        self.num_steps = int(num_steps)
        # Input projections: conditioning tokens H -> d_model, noisy trajectory
        # traj_dim -> d_model (the single query token).
        self.bev_proj = nn.Linear(hidden_size, d_model)
        self.x_proj = nn.Linear(traj_dim, d_model)
        # Timestep embedding: sinusoidal features of width max(d_model // 4, 32)
        # (64 for d_model = 256) followed by a 2-layer SiLU MLP to d_model.
        self.time_emb_dim = max(d_model // 4, 32)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, d_model), nn.SiLU(), nn.Linear(d_model, d_model)
        )
        # Cross-attention over the current-frame tokens (+ post-norm).
        self.cur_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=float(dropout)
        )
        self.cur_norm = nn.LayerNorm(d_model)
        # Cross-attention over the predicted next-frame tokens (+ post-norm).
        self.next_attn = nn.MultiheadAttention(
            d_model, num_heads, batch_first=True, dropout=float(dropout)
        )
        self.next_norm = nn.LayerNorm(d_model)
        # Position-wise FFN with 2x expansion (+ post-norm), then the epsilon head.
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Linear(d_model * 2, d_model)
        )
        self.ffn_norm = nn.LayerNorm(d_model)
        self.eps_head = nn.Linear(d_model, traj_dim)
        # Cosine beta schedule and the cumulative product alpha_bar_t of
        # alpha_t = 1 - beta_t.
        betas = self._cosine_beta_schedule(self.num_steps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        # Register as non-persistent buffers: they follow .to(device) but are not
        # stored in the state_dict (they are recomputed from num_steps on
        # construction).
        self.register_buffer("alphas_cumprod", alphas_cumprod, persistent=False)
        self.register_buffer("sqrt_alphas_cumprod", alphas_cumprod.sqrt(), persistent=False)
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", (1.0 - alphas_cumprod).sqrt(), persistent=False
        )
        # GRPO hand-off slots, filled by the world-model forward pass in
        # model.py on every call and read by grpo.py (see class docstring).
        self._grpo_cur_tokens: Optional[torch.Tensor] = None
        self._grpo_next_tokens_list: Optional[List] = None
        self._grpo_cur_tokens_grad: Optional[torch.Tensor] = None
        self._grpo_next_tokens_list_grad: Optional[List] = None

    @staticmethod
    def _resolve_next_tokens(
        next_tokens_list, device: torch.device, dtype: torch.dtype, fallback: torch.Tensor
    ) -> torch.Tensor:
        """Return a single (N, H) tensor from ``next_tokens_list``.

        If several horizons are present they are averaged; if none is present,
        ``cur_tokens`` (``fallback``) is used instead.

        Args:
            next_tokens_list: list (or None) of predicted next-frame token
                tensors, each (N, H) or (B, N, H). Non-tensors and empty tensors
                are ignored.
            device, dtype: target device/dtype for the returned tokens.
            fallback: returned unchanged (no device/dtype cast) when no valid
                next-frame tensor exists; callers pass ``cur_tokens``.

        Returns:
            (N, H) tensor (or ``fallback`` as-is). All valid horizons must have
            the same (N, H) shape for the stack/mean.

        Note:
            A 3-D tensor with batch > 1 is averaged over its batch dimension, so
            per-sample information is merged. The training collator uses batch
            size 1, so this branch does not occur in the released recipes.
        """
        # Keep only real, non-empty tensors.
        valid = [t for t in next_tokens_list or [] if torch.is_tensor(t) and t.numel() > 0]
        # Nothing predicted -> condition the "next" branch on the current frame.
        if not valid:
            return fallback
        # Reduce every horizon to (N, H): drop a size-1 batch dim, or average
        # over a larger batch dim; then move to the requested device/dtype.
        tensors = []
        for t in valid:
            if t.ndim == 3 and t.shape[0] == 1:
                t = t[0]
            elif t.ndim == 3:
                t = t.mean(0)
            tensors.append(t.to(device=device, dtype=dtype))
        # Single horizon (the released recipes): return it directly.
        if len(tensors) == 1:
            return tensors[0]
        # Multiple horizons: element-wise mean over horizons -> (N, H).
        return torch.stack(tensors, dim=0).mean(dim=0)

    def _denoise(
        self, x_t: torch.Tensor, t: torch.Tensor, cur_kv: torch.Tensor, next_kv: torch.Tensor
    ) -> torch.Tensor:
        """Predict the noise eps with the cross-attention denoising network.

        Args:
            x_t: (B, traj_dim) noisy normalised trajectory at step ``t``.
            t: (B,) int64 diffusion step indices in [0, num_steps).
            cur_kv: (B, N, d_model) projected current-frame tokens (from
                :meth:`_project_tokens`).
            next_kv: (B, N', d_model) projected next-frame tokens.

        Returns:
            (B, traj_dim) epsilon prediction in the dtype of the head parameters.
        """
        # Optional fp32 region (ODVLA_IK_HEAD_FP32); otherwise caller's autocast applies.
        with _ik_fp32_ctx(x_t.device.type if hasattr(x_t.device, "type") else "cuda"):
            # Cast all inputs to the head's parameter dtype (fp32 or the model dtype).
            p_dtype = next(self.x_proj.parameters()).dtype
            x_t = x_t.to(dtype=p_dtype)
            cur_kv = cur_kv.to(dtype=p_dtype)
            next_kv = next_kv.to(dtype=p_dtype)
            # Trajectory -> single query token (B, 1, d_model), plus the
            # MLP-transformed sinusoidal embedding of t (B, 1, d_model).
            h = self.x_proj(x_t).unsqueeze(1)
            te = self.time_mlp(
                self._timestep_embedding(t, self.time_emb_dim).to(device=h.device, dtype=h.dtype)
            ).unsqueeze(1)
            h = h + te
            # Residual cross-attention over the current-frame tokens.
            attn_out, _ = self.cur_attn(h, cur_kv, cur_kv, need_weights=False)
            h = self.cur_norm(h + attn_out)
            # Residual cross-attention over the predicted next-frame tokens.
            attn_out, _ = self.next_attn(h, next_kv, next_kv, need_weights=False)
            h = self.next_norm(h + attn_out)
            # Residual FFN, then project the query token back to traj_dim.
            h = self.ffn_norm(h + self.ffn(h))
            return self.eps_head(h.squeeze(1))

    def _project_tokens(self, cur_tokens: torch.Tensor, next_tokens: torch.Tensor, B: int):
        """Project the visual-state tokens to d_model and match the batch dimension.

        Args:
            cur_tokens: (N, H) or (B, N, H) current-frame tokens.
            next_tokens: (N', H) or (B, N', H) next-frame tokens.
            B: batch size to broadcast 2-D inputs to.

        Returns:
            ``(cur_kv, next_kv)``: (B, N, d_model) and (B, N', d_model) in the
            dtype of ``bev_proj``. Both sequences share the same ``bev_proj``.
        """
        # Device type string for the optional autocast-disable context.
        _pt_dev = (
            cur_tokens.device.type
            if torch.is_tensor(cur_tokens) and hasattr(cur_tokens.device, "type")
            else "cuda"
        )

        def _prep(tok):
            # 2-D (N, H): add a batch dim and broadcast (view, no copy) to B.
            if tok.ndim == 2:
                tok = tok.unsqueeze(0).expand(B, -1, -1)
            # Cast to the projection's weight dtype and apply Linear(H -> d_model).
            return self.bev_proj(tok.to(next(self.bev_proj.parameters()).dtype))

        with _ik_fp32_ctx(_pt_dev):
            cur_kv, next_kv = (_prep(cur_tokens), _prep(next_tokens))
        return (cur_kv, next_kv)

    def diffusion_loss(
        self,
        cur_tokens: torch.Tensor,
        next_tokens_list,
        x0: torch.Tensor,
        *,
        mc: Optional[int] = None,
        weight_xy: Optional[Tuple[float, float]] = None,
    ):
        """DDPM training loss (eps MSE).

        Same interface as ``DiffusionIK.diffusion_loss()``, except that it takes
        ``(cur_tokens, next_tokens_list)`` instead of a flat ``cond`` vector.

        Args:
            cur_tokens: (N, H) or (B0, N, H) current-frame conditioning tokens.
            next_tokens_list: list of predicted next-frame tokens (see
                :meth:`_resolve_next_tokens`).
            x0: (B0, traj_dim) clean target trajectory in normalised units
                (meters / ``ik_traj_scale``), flattened as [x1, y1, ..., xT, yT].
            mc: Monte-Carlo expansion factor: each target is repeated ``mc``
                times with independent (t, noise) draws. ``None`` means 8 when
                B0 == 1 and 1 otherwise. ``model.py`` passes ``ik_diff_mc``
                (``--ik-diff-mc``, default 8).
            weight_xy: optional ``(w_longitudinal, w_lateral)``. Applied to the
                y (odd, forward) and x (even, lateral) components respectively.

        Returns:
            Scalar loss.

        Note:
            The weights multiply the residual ``eps_pred - noise`` *before* it is
            squared, so the squared error of each component is scaled by
            ``w**2`` (e.g. longitudinal weight 10 -> factor 100).
        """
        # Resolve the next-frame tokens to one (N, H) tensor (or fall back to cur).
        device, dtype = (x0.device, x0.dtype)
        next_tokens = self._resolve_next_tokens(next_tokens_list, device, dtype, cur_tokens)
        # Monte-Carlo expansion: repeat each target mc_eff times so a batch of
        # one sample still sees several (t, noise) pairs per optimizer step.
        B0 = int(x0.shape[0])
        mc_eff = (8 if B0 == 1 else 1) if mc is None else max(1, int(mc))
        if mc_eff > 1:
            x0 = x0.repeat_interleave(mc_eff, dim=0)
            # Batched (3-D) tokens are repeated to match. 2-D tokens are later
            # broadcast by _project_tokens. If next_tokens was reduced to 2-D
            # while cur_tokens is 3-D, this repeats it along the token axis
            # (N -> N*mc); duplicated keys leave the attention output unchanged
            # (without dropout) but cost extra compute.
            if cur_tokens.ndim == 3:
                cur_tokens = cur_tokens.repeat_interleave(mc_eff, dim=0)
                next_tokens = next_tokens.repeat_interleave(mc_eff, dim=0)
        # Forward diffusion q(x_t | x0): t ~ U{0, ..., T-1}, eps ~ N(0, I),
        # x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps.
        B = int(x0.shape[0])
        t = torch.randint(0, self.num_steps, (B,), device=device)
        noise = torch.randn_like(x0)
        sqrt_ab = self.sqrt_alphas_cumprod[t].unsqueeze(-1)
        sqrt_1mab = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(-1)
        x_t = sqrt_ab * x0 + sqrt_1mab * noise
        # Predict eps conditioned on the projected visual-state tokens.
        cur_kv, next_kv = self._project_tokens(cur_tokens, next_tokens, B)
        eps_pred = self._denoise(x_t, t, cur_kv, next_kv)
        # Unweighted path: plain MSE between predicted and true noise.
        if weight_xy is None:
            return F.mse_loss(eps_pred, noise)
        diff = eps_pred - noise
        # Per-axis weights: even indices are x (lateral) -> weight_xy[1];
        # odd indices are y (forward / longitudinal) -> weight_xy[0].
        if self.traj_dim % 2 == 0:
            w = x0.new_zeros(self.traj_dim)
            w[0::2] = float(weight_xy[1])
            w[1::2] = float(weight_xy[0])
            diff = diff * w.unsqueeze(0)
        return (diff**2).mean()

    @torch.no_grad()
    def sample(
        self,
        cur_tokens: torch.Tensor,
        next_tokens_list,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        """Deterministic DDIM sampling (eta = 0), run in fp32.

        x_T is re-drawn on every call, so repeated calls give different
        trajectories; the chain itself is deterministic given x_T. (The
        stochastic GRPO sampler is :meth:`sample_with_logprob_data`.)

        Args:
            cur_tokens: (N, H) or (B, N, H) current-frame tokens. Their device and
                dtype define the output device/dtype.
            next_tokens_list: list of predicted next-frame tokens.
            num_steps: number of DDIM steps K (``None`` -> min(50, num_steps)).
                ``model.py`` uses ``ik_debug_sample_steps`` (50, capped at the
                training T) for evaluation.

        Returns:
            (B, traj_dim) sampled trajectory in normalised units, cast to the
            dtype of ``cur_tokens``. Runs under ``torch.no_grad()``.
        """
        # Output device/dtype follow the conditioning tokens (params as fallback).
        device = (
            cur_tokens.device if torch.is_tensor(cur_tokens) else next(self.parameters()).device
        )
        dtype = cur_tokens.dtype if torch.is_tensor(cur_tokens) else next(self.parameters()).dtype
        next_tokens = self._resolve_next_tokens(next_tokens_list, device, dtype, cur_tokens)
        # Ensure a batch dim on the current tokens (2-D next tokens are
        # broadcast later by _project_tokens).
        if cur_tokens.ndim == 2:
            cur_b = cur_tokens.unsqueeze(0)
            next_b = next_tokens.unsqueeze(0)
        else:
            cur_b = cur_tokens
            next_b = next_tokens
        B = cur_b.shape[0]
        # Strided DDIM timestep subsequence, descending: K indices
        # round(i * T / K) for i = K-1, ..., 0 (capped at T-1). With K == T this
        # is simply T-1, ..., 0.
        T_total = self.num_steps
        T_steps = int(num_steps or min(50, T_total))
        step_ratio = T_total / T_steps
        timesteps = list(
            reversed([min(int(round(i * step_ratio)), T_total - 1) for i in range(T_steps)])
        )
        # Draw x_T ~ N(0, I) and project the conditioning tokens once (they are
        # reused at every step), all in fp32.
        x = torch.randn(B, self.traj_dim, device=device, dtype=torch.float32)
        cur_b_f32 = cur_b.float()
        next_b_f32 = next_b.float()
        cur_kv, next_kv = self._project_tokens(cur_b_f32, next_b_f32, B)
        for i, t_idx in enumerate(timesteps):
            # Next (less noisy) index; t_prev == 0 is treated as the clean
            # endpoint (alpha_bar_prev = 1). The subsequence itself ends at
            # t = 0, so the final iteration is one more denoising pass at t = 0.
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
            ac_t = float(self.alphas_cumprod[t_idx])
            ac_prev = float(self.alphas_cumprod[t_prev]) if t_prev > 0 else 1.0
            # eps prediction with autocast disabled (inputs keep their dtype).
            with torch.amp.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda", enabled=False
            ):
                eps = self._denoise(x, t_tensor, cur_kv, next_kv)
            # x0 estimate from eps, clamped to +-10 normalised units (+-200 m at
            # the default 20 m scale).
            x0_pred = (x - math.sqrt(1.0 - ac_t) * eps) / math.sqrt(ac_t)
            x0_pred = x0_pred.clamp(-10.0, 10.0)
            # Deterministic DDIM update (Song et al. 2021, sigma_t = 0), a_* = alpha_bar:
            # x_prev = sqrt(a_prev) * x0_hat + sqrt(1 - a_prev) * eps.
            dir_coef = math.sqrt(max(1.0 - ac_prev, 0.0))
            x = math.sqrt(ac_prev) * x0_pred + dir_coef * eps
        return x.to(dtype)

    @torch.no_grad()
    def sample_with_logprob_data(
        self,
        cur_tokens: torch.Tensor,
        next_tokens_list,
        num_samples: int = 1,
        num_steps: int = 20,
        eta: float = 1.0,
        sigma_min: float = 0.0,
        xt_clip_value: float = 0.0,
    ):
        """Stochastic DDIM sampling. Returns ``(x_0, chain_data)``.

        The chain (x and the projected conditioning tokens) runs in fp32.

        chain_data: List[Dict]  (length = num_steps)
            { 't_idx', 't_prev', 'x_t' (G, traj_dim), 'x_prev' (G, traj_dim),
              'sigma_t' (float), 'x0_clean' (G, traj_dim) }
            Tensors are detached fp32 clones, produced under no_grad. They are reused later by compute_logprob_for_chain().
            ``sigma_t`` is the noise std actually used for the step (after the
            ``sigma_min`` floor); ``x0_clean`` is the noise-free transition mean
            (for the last step this is the clean x0 estimate).

        eta=0  -> deterministic DDIM (log p is undefined, unusable for GRPO)
        eta=1  -> stochastic, close to DDPM
        eta in (0, 1) -> in between

        sigma_min: floor on sigma_t (ReCogDrive sigma_min^exp). 0 = unchanged,
        > 0 = sigma_t clamped to >= sigma_min. Prevents sample collapse caused by
        sigma_t -> 0 in the late DDIM steps. Because the floor is also applied
        to steps whose natural sigma is 0 (t_prev == 0), the returned sample
        carries sigma_min-scaled noise.

        xt_clip_value: +-clip of the prior x_T (and of the per-step injected
        noise). 0 = no clipping (legacy). ReCogDrive uses train=5.0, eval=1.0;
        cutting the tails makes outliers less likely.

        The NAVSIM GRPO recipe calls this with num_samples = group size 4,
        num_steps = 20, eta = 1.0, sigma_min = 0.04, xt_clip_value = 5.0.

        Args:
            cur_tokens: (N, H) or (B, N, H) current-frame tokens.
            next_tokens_list: list of predicted next-frame tokens.
            num_samples: G, number of trajectories to draw per conditioning.
            num_steps: DDIM steps K (``0``/``None`` -> min(20, num_steps)).
            eta, sigma_min, xt_clip_value: see above.

        Returns:
            ``(x, chain_data)`` with x of shape (G, traj_dim) in the dtype of
            ``cur_tokens`` (normalised units).
        """
        # Output device/dtype follow the conditioning tokens.
        device = (
            cur_tokens.device if torch.is_tensor(cur_tokens) else next(self.parameters()).device
        )
        dtype = cur_tokens.dtype if torch.is_tensor(cur_tokens) else next(self.parameters()).dtype
        next_tokens = self._resolve_next_tokens(next_tokens_list, device, dtype, cur_tokens)
        cur_b = cur_tokens.unsqueeze(0) if cur_tokens.ndim == 2 else cur_tokens
        next_b = next_tokens.unsqueeze(0) if next_tokens.ndim == 2 else next_tokens
        # Replicate the conditioning to G rows: broadcast a single sample
        # (expand, no copy), or repeat each of B samples num_samples times
        # (G = B * num_samples).
        if cur_b.shape[0] == 1 and num_samples > 1:
            cur_b = cur_b.expand(num_samples, -1, -1)
            next_b = next_b.expand(num_samples, -1, -1)
        elif cur_b.shape[0] != num_samples:
            cur_b = cur_b.repeat_interleave(num_samples, dim=0)
            next_b = next_b.repeat_interleave(num_samples, dim=0)
        G = int(cur_b.shape[0])
        # Strided, descending DDIM timestep subsequence (see sample()).
        T_total = self.num_steps
        T_steps = int(num_steps or min(20, T_total))
        step_ratio = T_total / T_steps
        timesteps = list(
            reversed([min(int(round(i * step_ratio)), T_total - 1) for i in range(T_steps)])
        )
        # Independent x_T per sample (fp32); project the conditioning once.
        x = torch.randn(G, self.traj_dim, device=device, dtype=torch.float32)
        cur_kv, next_kv = self._project_tokens(cur_b.float(), next_b.float(), G)
        # Optional truncation of the prior.
        if xt_clip_value > 0.0:
            x = x.clamp(-float(xt_clip_value), float(xt_clip_value))
        chain_data: List[Dict[str, Any]] = []
        for i, t_idx in enumerate(timesteps):
            # Step indices / alpha_bar values (t_prev == 0 -> alpha_bar_prev = 1).
            t_prev = timesteps[i + 1] if i + 1 < len(timesteps) else 0
            t_tensor = torch.full((G,), t_idx, device=device, dtype=torch.long)
            ac_t = float(self.alphas_cumprod[t_idx])
            ac_prev = float(self.alphas_cumprod[t_prev]) if t_prev > 0 else 1.0
            # eps prediction, autocast disabled, result in fp32.
            with torch.amp.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda", enabled=False
            ):
                eps = self._denoise(x, t_tensor, cur_kv, next_kv).float()
            # Clamped x0 estimate (sqrt(alpha_bar) floored at 1e-6).
            x0_pred = (x - math.sqrt(1.0 - ac_t) * eps) / max(math.sqrt(ac_t), 1e-06)
            x0_pred = x0_pred.clamp(-10.0, 10.0)
            # DDIM sigma_t (unlike sample(), computed for any eta; it is 0 at eta = 0).
            if t_prev > 0:
                sigma_t = float(
                    eta
                    * math.sqrt(max((1.0 - ac_prev) / max(1.0 - ac_t, 1e-20), 0.0))
                    * math.sqrt(max(1.0 - ac_t / max(ac_prev, 1e-20), 0.0))
                )
            else:
                sigma_t = 0.0
            # Noise floor (also applied at t_prev == 0).
            if sigma_min > 0.0:
                sigma_t = max(sigma_t, float(sigma_min))
            # Gaussian transition p(x_prev | x_t) = N(mean_x_prev, sigma_t^2 I).
            dir_coef = math.sqrt(max(1.0 - ac_prev - sigma_t * sigma_t, 0.0))
            mean_x_prev = math.sqrt(ac_prev) * x0_pred + dir_coef * eps
            if sigma_t > 0:
                noise = torch.randn_like(x)
                if xt_clip_value > 0.0:
                    noise = noise.clamp(-float(xt_clip_value), float(xt_clip_value))
                x_prev = mean_x_prev + sigma_t * noise
            else:
                x_prev = mean_x_prev
            # Record the transition so its log-prob can be recomputed with
            # gradient (and under the reference policy) later.
            chain_data.append(
                {
                    "t_idx": int(t_idx),
                    "t_prev": int(t_prev),
                    "x_t": x.detach().clone(),
                    "x_prev": x_prev.detach().clone(),
                    "sigma_t": float(sigma_t),
                    "x0_clean": mean_x_prev.detach().clone(),
                }
            )
            x = x_prev
        return (x.to(dtype), chain_data)

    def compute_logprob_for_chain(
        self,
        cur_tokens: torch.Tensor,
        next_tokens_list,
        chain_data: List[Dict[str, Any]],
        return_means: bool = False,
        sigma_min_logprob: float = 0.0,
        gamma_denoising: float = 1.0,
    ):
        """Compute sum_t log p_theta(x_{t-1} | x_t, cond) over a stored sampling chain, with gradient.

        Each stored transition is scored under the *current* parameters as an
        isotropic Gaussian ``N(mu_theta(x_t, t, c), sigma^2 I)`` over the
        traj_dim components:
        ``log p = -0.5 * ||x_prev - mu||^2 / sigma^2 - D * log(sigma)``
        (the constant ``-D/2 * log(2*pi)`` is dropped; D = traj_dim). The mean
        ``mu`` is rebuilt exactly as in :meth:`sample_with_logprob_data`, using
        the stored ``sigma_t`` for the DDIM direction coefficient.

        Args:
            cur_tokens: (N, H) or (B, N, H) current-frame tokens. The GRPO
                trainer passes ``_grpo_cur_tokens_grad`` (falling back to the
                detached copy), and the reference policy's ``_grpo_cur_tokens``.
            next_tokens_list: list of predicted next-frame tokens.
            chain_data: output of :meth:`sample_with_logprob_data`.
            return_means, sigma_min_logprob, gamma_denoising: see below.

        Returns:
            return_means=False: (G,) tensor -- per-sample sum of log-probs over the chain
            return_means=True : (log_p, means, sigmas)
                means  = List[(G, traj_dim)]  -- mu_theta(x_t, t, c) at each scored step
                         (sigma_t > 0, or every step when sigma_min_logprob > 0)
                sigmas = List[float]          -- the sigma used at the same steps
            (used to compare against the reference policy in "KL Option B")
            An empty chain returns a scalar zero (and empty lists).

        ReCogDrive-recommended options (Phase 2):
            sigma_min_logprob   -- floor on the sigma in the log p denominator (separate
                                   from the sampling sigma_min). When > 0, steps with
                                   sigma_t <= 0 are also evaluated with sigma_min_logprob
                                   (not skipped).
            gamma_denoising     -- per-step log p weight gamma^(K-1-i), i = chain index
                                   (0 = noisiest step). 1.0 = uniform (legacy);
                                   0.6 (ReCogDrive) = larger weight on later steps.

        The NAVSIM GRPO recipe uses sigma_min_logprob = 0.05 and
        gamma_denoising = 0.6.
        """
        # Empty chain: nothing to score.
        if not chain_data:
            zero = torch.zeros(
                (), device=next(self.parameters()).device, dtype=next(self.parameters()).dtype
            )
            if return_means:
                return (zero, [], [])
            return zero
        # Group size and device come from the stored chain.
        device = chain_data[0]["x_t"].device
        G = int(chain_data[0]["x_t"].shape[0])
        next_tokens = self._resolve_next_tokens(
            next_tokens_list, device, cur_tokens.dtype, cur_tokens
        )
        # Replicate the conditioning to G rows exactly as during sampling.
        cur_b = cur_tokens.unsqueeze(0) if cur_tokens.ndim == 2 else cur_tokens
        next_b = next_tokens.unsqueeze(0) if next_tokens.ndim == 2 else next_tokens
        if cur_b.shape[0] == 1 and G > 1:
            cur_b = cur_b.expand(G, -1, -1)
            next_b = next_b.expand(G, -1, -1)
        elif cur_b.shape[0] != G:
            cur_b = cur_b.repeat_interleave(G, dim=0)
            next_b = next_b.repeat_interleave(G, dim=0)
        # Project once in fp32 (with grad); accumulate log p in fp32.
        cur_kv, next_kv = self._project_tokens(cur_b.float(), next_b.float(), G)
        log_p = cur_tokens.new_zeros(G, dtype=torch.float32)
        means_list: List[torch.Tensor] = [] if return_means else None
        sigmas_list: List[float] = [] if return_means else None
        D = float(self.traj_dim)
        K_chain = len(chain_data)
        for chain_idx, entry in enumerate(chain_data):
            # Choose the std for the likelihood. Without a floor, deterministic
            # steps (sigma = 0) have no density and are skipped.
            sigma_t_natural = float(entry["sigma_t"])
            if sigma_min_logprob > 0.0:
                sigma_t_eff = max(sigma_t_natural, sigma_min_logprob)
            else:
                if sigma_t_natural <= 0.0:
                    continue
                sigma_t_eff = sigma_t_natural
            # Restore the stored transition (x_t -> x_prev) in fp32.
            t_idx = int(entry["t_idx"])
            t_prev = int(entry["t_prev"])
            x_t = entry["x_t"].to(device=device, dtype=torch.float32)
            x_prev_target = entry["x_prev"].to(device=device, dtype=torch.float32)
            ac_t = float(self.alphas_cumprod[t_idx])
            ac_prev = float(self.alphas_cumprod[t_prev]) if t_prev > 0 else 1.0
            t_tensor = torch.full((G,), t_idx, device=device, dtype=torch.long)
            # Re-run the denoiser with gradient (autocast disabled).
            with torch.amp.autocast(
                device_type=device.type if hasattr(device, "type") else "cuda", enabled=False
            ):
                eps = self._denoise(x_t, t_tensor, cur_kv, next_kv).float()
            # Rebuild the transition mean mu_theta exactly as in sampling.
            x0_pred = (x_t - math.sqrt(1.0 - ac_t) * eps) / max(math.sqrt(ac_t), 1e-06)
            x0_pred = x0_pred.clamp(-10.0, 10.0)
            dir_coef = math.sqrt(max(1.0 - ac_prev - sigma_t_natural * sigma_t_natural, 0.0))
            mean_x_prev = math.sqrt(ac_prev) * x0_pred + dir_coef * eps
            # Isotropic Gaussian log-density per sample (constant term dropped).
            sq = (x_prev_target - mean_x_prev).pow(2).sum(dim=-1)
            log_p_step = -0.5 * sq / (sigma_t_eff * sigma_t_eff) - D * math.log(sigma_t_eff)
            # Discounted accumulation: the last step has weight 1, earlier steps
            # gamma, gamma^2, ...; uniform sum when gamma == 1.
            if gamma_denoising != 1.0:
                step_weight = gamma_denoising ** (K_chain - chain_idx - 1)
                log_p = log_p + step_weight * log_p_step
            else:
                log_p = log_p + log_p_step
            # Keep per-step means/stds for the KL term against the reference.
            if return_means:
                means_list.append(mean_x_prev)
                sigmas_list.append(sigma_t_eff)
        if return_means:
            return (log_p, means_list, sigmas_list)
        return log_p


class CameraConditionedQueryPool(nn.Module):
    """4 learnable queries: Q0/Q1/Q2 attend to CAM_FRONT / CAM_FRONT_LEFT / CAM_FRONT_RIGHT
    only, Q3 attends to all 540 DINO2D tokens. Token layout:
        [0:180] CAM_FRONT_LEFT | [180:360] CAM_FRONT | [360:540] CAM_FRONT_RIGHT
    Output: (B, 4, H) per-query features (NOT flattened).

    ``train.py`` attaches it as ``model.ik_attn_pool``. The world-model forward
    pass feeds the (B, 4, H) output to :class:`ObjectQueryHead`, whose
    per-query targets (built in ``data.py``) are constant "no object" targets
    for NAVSIM, whose infos carry no GT boxes. All four queries share the same
    attention projections. If the sequence length is not
    ``3 * patches_per_cam``, every query falls back to attending over all tokens.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        patches_per_cam: int = 180,
        dropout: float = 0.0,
    ):
        """Create the 4 learnable queries and the shared attention layer.

        Args:
            hidden_size: token width H (the LLM hidden size).
            num_heads: attention heads (must divide ``hidden_size``).
            patches_per_cam: DINOv3 patch tokens per camera (180).
            dropout: attention dropout (applied only in training mode).
        """
        super().__init__()
        # Geometry of the token sequence and the fixed number of queries.
        self.ppc = int(patches_per_cam)
        self.hidden_size = int(hidden_size)
        self.num_queries = 4
        # Learnable queries (1, 4, H), small-std (0.02) random init.
        self.query = nn.Parameter(torch.randn(1, 4, hidden_size) * 0.02)
        # One MHA module whose weights are shared by all queries / camera slices.
        self.attn = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Pool the camera tokens into 4 per-query features.

        Args:
            x: (N, H) or (B, N, H) tokens (current-frame projected DINOv3 tokens
                or predicted ``<BEV_STATE>`` hidden states).

        Returns:
            (B, 4, H) LayerNorm-ed query features in the dtype of ``x``. The
            attention itself always runs in fp32 with autocast disabled.
        """
        # Accept unbatched input.
        if x.ndim == 2:
            x = x.unsqueeze(0)
        B, N, H = x.shape
        # Per-camera routing needs exactly 3 * ppc tokens; otherwise all queries
        # attend globally.
        _all_global = N != 3 * self.ppc
        q = self.query.expand(B, -1, -1)
        device_type = "cuda" if x.is_cuda else "cpu"
        with torch.amp.autocast(device_type, enabled=False):
            # fp32 copies of the MHA weights so the attention runs in full
            # precision regardless of the module / model dtype.
            iw = self.attn.in_proj_weight.float()
            ib = self.attn.in_proj_bias.float() if self.attn.in_proj_bias is not None else None
            ow = self.attn.out_proj.weight.float()
            ob = self.attn.out_proj.bias.float() if self.attn.out_proj.bias is not None else None
            dp = self.attn.dropout if self.training else 0.0

            def _mha_fp32(q_in, kv_in):
                # Functional MHA expects sequence-first (L, B, E) tensors.
                # Positional args after the weights: bias_k=None, bias_v=None,
                # add_zero_attn=False, dropout_p=dp, then the output projection.
                q_t = q_in.float().transpose(0, 1)
                kv_t = kv_in.float().transpose(0, 1)
                out, _ = F.multi_head_attention_forward(
                    q_t,
                    kv_t,
                    kv_t,
                    self.attn.embed_dim,
                    self.attn.num_heads,
                    iw,
                    ib,
                    None,
                    None,
                    False,
                    dp,
                    ow,
                    ob,
                    training=self.training,
                    need_weights=False,
                )
                # Back to batch-first (B, L_q, E).
                return out.transpose(0, 1)

            if _all_global:
                # Fallback: all 4 queries attend over the whole sequence.
                out = _mha_fp32(q, x)
            else:
                # Split the 540 tokens into the three 180-token camera blocks.
                kv_FL = x[:, : self.ppc]
                kv_F = x[:, self.ppc : 2 * self.ppc]
                kv_FR = x[:, 2 * self.ppc : 3 * self.ppc]
                # Query -> camera routing: Q0 -> FRONT, Q1 -> FRONT_LEFT,
                # Q2 -> FRONT_RIGHT (each a separate attention call over its block).
                kv_per = [kv_F, kv_FL, kv_FR]
                outs = [_mha_fp32(q[:, qi : qi + 1], kv_per[qi]) for qi in range(3)]
                # Q3 attends over all three cameras.
                outs.append(_mha_fp32(q[:, 3:4], x))
                out = torch.cat(outs, dim=1)
        # Cast back to the input dtype and normalise -> (B, 4, H).
        out = out.to(x.dtype)
        return self.ln(out)


class ObjectQueryHead(nn.Module):
    """Per-query heads: regress 6d BEV box + object presence logit.

    The 6-d box target layout (``query_box_targets`` built in ``data.py``) is
    ``(cx, cy, length, width, sin_yaw, cos_yaw)`` in the ego BEV frame (meters).
    ``model.py`` trains it with BCE-with-logits on presence and an L1 box loss
    masked to queries that have an object, on both the current-frame query
    features and the predicted next-frame query features.
    """

    def __init__(self, hidden_size: int, box_dim: int = 6):
        """Build the box MLP (H -> H -> box_dim, GELU) and the presence logit layer.

        Args:
            hidden_size: query feature width H.
            box_dim: box regression size (6).
        """
        super().__init__()
        self.box = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, box_dim)
        )
        self.obj = nn.Linear(hidden_size, 1)

    def forward(self, q_feats: torch.Tensor):
        """Apply both heads to per-query features.

        Args:
            q_feats: (..., H) query features, e.g. (B, 4, H).

        Returns:
            ``(box, obj_logit)``: (..., box_dim) box regression and (...,)
            presence logits (pre-sigmoid).
        """
        return (self.box(q_feats), self.obj(q_feats).squeeze(-1))
