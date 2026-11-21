"""Shared setup for the supported world-model training recipes.

``main`` is the single place where every DriveIK run is assembled. It is reached

* from ``scripts/train_bev_latent_action_world_model.py`` for the NAVSIM SFT
  stage (world model + IK), with the default trainer
  ``trainer.LatentActionDebugTrainer``;
* from ``grpo.run`` (``scripts/train_bev_latent_action_world_model_grpo.py``) for
  the NAVSIM RFT stage, with ``trainer_cls`` set to the GRPO trainer;
* from ``scripts/evaluate_checkpoint.py`` (``EVAL_CKPT=...``), which first
  replaces ``transformers.Trainer.train`` with an evaluation call and
  ``Trainer.save_model`` with a no-op, so the exact same setup is reused for
  inference / evaluation.

The launcher ``scripts/train_common.sh`` provides the per-recipe flags.

What ``main`` does, in order:

1. Environment / distributed setup, DeepSpeed config, resume-checkpoint lookup.
2. Tokenizer loading (with a ``vocab.json`` fallback) and pad-token fallback.
3. Loading the Qwen2.5 LLaVA backbone (``LlavaQwenForCausalLM``), adding the
   DriveIK special tokens and checking that the BEV marker tokens are single
   tokens.
4. Writing the world-model hyper-parameters onto the model instance (BEV
   horizon(s), loss weights, IK trajectory scale, ...).
5. Setting up the visual input: the frozen DINOv3 camera encoder
   (``vision.CameraEncoder``, attached as ``model.camera_encoder``); then building /
   restoring the visual projector.
6. Attaching the IK modules: the camera-conditioned query pool over visual
   tokens (``CameraConditionedQueryPool`` + ``ObjectQueryHead``) and the
   ``CrossAttentionDiffusionIK`` head, and warm-starting them from the
   checkpoint when possible.
7. Binding ``BEVLatentActionWorldModelMixin`` methods (including ``forward``)
   onto the model instance and enabling gradient checkpointing.
8. Resolving the ego-status feature width (``--ego-can-bus-dim``,
   ``config.ego_base_feature_dim``, legacy fallbacks) and widening the ego
   projector by the step-conditioning channel.
9. Building the train / validation datasets and the collator,
   ``TrainingArguments`` and the trainer, plus callbacks (progress bar,
   curriculum with its seeded epoch sampler, ``ik_mode.json`` writer, optional
   DDP static graph).
10. Patching DeepSpeed's ``load_checkpoint``, running ``trainer.train`` (resuming
    if a checkpoint was found) and saving the final model.

Note on fp16: the SFT recipe trains with ``--fp16`` (fp32 master weights +
``GradScaler``). The loss scaler skips the first ~10 optimizer steps while it
calibrates, so ``grad_norm`` is logged as ``nan`` and the learning rate stays 0
for those steps. This is expected.
"""

from __future__ import annotations
import os
import sys
import torch
import torch.nn as nn
from transformers import AutoTokenizer, TrainingArguments
from transformers.trainer_callback import ProgressCallback, TrainerCallback
from llava.constants import DEFAULT_BEV_START_TOKEN, DEFAULT_BEV_END_TOKEN, DRIVEVLA_SPECIAL_TOKENS
from llava.model import LlavaQwenForCausalLM
from llava.utils import rank0_print
from . import common as wm

# Imported for its side effects only: compatibility patches for the installed
# PyTorch / Transformers / Accelerate / DeepSpeed versions (Trainer ``tokenizer=``
# kwarg, TrainerState JSON loading, torch.load default, ...).
from . import compat
from .config import build_argparser
from .checkpoint import (
    _read_ckpt_ik_head_type,
    _resolve_ckpt_dir_for_weight_loading,
    _restore_ik_head_weights_from_ckpt,
    _restore_projector_weights_from_ckpt,
)
from .data import (
    BEVLatentActionWorldModelDataset,
    DEFAULT_BEV_STATE_TOKEN,
    DataCollatorForLatentActionWM,
    _compute_effective_max_len,
    ensure_mm_projectors,
)
from .heads import (
    CameraConditionedQueryPool,
    CrossAttentionDiffusionIK,
    ObjectQueryHead,
    _ODVLA_IK_HEAD_FP32,
)
from .model import BEVLatentActionWorldModelMixin
from .vision import CameraEncoder
from .trainer import (
    CurriculumEpochSampler,
    LatentActionDebugTrainer,
    TwoStageCurriculumCallback,
)


def main(args=None, trainer_cls=None) -> None:
    """Build the model, data and trainer for one DriveIK recipe and run it.

    Args:
        args: Parsed ``argparse.Namespace`` from ``config.build_argparser()``
            (plus ``config.add_grpo_args`` for GRPO). If ``None``, the command line
            is parsed here. Some fields are mutated in place
            (``gradient_checkpointing`` when ``ODVLA_GC=1``,
            ``ddp_find_unused_parameters`` when left on auto).
        trainer_cls: Trainer class to instantiate. Defaults to
            ``LatentActionDebugTrainer``; ``grpo.run`` passes its GRPO trainer
            (which inherits from ``LatentActionWMTrainer`` and
            ``LatentActionDebugTrainer``). It must accept the ``DebugTrainer``
            keyword arguments (``debug_print_steps``, ``debug_one_batch``, ...).

    Returns:
        None. Side effects: trains (or, under ``scripts/evaluate_checkpoint.py``,
        evaluates) the model, writes checkpoints, ``ik_mode.json`` files,
        tokenizer files and the final model to ``args.output_dir``, and globally
        monkey-patches a few library functions (see the comments below).

    Raises:
        RuntimeError: tokenizer cannot be loaded, fp16 trainable parameters are
            found, tokenizer / embedding vocab sizes differ, or the BEV marker
            tokens are not single tokens.
        ValueError: the ego projector width matches neither the raw ego width
            nor raw width + 1 (step-conditioning channel).
        FileNotFoundError: (from helpers) missing resume checkpoint, DeepSpeed
            config or camera images.
    """
    import transformers.trainer as tr
    import transformers.utils.import_utils as iu

    # Disable transformers' torch.load safety gate (which refuses to call
    # torch.load on older torch versions). Trainer checkpoints here hold RNG /
    # NumPy state and are loaded with weights_only=False (see compat.py), i.e.
    # they are treated as trusted files.
    tr.check_torch_load_is_safe = lambda *args, **kwargs: None
    iu.check_torch_load_is_safe = lambda *args, **kwargs: None
    # Narrow default terminal width for tqdm progress bars (only if unset).
    os.environ.setdefault("COLUMNS", "50")
    os.environ.setdefault("TQDM_NCOLS", "50")
    # Parse the command line unless the caller already did.
    args = args if args is not None else build_argparser().parse_args()
    # ODVLA_GC=1 forces gradient checkpointing on regardless of the CLI flag.
    if os.environ.get("ODVLA_GC", "") == "1":
        args.gradient_checkpointing = True
    # Distributed info from env vars (torchrun) or --local_rank (legacy
    # launchers), read before torch.distributed is initialised. Export LOCAL_RANK
    # when only the CLI argument provided it, so downstream code sees it.
    dist_info = wm._get_dist_info(getattr(args, "local_rank", -1))
    if dist_info.get("local_rank", -1) >= 0 and os.environ.get("LOCAL_RANK", "") == "":
        os.environ["LOCAL_RANK"] = str(dist_info["local_rank"])
    # --ddp-find-unused-parameters left on AUTO (None): enable it for multi-GPU
    # runs only.
    if getattr(args, "ddp_find_unused_parameters", None) is None:
        args.ddp_find_unused_parameters = bool(dist_info.get("world_size", 1) > 1)
    wm.rank0_print(
        f"[DIST] world_size={dist_info.get('world_size', 1)} rank={dist_info.get('rank', 0)} local_rank={dist_info.get('local_rank', -1)} | ddp_find_unused_parameters={args.ddp_find_unused_parameters}"
    )
    # DeepSpeed (ZeRO-3) must register its HF config before from_pretrained so the
    # model is sharded at construction time. No-op without --deepspeed.
    wm._maybe_init_deepspeed(getattr(args, "deepspeed", None))
    # Resolve --resume-from-checkpoint ("auto"/"latest"/"last" -> newest
    # checkpoint-* in output_dir; any other value must be an existing path).
    # The launcher passes "auto" only when output_dir already has checkpoints.
    resume_ckpt = wm._resolve_resume_from_checkpoint(args.resume_from_checkpoint, args.output_dir)
    # Warn about partial resumes (weights or tokenizer taken from --model-path
    # while optimizer/scheduler state comes from the checkpoint).
    if resume_ckpt:
        rank0_print(f"[RESUME] resume_from_checkpoint={resume_ckpt}")
        if not bool(args.resume_load_model):
            rank0_print(
                "[RESUME][WARN] --no-resume-load-model: model weights will be loaded from --model-path, but optimizer/scheduler will resume from checkpoint. This is usually NOT an exact resume."
            )
        if not bool(args.resume_load_tokenizer):
            rank0_print(
                "[RESUME][WARN] --no-resume-load-tokenizer: tokenizer will be loaded from --model-path, but exact resume requires tokenizer/model vocab to match the checkpoint."
            )
    # Temporal infos pkl (token -> prev/next links) of the training split.
    ann_file = str(args.ann_file)
    rank0_print(f"[INFO] Using ann_file: {ann_file}")
    # Visual input: raw camera images (--sensor-root), encoded in the model forward
    # by a frozen DINOv3. Checked before the model is loaded so a wrong call fails
    # fast.
    if not args.sensor_root:
        raise SystemExit("[FATAL] --sensor-root (camera image directory) is required.")
    # ------------------------------------------------------------------
    # Tokenizer
    # ------------------------------------------------------------------
    # Load the tokenizer from the resume checkpoint (exact resume) or from
    # --model-path (base Qwen model / SFT checkpoint).
    tok_src = resume_ckpt if resume_ckpt and bool(args.resume_load_tokenizer) else args.model_path

    def _search_for_vocab_file():
        """Search for vocab.json in all reachable model dirs (same tokenizer everywhere).

        Candidates, in order: ``tok_src`` and its parent directory,
        ``args.output_dir``, ``args.model_path``, then every entry of
        ``./outputs/`` (sorted). Non-directories and duplicates are skipped.

        Returns:
            Path of the first ``vocab.json`` found, or ``None``.
        """
        # Collect candidate directories.
        candidates = []
        if os.path.isdir(str(tok_src)):
            candidates += [str(tok_src), os.path.dirname(str(tok_src))]
        if getattr(args, "output_dir", None):
            candidates.append(str(args.output_dir))
        if getattr(args, "model_path", None) and os.path.isdir(str(args.model_path)):
            candidates.append(str(args.model_path))
        outputs_root = os.path.join(os.getcwd(), "outputs")
        if os.path.isdir(outputs_root):
            for name in sorted(os.listdir(outputs_root)):
                candidates.append(os.path.join(outputs_root, name))
        # Return the first existing vocab.json.
        seen = set()
        for d in candidates:
            if d in seen or not os.path.isdir(d):
                continue
            seen.add(d)
            p = os.path.join(d, "vocab.json")
            if os.path.isfile(p):
                return p
        return None

    def _load_tokenizer(src):
        """Load the slow (``use_fast=False``) tokenizer from ``src``.

        If the plain load raises ``TypeError`` (as happens when the saved
        ``tokenizer_config.json`` has ``vocab_file=null`` and no vocab file sits
        next to it), retry with an explicit ``vocab_file`` / ``merges_file`` found by
        ``_search_for_vocab_file`` and copy those files into ``args.output_dir``
        (if it exists and lacks them) so later resumes find them.

        Args:
            src: Checkpoint directory or HF hub id.

        Returns:
            The tokenizer.

        Raises:
            RuntimeError: No ``vocab.json`` could be found for the fallback.
        """
        # Normal path.
        try:
            return AutoTokenizer.from_pretrained(src, use_fast=False, trust_remote_code=True)
        except TypeError:
            pass
        # Fallback: locate vocab.json (+ merges.txt in the same directory).
        vocab_path = _search_for_vocab_file()
        if vocab_path is None:
            raise RuntimeError(
                f"Cannot load tokenizer from {src}: tokenizer_config.json has vocab_file=null and vocab.json was not found anywhere under outputs/. Copy a vocab.json+merges.txt from any Qwen2.5 model into {src}."
            )
        merges_path = os.path.join(os.path.dirname(vocab_path), "merges.txt")
        if not os.path.isfile(merges_path):
            merges_path = None
        rank0_print(f"[TOKENIZER] Using vocab_file={vocab_path} (vocab_file=null in config)")
        extra = {"vocab_file": vocab_path}
        if merges_path:
            extra["merges_file"] = merges_path
        tok = AutoTokenizer.from_pretrained(src, use_fast=False, trust_remote_code=True, **extra)
        # Persist the vocab/merges files into output_dir for future resumes.
        out_dir = getattr(args, "output_dir", None)
        if out_dir and os.path.isdir(str(out_dir)):
            import shutil

            for fname in ("vocab.json", "merges.txt"):
                src_f = os.path.join(os.path.dirname(vocab_path), fname)
                dst_f = os.path.join(str(out_dir), fname)
                if os.path.isfile(src_f) and (not os.path.isfile(dst_f)):
                    shutil.copy2(src_f, dst_f)
                    rank0_print(f"[TOKENIZER] Saved {fname} to {out_dir} for future resumes")
        return tok

    tokenizer = _load_tokenizer(tok_src)
    # Make sure a pad token exists (eos, then unk, then bos) and that its id is
    # consistent with the token string.
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        elif tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.bos_token is not None:
            tokenizer.pad_token = tokenizer.bos_token
    if tokenizer.pad_token is not None:
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    # ------------------------------------------------------------------
    # Backbone model
    # ------------------------------------------------------------------
    # Weight dtype: bf16 runs keep bf16 weights; fp16 runs keep fp32 master
    # weights (AMP autocasts to fp16 and GradScaler requires fp32 trainable
    # parameters); otherwise fp32.
    if bool(args.bf16):
        model_dtype = torch.bfloat16
    elif bool(args.fp16):
        model_dtype = torch.float32
    else:
        model_dtype = torch.float32
    # Weights come from the resume checkpoint (exact resume) or --model-path.
    # Note: for the second stage (NAVSIM RFT) and for EVAL_CKPT the
    # launcher points --model-path at the previous / evaluated checkpoint.
    model_src = resume_ckpt if resume_ckpt and bool(args.resume_load_model) else args.model_path
    base_model = LlavaQwenForCausalLM.from_pretrained(
        model_src,
        torch_dtype=model_dtype,
        device_map=None,
        low_cpu_mem_usage=False,
        trust_remote_code=True,
        attn_implementation=args.attn_implementation,
    )
    rank0_print(f"Loaded LLaVA model: {model_src}")
    rank0_print(f"Model Class: {base_model.__class__.__name__}")
    # Remember the training precision; written to ik_mode.json as "train_dtype".
    base_model._trained_bf16 = bool(args.bf16)
    base_model._trained_fp16 = bool(args.fp16) and (not bool(args.bf16))
    # If any parameters/buffers are still on the "meta" device after loading,
    # materialise them on CPU (in model_dtype) to avoid "Cannot copy out of meta
    # tensor" errors later. Any failure here is reported and skipped.
    try:
        from llava.model.builder import _materialize_meta_tensors

        meta_names = [
            n
            for (n, p) in base_model.named_parameters(recurse=True)
            if getattr(p, "is_meta", False)
        ]
        meta_names += [
            n for (n, b) in base_model.named_buffers(recurse=True) if getattr(b, "is_meta", False)
        ]
        if len(meta_names) > 0:
            rank0_print(
                f"[WARN] Found {len(meta_names)} meta tensors after loading. Materializing them on CPU to avoid 'Cannot copy out of meta tensor'. Examples: {meta_names[:8]}"
            )
            _materialize_meta_tensors(base_model, device="cpu", dtype=model_dtype)
    except Exception as e:
        rank0_print(f"[WARN] meta tensor check/materialize skipped: {type(e).__name__}: {e}")
    # fp16 AMP sanity check: GradScaler cannot unscale fp16 gradients, so every
    # trainable parameter must be fp32 (guaranteed by model_dtype above unless the
    # checkpoint forced fp16 somewhere).
    if args.fp16 and (not args.bf16):
        bad = [
            (n, p.dtype)
            for (n, p) in base_model.named_parameters()
            if p.requires_grad and p.dtype == torch.float16
        ]
        if len(bad) > 0:
            raise RuntimeError(f"Found fp16 trainable params (GradScaler will crash): {bad[:10]}")
    # ------------------------------------------------------------------
    # Special tokens
    # ------------------------------------------------------------------
    # DriveVLA prompt / answer tags plus <BEV_STATE> (de-duplicated, order kept).
    extra_tokens = list(dict.fromkeys(list(DRIVEVLA_SPECIAL_TOKENS) + [DEFAULT_BEV_STATE_TOKEN]))
    # Fresh runs (including starting from an SFT checkpoint): register every tag
    # that is not yet a single token and resize the embeddings. On resume the
    # tokenizer/model come from the checkpoint and are left untouched.
    if resume_ckpt is None:
        added = wm._add_special_tokens_and_resize(
            tokenizer, base_model, extra_tokens, allow_add=True
        )
        if len(added) > 0:
            rank0_print(f"[TOKENIZER] added {len(added)} special tokens (examples: {added[:10]})")
    # Vocabulary consistency: tokenizer size must equal the number of embedding
    # rows, otherwise token ids and embeddings are misaligned.
    tok_n = len(tokenizer)
    emb_n = int(base_model.get_model().embed_tokens.weight.shape[0])
    cfg_n = getattr(base_model.config, "vocab_size", None)
    ckpt_cfg_n = wm._read_ckpt_vocab_size(resume_ckpt) if resume_ckpt else None
    rank0_print(
        f"[VOCAB] tokenizer_len={tok_n} | model_embed_vocab={emb_n} | config.vocab_size={cfg_n} | ckpt_config.vocab_size={ckpt_cfg_n}"
    )
    if tok_n != emb_n:
        raise RuntimeError(
            f"[FATAL] tokenizer_len != model embedding vocab size.\n  tokenizer_len={tok_n}\n  model_embed_vocab={emb_n}\n  ckpt_config_vocab_size={ckpt_cfg_n}\nExact resume requires all of these to match. This usually means tokenizer/model were not loaded from the same checkpoint."
        )
    # <bev_start>, <bev_end> and <BEV_STATE> MUST be single tokens: masking the
    # BEV span out of the LM loss and locating the <BEV_STATE> positions for the
    # next-visual-state MSE both compare token ids for equality. (The error message
    # below says the same in Korean: the checkpoint tokenizer must contain these
    # tokens as special tokens.)
    must_single = [DEFAULT_BEV_START_TOKEN, DEFAULT_BEV_END_TOKEN, DEFAULT_BEV_STATE_TOKEN]
    bad = [t for t in must_single if not wm._is_single_token(tokenizer, t)]
    if len(bad) > 0:
        raise RuntimeError(
            f"[FATAL] Required BEV tokens are not single-token in the loaded tokenizer: {bad}\nBEV span masking + <BEV_STATE> 위치 MSE가 id equality에 의존합니다. Checkpoint tokenizer가 이 토큰들을 special token으로 포함해야 합니다."
        )
    # Other tags are allowed to be multi-token on resume (exact resume still
    # works, the tags are just split into sub-words); warn only.
    warn = [
        t for t in extra_tokens if t not in must_single and (not wm._is_single_token(tokenizer, t))
    ]
    if resume_ckpt and len(warn) > 0:
        rank0_print(
            f"[RESUME][WARN] {len(warn)} tokens are not single-token (examples: {warn[:10]}). Exact resume는 가능하지만, 해당 태그들은 subword로 쪼개져 토크나이즈됩니다."
        )
    # ------------------------------------------------------------------
    # World-model hyper-parameters stored on the model instance (read by the
    # bound BEVLatentActionWorldModelMixin.forward)
    # ------------------------------------------------------------------
    # Token ids used for BEV span masking and <BEV_STATE> lookup.
    base_model.bev_state_token_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_BEV_STATE_TOKEN))
    base_model.bev_start_token_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_BEV_START_TOKEN))
    base_model.bev_end_token_id = int(tokenizer.convert_tokens_to_ids(DEFAULT_BEV_END_TOKEN))
    # Weight of the next-visual-state (BEV) prediction loss.
    base_model.bev_loss_weight = float(args.bev_loss_weight)
    # Future horizons (in dataset keyframe steps) whose visual state is predicted,
    # parsed from "--bev-horizon-steps" (e.g. "2,4,6"; the recipes pass a single
    # horizon, 8). Falls back to [2, 4, 6] when empty or
    # malformed.
    try:
        _steps = [
            int(x)
            for x in str(getattr(args, "bev_horizon_steps", "2,4,6")).split(",")
            if str(x).strip() != ""
        ]
        if len(_steps) == 0:
            _steps = [2, 4, 6]
    except Exception:
        _steps = [2, 4, 6]
    base_model.bev_horizon_steps = _steps
    base_model.num_bev_horizons = int(len(_steps))
    # Number of predicted visual tokens per horizon (recipes: 540 = 3 front
    # cameras x 180 DINOv3 patches).
    base_model.bev_pred_len = int(getattr(args, "bev_pred_len", 540) or 540)
    base_model.tokenizer = tokenizer
    # Weight of the LM (text) cross-entropy.
    base_model.qa_loss_weight = float(args.qa_loss_weight)
    # IK loss starts disabled (curriculum stage 1, set again below) and is enabled
    # by TwoStageCurriculumCallback at train begin. ik_stop_grad is initialised
    # from --ik-stop-grad; the callback overwrites it (stage 2: False).
    base_model.ik_loss_weight = 0.0
    base_model.ik_stop_grad = bool(args.ik_stop_grad)
    # Trajectories are divided by this scale (meters) before diffusion and the
    # samples are multiplied back by it.
    base_model.ik_traj_scale = float(args.ik_traj_scale)
    # Per-axis weights of the IK diffusion loss (longitudinal, lateral). Because
    # of the `or 1.0`, a value of 0 falls back to 1.0.
    base_model.ik_weight_longitudinal = float(getattr(args, "ik_longitudinal_weight", 1.0) or 1.0)
    base_model.ik_weight_lateral = float(getattr(args, "ik_lateral_weight", 1.0) or 1.0)
    # ------------------------------------------------------------------
    # DINOv3 visual input and its projector
    # ------------------------------------------------------------------
    # Load the frozen DINOv3 and keep it as a plain attribute of the model
    # (CameraEncoder is not an nn.Module, so its weights are not saved with
    # checkpoints, trained, wrapped by DDP / ZeRO-3 or cast with the model); the
    # forward encodes the camera images with it (model._encode_camera_frames). The
    # token width C is the DINOv3 hidden size.
    camera_encoder = CameraEncoder(args.dinov3_model)
    mm_hidden_size = camera_encoder.hidden_size
    rank0_print(f"[INFO] Visual input: {camera_encoder!r} | mm_hidden_size={mm_hidden_size}")
    if int(args.bev_pred_len) != camera_encoder.num_tokens:
        rank0_print(
            f"[WARN] --bev-pred-len {args.bev_pred_len} != {camera_encoder.num_tokens} DINOv3 tokens per frame; the next-state targets are resampled to it."
        )
    base_model.camera_encoder = camera_encoder
    # Create (or reuse) mm_projector_bev: C -> LLM hidden size, and alias it as
    # mm_projector_scene / mm_projector.
    ensure_mm_projectors(base_model, mm_hidden_size=mm_hidden_size)
    # The projector is attached after from_pretrained, so restore its weights
    # explicitly from the local checkpoint (if model_src is one); keys are matched
    # by name and shape.
    _w_src = _resolve_ckpt_dir_for_weight_loading(str(model_src))
    if _w_src:
        _restore_projector_weights_from_ckpt(base_model, _w_src)
    # ------------------------------------------------------------------
    # IK modules (camera-conditioned query pool + cross-attention diffusion head)
    # ------------------------------------------------------------------
    # The IK modules operate on LLM hidden states (Qwen2.5-0.5B: 896).
    H = int(base_model.config.hidden_size)
    # Number of predicted future horizons (>= 1).
    num_h = int(getattr(base_model, "num_bev_horizons", 1) or 1)
    if num_h < 1:
        num_h = 1
    # 4-query camera-conditioned pool (one query per front camera over its 180
    # patches + one global query over all 540 tokens) and a per-query object box /
    # presence head for the auxiliary object losses. cond_dim (4 pooled vectors for
    # the current frame + 4 per horizon) is only reported in the log; the
    # cross-attention head below attends to token sequences and does not take it
    # as an argument.
    base_model.ik_attn_pool = CameraConditionedQueryPool(
        hidden_size=H,
        num_heads=int(getattr(args, "ik_attn_heads", 4) or 4),
        patches_per_cam=camera_encoder.patches_per_camera,
    )
    base_model.object_query_head = ObjectQueryHead(hidden_size=H)
    cond_dim = 4 * H + num_h * 4 * H
    rank0_print(
        f"[INFO] CameraConditionedQueryPool (K=4) + ObjectQueryHead enabled. cond_dim={cond_dim}"
    )
    # Flattened trajectory size = 2 (x, y) * num_waypoints (8 -> 16 in the NAVSIM
    # recipes).
    _ik_nw = int(getattr(args, "ik_num_waypoints", 8) or 8)
    _ik_traj_dim = 2 * _ik_nw
    # The IK head: a conditional diffusion denoiser whose noisy trajectory
    # (B, traj_dim) cross-attends to the current-frame visual tokens and to the
    # predicted future visual-state tokens (each (N, H)). It sees no ego status
    # or text. num_steps = diffusion timesteps T (--ik-num-steps; 50 in the
    # recipes).
    base_model.ik_head = CrossAttentionDiffusionIK(
        hidden_size=H,
        traj_dim=_ik_traj_dim,
        d_model=int(getattr(args, "ik_decoder_dim", 256) or 256),
        num_heads=int(getattr(args, "ik_decoder_heads", 8) or 8),
        num_steps=int(args.ik_num_steps),
        dropout=float(getattr(args, "ik_decoder_dropout", 0.0) or 0.0),
    )
    rank0_print(
        f"[INFO] CrossAttentionDiffusionIK: hidden={H}, d_model={getattr(args, 'ik_decoder_dim', 256)}, heads={getattr(args, 'ik_decoder_heads', 8)}, steps={args.ik_num_steps}"
    )
    # Waypoint count used by the forward and by evaluation (ADE / L2).
    base_model.ik_num_waypoints = _ik_nw
    # ODVLA_IK_HEAD_FP32=1 (env): force the IK head parameters to fp32.
    if _ODVLA_IK_HEAD_FP32:
        _pre_dtypes = {p.dtype for p in base_model.ik_head.parameters()}
        base_model.ik_head.float()
        _post_dtypes = {p.dtype for p in base_model.ik_head.parameters()}
        rank0_print(
            f"[ODVLA_IK_HEAD_FP32] ON: ik_head params fp32-forced (pre={sorted((str(d) for d in _pre_dtypes))} -> post={sorted((str(d) for d in _post_dtypes))}); denoise/projection wrapped in autocast(enabled=False). model_dtype={model_dtype}"
        )
    # Warm-start the dynamically attached IK modules (ik_head, ik_attn_pool,
    # object_query_head) from a local checkpoint, because from_pretrained does not
    # know about them. If the checkpoint's ik_mode.json records a different ik_head
    # class, the ik_head weights are skipped (fresh init) while the pool / query
    # head are still restored.
    _w_src = _resolve_ckpt_dir_for_weight_loading(str(model_src))
    if _w_src:
        _ckpt_head_type = _read_ckpt_ik_head_type(_w_src)
        _new_head_type = type(base_model.ik_head).__name__
        _skip_ik = bool(_ckpt_head_type and _ckpt_head_type != _new_head_type)
        if _skip_ik:
            rank0_print(
                f"[LOAD][WARN] ckpt ik_head_type={_ckpt_head_type!r} != new {_new_head_type!r} -> auto fresh-init ik_head (skip warm-start; ik_attn_pool + backbone still restored)."
            )
            rank0_print(f"[LOAD] ik_head fresh-init (warm-start skipped); new type={_new_head_type}")
        _restore_ik_head_weights_from_ckpt(base_model, _w_src, skip_ik_head=_skip_ik)
    # Monte-Carlo expansion factor of the diffusion loss (several noise /
    # timestep draws per sample, since the batch size is 1).
    base_model.ik_diff_mc = int(getattr(args, "ik_diff_mc", 8) or 8)
    # Number of DDIM steps used whenever the forward samples a trajectory for
    # debugging / evaluation (return_ik_pred_for_debug): --ik-debug-sample-steps,
    # defaulting to the training step count when <= 0, and clamped to
    # [1, --ik-num-steps].
    dbg_steps = int(args.ik_debug_sample_steps or 0)
    if dbg_steps <= 0:
        dbg_steps = int(args.ik_num_steps or 1000)
    ntrain = int(args.ik_num_steps or dbg_steps)
    base_model.ik_debug_sample_steps = max(1, min(dbg_steps, ntrain))
    # Weights of the per-query object (BCE) / box (L1) losses for the current and
    # next frame. (`or default`: a value of 0 falls back to the default.)
    base_model.obj_loss_weight = float(getattr(args, "obj_loss_weight", 0.5) or 0.5)
    base_model.box_loss_weight = float(getattr(args, "box_loss_weight", 0.5) or 0.5)
    base_model.obj_loss_weight_next = float(getattr(args, "obj_loss_weight_next", 0.3) or 0.3)
    base_model.box_loss_weight_next = float(getattr(args, "box_loss_weight_next", 0.3) or 0.3)
    # ------------------------------------------------------------------
    # Bind the world-model forward and its helpers onto this model instance.
    # Instance methods are bound with __get__(base_model); static helpers are
    # assigned as plain functions. From here on base_model(**batch) runs
    # BEVLatentActionWorldModelMixin.forward.
    # ------------------------------------------------------------------
    # Visual-feature helpers: camera images -> DINOv3 tokens (online input), and the
    # projected (LLM-width) DINOv3 tokens of a feature dict.
    base_model._encode_camera_frames = (
        BEVLatentActionWorldModelMixin._encode_camera_frames.__get__(base_model)
    )
    base_model._project_scene_feature_from_vt_result = (
        BEVLatentActionWorldModelMixin._project_scene_feature_from_vt_result.__get__(base_model)
    )
    # Sequence utilities (static): length alignment and label-span masking
    # between <bev_start>/<bev_end>.
    base_model._align_len = BEVLatentActionWorldModelMixin._align_len
    base_model._mask_spans_between_tokens_inplace = (
        BEVLatentActionWorldModelMixin._mask_spans_between_tokens_inplace
    )
    # The world-model forward itself (instance attribute, so it takes precedence
    # over LlavaQwenForCausalLM.forward for model(...) calls).
    base_model.forward = BEVLatentActionWorldModelMixin.forward.__get__(base_model)
    # Training does not use the KV cache.
    base_model.config.use_cache = False
    # Gradient checkpointing (on by default). Try the modern keyword API first,
    # then a positional dict, then the bare call for older transformers / models.
    # Note: the [INFO] messages always print 'use_reentrant': False, even when
    # --gradient-checkpointing-use-reentrant set it to True.
    if args.gradient_checkpointing and hasattr(base_model, "gradient_checkpointing_enable"):
        _use_reentrant = bool(getattr(args, "gradient_checkpointing_use_reentrant", False))
        gc_kwargs = {"use_reentrant": _use_reentrant}
        try:
            base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
            rank0_print(
                "[INFO] gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})"
            )
        except TypeError as e1:
            try:
                base_model.gradient_checkpointing_enable(gc_kwargs)
                rank0_print(
                    "[INFO] gradient_checkpointing_enable({'use_reentrant': False}) [positional]"
                )
            except TypeError as e2:
                base_model.gradient_checkpointing_enable()
                rank0_print(
                    f"[WARN] gradient_checkpointing_enable() only. (no kwargs support in this transformers/model). e1={e1}, e2={e2}"
                )
    # Truncation length for the collator: the smaller of the tokenizer's
    # model_max_length (ignored if > 100000, i.e. "unset") and the model's
    # max_position_embeddings.
    tok_max, model_max, eff_max = _compute_effective_max_len(tokenizer, base_model)
    rank0_print(
        f"[INFO] tokenizer.model_max_length={getattr(tokenizer, 'model_max_length', None)} (effective={tok_max}), model.max_position_embeddings={model_max}, effective_max_len_used_for_truncation={eff_max}"
    )
    # ------------------------------------------------------------------
    # Ego-status feature width
    # ------------------------------------------------------------------
    # ego_dim is the RAW ego / CAN-bus width that the dataset builds for each
    # sample, WITHOUT the step-conditioning channel. The ego projector's first
    # Linear takes ego_dim + 1 inputs.
    # Resolution order:
    #   1. --ego-can-bus-dim > 0 (explicit flag; wins even over a checkpoint
    #      marker, with a warning if they differ);
    #   2. config.ego_base_feature_dim (marker written below when step
    #      conditioning widened the projector, so later launches know the raw
    #      width and do not widen it again);
    #   3. legacy fallback: in_features of the loaded ego projector (correct only
    #      for checkpoints that were never step-conditioned);
    #   4. config.ego_feature_dim (default 18).
    # Recipes: NAVSIM SFT passes 18 (raw 18 +
    # step channel = 19); NAVSIM GRPO passes 19 (see the warning branch below).
    ego_dim = int(getattr(base_model.config, "ego_feature_dim", 18))
    mm = base_model.get_model()
    # Current input width of the ego projector, if it has the expected
    # Sequential(Linear, ...) layout.
    _proj_in = None
    if (
        hasattr(mm, "ego_projector")
        and isinstance(mm.ego_projector, nn.Sequential)
        and (len(mm.ego_projector) > 0)
    ):
        if isinstance(mm.ego_projector[0], nn.Linear):
            _proj_in = int(mm.ego_projector[0].in_features)
    _base_flag = int(getattr(args, "ego_can_bus_dim", 0) or 0)
    _base_marker = getattr(base_model.config, "ego_base_feature_dim", None)
    # 1. Explicit --ego-can-bus-dim.
    if _base_flag > 0:
        if _base_marker is not None and int(_base_marker) != _base_flag:
            # Explicit flag wins. NAVSIM GRPO relies on this: it starts from an SFT
            # checkpoint saved with base dim 18 (+1 step channel) and re-reads that
            # 19-wide projector as raw width 19, widening it to 19+1.
            # (So the projector column that was trained on the SFT step channel
            # is applied to the 19th raw ego feature in GRPO, and the new 20th
            # column, zero-initialised below, receives the step channel.)
            rank0_print(
                f"[WARN] --ego-can-bus-dim {_base_flag} overrides checkpoint's config.ego_base_feature_dim={_base_marker}"
            )
        ego_dim = _base_flag
        rank0_print(f"[INFO] ego_dim={ego_dim} (from --ego-can-bus-dim)")
    # 2. Marker saved in the checkpoint config.
    elif _base_marker is not None:
        ego_dim = int(_base_marker)
        rank0_print(f"[INFO] ego_dim={ego_dim} (from config.ego_base_feature_dim marker)")
    # 3. Legacy: trust the loaded projector width.
    elif _proj_in is not None:
        ego_dim = _proj_in
        rank0_print(
            f"[INFO] ego_dim={ego_dim} (legacy fallback: loaded ego_projector width). If this checkpoint was already trained with step conditioning, pass --ego-can-bus-dim to avoid re-applying the +1 bump."
        )
    # 4. Config value (or the default 18).
    else:
        rank0_print(f"[INFO] ego_dim={ego_dim} (from config.ego_feature_dim)")
    # Width the projector must have: +1 input for the step-conditioning channel
    # that the dataset appends (k_norm = actual number of steps to the first
    # future horizon / max horizon, in [0, 1]).
    _ego_dim_model = ego_dim + 1
    # Step conditioning: make the ego projector's first Linear exactly
    # ego_dim + 1 inputs wide.
    if (
        hasattr(mm, "ego_projector")
        and isinstance(mm.ego_projector, nn.Sequential)
        and (len(mm.ego_projector) > 0)
    ):
        old = mm.ego_projector[0]
        # Case A, projector is ego_dim wide (fresh or non-step-conditioned
        # checkpoint, or NAVSIM GRPO's re-read 19-wide SFT projector): build an
        # (ego_dim + 1)-input Linear, copy the old weights into the first ego_dim
        # columns, zero the new column (so the output is unchanged at start),
        # copy the bias, and match the old device/dtype.
        if isinstance(old, nn.Linear) and int(old.in_features) == ego_dim:
            new = nn.Linear(_ego_dim_model, int(old.out_features), bias=old.bias is not None)
            with torch.no_grad():
                new.weight[:, :ego_dim].copy_(old.weight)
                new.weight[:, ego_dim:].zero_()
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            new = new.to(device=old.weight.device, dtype=old.weight.dtype)
            mm.ego_projector[0] = new
            # Record the full and raw widths in the config (saved with every
            # checkpoint); the raw-width marker prevents a second +1 bump when
            # this checkpoint is loaded again.
            base_model.config.ego_feature_dim = int(_ego_dim_model)
            base_model.config.ego_base_feature_dim = int(ego_dim)
            rank0_print(
                f"[INFO] step conditioning: ego_projector resized {ego_dim}→{_ego_dim_model} (new column zero-init, warm start)."
            )
        # Case B, already ego_dim + 1 wide (resume / reload of a step-conditioned
        # checkpoint): keep the weights, just (re)write the config markers.
        elif isinstance(old, nn.Linear) and int(old.in_features) == _ego_dim_model:
            base_model.config.ego_feature_dim = int(_ego_dim_model)
            base_model.config.ego_base_feature_dim = int(ego_dim)
            rank0_print(
                f"[INFO] step conditioning: ego_projector already at in_features={_ego_dim_model}. Resume path, no resize."
            )
        # Case C, any other width: refuse, because it is ambiguous which column
        # holds the trained step channel.
        elif isinstance(old, nn.Linear):
            raise ValueError(
                f"step conditioning: ego_projector.in_features={int(old.in_features)} matches neither ego_dim={ego_dim} (needs bump) nor ego_dim+1={_ego_dim_model} (already bumped). This checkpoint was likely over-bumped by the pre-fix double-bump bug (dims grow +1 per launch) — its trained k_norm column is at an ambiguous position. Use a clean checkpoint, or set --ego-can-bus-dim to the correct raw can_bus width if this width is intentional."
            )
    # With a correctly sized projector, register a load_state_dict pre-hook on
    # its first Linear: any later load of a narrower weight matrix into this layer
    # (e.g. checkpoint weights loaded by the Trainer on resume) is right-padded
    # with zero columns to the expected width instead of failing with a size
    # mismatch.
    if (
        hasattr(mm, "ego_projector")
        and isinstance(mm.ego_projector, nn.Sequential)
        and (len(mm.ego_projector) > 0)
    ):
        _proj_lin = mm.ego_projector[0]
        if isinstance(_proj_lin, nn.Linear) and int(_proj_lin.in_features) == _ego_dim_model:
            _expected_in = int(_proj_lin.in_features)

            def _ego_proj_load_pre_hook(
                state_dict,
                prefix,
                local_metadata,
                strict,
                missing_keys,
                unexpected_keys,
                error_msgs,
            ):
                """``nn.Module`` load-state-dict pre-hook for the ego projector Linear.

                Args:
                    state_dict: The state dict being loaded; modified in place.
                    prefix: Key prefix of this module (e.g.
                        ``"model.ego_projector.0."``).
                    local_metadata, strict, missing_keys, unexpected_keys,
                    error_msgs: Standard hook arguments; unused.

                Side effects:
                    If ``state_dict[prefix + "weight"]`` is a 2-D tensor with
                    ``0 < in < _expected_in`` columns, it is replaced by a copy
                    padded with zero columns on the right to
                    ``(out_features, _expected_in)``. Other widths (including wider
                    ones) and the bias are left untouched.
                """
                w_key = prefix + "weight"
                w = state_dict.get(w_key, None)
                # Nothing to do without a 2-D weight for this layer.
                if w is None or w.dim() != 2:
                    return
                in_now = int(w.shape[1])
                # Already the right width.
                if in_now == _expected_in:
                    return
                # Narrower: append zero columns (warm start for the new channel).
                if 0 < in_now < _expected_in:
                    pad_cols = _expected_in - in_now
                    pad = w.new_zeros(w.shape[0], pad_cols)
                    state_dict[w_key] = torch.cat([w, pad], dim=1)
                    rank0_print(
                        f"[RESUME] padded {w_key} from in={in_now} → in={_expected_in} (+{pad_cols} zero columns for step-conditioning warm start)."
                    )

            _proj_lin._register_load_state_dict_pre_hook(_ego_proj_load_pre_hook)
    # ------------------------------------------------------------------
    # Curriculum stage and datasets
    # ------------------------------------------------------------------
    # Every run starts in curriculum stage 1 with the IK loss off;
    # TwoStageCurriculumCallback switches to stage 2 (IK loss at --ik-loss-weight)
    # at train begin. Note that under scripts/evaluate_checkpoint.py the training
    # loop (and thus the callbacks) never runs, so these initial values stay in
    # place during evaluation.
    base_model.curriculum_stage = 1
    base_model.ik_loss_weight = 0.0
    # Training dataset: world-model transitions (current frame -> future frame(s)
    # at --bev-horizon-steps) from --data / --ann-file with camera images from
    # --sensor-root. The ego vector is
    # ego_dim wide plus the step channel; the training set also carries the
    # camera-query object targets.
    train_dataset = BEVLatentActionWorldModelDataset(
        tokenizer=tokenizer,
        data_path=args.data,
        sensor_root=args.sensor_root,
        ann_file=ann_file,
        cached_info_pkl=args.cached_info_pkl,
        bev_pred_len=args.bev_pred_len,
        ego_dim=int(ego_dim),
        bev_horizon_steps=str(args.bev_horizon_steps),
        ik_num_waypoints=int(getattr(args, "ik_num_waypoints", 8) or 8),
        use_camera_query_targets=True,
    )
    # Batch-size-1 collator that truncates input_ids / labels to eff_max tokens and
    # packs per-sample metadata into batch["wm_meta"].
    collator = DataCollatorForLatentActionWM(tokenizer=tokenizer, max_length=eff_max)
    # World-model validation dataset (--val-data): same options as training, but
    # without the camera-query targets. The ann file and the camera image directory
    # (--val-sensor-root) default to the training ones when not given.
    val_dataset = None
    if args.val_data:
        if args.val_ann_file:
            val_ann_file = str(args.val_ann_file)
        else:
            val_ann_file = ann_file
        val_sensor_root = str(args.val_sensor_root or args.sensor_root)
        rank0_print(
            f"[INFO] Validation data: {args.val_data} | val_ann_file: {val_ann_file} | camera images: {val_sensor_root}"
        )
        val_dataset = BEVLatentActionWorldModelDataset(
            tokenizer=tokenizer,
            data_path=args.val_data,
            sensor_root=val_sensor_root,
            ann_file=val_ann_file,
            cached_info_pkl=args.cached_info_pkl,
            bev_pred_len=args.bev_pred_len,
            ego_dim=int(ego_dim),
            bev_horizon_steps=str(args.bev_horizon_steps),
            ik_num_waypoints=int(getattr(args, "ik_num_waypoints", 8) or 8),
        )
        rank0_print(f"[INFO] Validation dataset (wm) size: {len(val_dataset)}")
    # Evaluation input for the trainer: the WM validation dataset (metric prefix
    # eval/...), or None (no evaluation).
    eval_dataset_for_trainer = val_dataset
    # Periodic evaluation every --eval-steps only when validation data exists.
    _has_val = eval_dataset_for_trainer is not None
    _eval_strategy = "steps" if _has_val else "no"
    _eval_steps = int(args.eval_steps) if _has_val else None
    import inspect as _inspect

    # transformers renamed `evaluation_strategy` to `eval_strategy`; use whichever
    # keyword the installed TrainingArguments accepts.
    _ta_params = _inspect.signature(TrainingArguments.__init__).parameters
    _eval_strategy_kw = "eval_strategy" if "eval_strategy" in _ta_params else "evaluation_strategy"
    # HF TrainingArguments built from the CLI. Notes:
    #   * fp16 is disabled whenever bf16 is on (bf16 wins);
    #   * save_total_limit < 0 -> keep every checkpoint;
    #   * max_steps <= 0 -> -1 (train for num_train_epochs);
    #   * remove_unused_columns=False keeps the custom batch keys (uniad_pth,
    #     gt_traj, wm_meta, ...) that the model forward consumes;
    #   * logging goes to TensorBoard only (logs in <output_dir>/tb).
    targs = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=int(args.per_device_eval_batch_size),
        learning_rate=args.learning_rate,
        lr_scheduler_type=str(getattr(args, "lr_scheduler_type", "linear")),
        warmup_ratio=float(getattr(args, "warmup_ratio", 0.0) or 0.0),
        warmup_steps=int(getattr(args, "warmup_steps", 0) or 0),
        weight_decay=float(getattr(args, "weight_decay", 0.0) or 0.0),
        max_grad_norm=float(getattr(args, "max_grad_norm", 1.0) or 1.0),
        **{_eval_strategy_kw: _eval_strategy},
        eval_steps=_eval_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=None if int(args.save_total_limit) < 0 else int(args.save_total_limit),
        fp16=bool(args.fp16) and (not bool(args.bf16)),
        bf16=bool(args.bf16),
        max_steps=int(args.max_steps) if int(args.max_steps) > 0 else -1,
        gradient_accumulation_steps=int(args.gradient_accumulation_steps),
        dataloader_num_workers=int(args.dataloader_num_workers),
        ddp_find_unused_parameters=(
            True
            if args.ddp_find_unused_parameters is None
            else bool(args.ddp_find_unused_parameters)
        ),
        deepspeed=getattr(args, "deepspeed", None),
        remove_unused_columns=False,
        report_to=["tensorboard"],
        logging_dir=os.path.join(args.output_dir, "tb"),
        run_name=os.path.basename(args.output_dir),
    )
    # Extra attribute read by checkpoint._create_optimizer_with_ik_lr: learning
    # rate of the IK modules (None -> same as --learning-rate). Recipes: 1e-4.
    if args.ik_learning_rate is not None:
        setattr(targs, "ik_learning_rate", float(args.ik_learning_rate))
    else:
        setattr(targs, "ik_learning_rate", None)
    # Instantiate the trainer (default LatentActionDebugTrainer; GRPO passes its
    # own class). `tokenizer=` is mapped to `processing_class=` by compat.py on
    # transformers versions that renamed it; the debug_* / warn_* kwargs are
    # consumed by common.DebugTrainer.
    trainer = (trainer_cls or LatentActionDebugTrainer)(
        model=base_model,
        args=targs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset_for_trainer,
        data_collator=collator,
        tokenizer=tokenizer,
        debug_print_steps=int(args.debug_print_steps),
        debug_print_only_truncated=bool(args.debug_print_only_truncated),
        debug_print_max_chars=int(args.debug_print_max_chars),
        warn_truncation=bool(args.warn_truncation),
        warn_truncation_steps=int(args.warn_truncation_steps),
        debug_one_batch=bool(args.debug_one_batch),
        debug_one_batch_max_show=int(args.debug_one_batch_max_show),
    )
    # Validation sample budget on the trainer instance (shadowing the class
    # default 500). A value <= 0 means "evaluate the whole split", encoded
    # as float("inf") so that min(len(dataset), budget) == len(dataset).
    _eval_max_wm = int(getattr(args, "eval_max_wm_samples", 500))
    if _eval_max_wm > 0:
        trainer._EVAL_MAX_WM_SAMPLES = _eval_max_wm
    else:
        trainer._EVAL_MAX_WM_SAMPLES = float("inf")
    # Let the optimizer factory inspect the resumed optimizer.pt (param-group
    # sizes) when building the parameter groups.
    if resume_ckpt:
        trainer._resume_ckpt = str(resume_ckpt)
    # Replace the default tqdm ProgressCallback with the compact, run-tagged one
    # defined below.
    try:
        trainer.remove_callback(ProgressCallback)
    except Exception:
        pass

    class RunInfoProgressCallback(wm.CompactProgressCallback):
        """Compact progress bar that tags every log line with the run's
        output_dir basename, and prints a startup banner showing the
        launcher script + output_dir so multi-run logs are disambiguable."""

        def _run_tag(self, targs) -> str:
            """Basename of ``targs.output_dir`` (empty string if unset)."""
            return os.path.basename(str(getattr(targs, "output_dir", "") or ""))

        def on_train_begin(self, targs, state, control, **kwargs):
            """Print the ``[RUN]`` banner (local main process only), then create the bar.

            The launcher is taken from ``$LAUNCH_SCRIPT`` or, if unset, from the
            parent process' command line in ``/proc/<ppid>/cmdline`` (Linux).
            """
            if state.is_local_process_zero:
                out = str(getattr(targs, "output_dir", ""))
                # Launcher description: env var first, then the parent process.
                launch_src = os.environ.get("LAUNCH_SCRIPT", "")
                if not launch_src:
                    try:
                        with open(f"/proc/{os.getppid()}/cmdline", "rb") as _f:
                            launch_src = (
                                _f.read().replace(b"\x00", b" ").decode("utf-8", "replace").strip()
                            )
                    except Exception:
                        launch_src = ""
                rank0_print("=" * 80)
                rank0_print(f"[RUN] output_dir = {out}")
                if launch_src:
                    rank0_print(f"[RUN] launcher   = {launch_src}")
                rank0_print(f"[RUN] argv       = {' '.join(sys.argv)}")
                rank0_print("=" * 80)
            # CompactProgressCallback creates the narrow tqdm bar (resumed at
            # global_step when resuming).
            super().on_train_begin(targs, state, control, **kwargs)

        def on_log(self, targs, state, control, logs=None, **kwargs):
            """Write each log dict above the progress bar as ``[RUN=<tag>] {...}``.

            ``total_flos`` is dropped. Falls back to the parent behaviour on
            non-main processes or before the bar exists.
            """
            if state.is_world_process_zero and getattr(self, "training_bar", None) is not None:
                _logs = dict(logs) if logs is not None else {}
                _logs.pop("total_flos", None)
                self.training_bar.write(f"[RUN={self._run_tag(targs)}] {_logs}")
                return
            return super().on_log(targs, state, control, logs=logs, **kwargs)

    # add_callback accepts the class and instantiates it.
    trainer.add_callback(RunInfoProgressCallback)
    # Curriculum: the callback switches the model from stage 1 to stage 2 (IK loss
    # at --ik-loss-weight) at train begin.
    _two_stage_cb = TwoStageCurriculumCallback(
        model=base_model,
        ik_loss_weight=float(args.ik_loss_weight),
    )
    trainer.add_callback(_two_stage_cb)
    # Replace the Trainer's train sampler with CurriculumEpochSampler (a seeded,
    # epoch- and stage-dependent shuffle) and let the callback feed it the
    # epoch number.
    # data_seed if set, else seed (TrainingArguments default 42).
    _seed = int(getattr(targs, "data_seed", None) or getattr(targs, "seed", 42) or 42)
    _stage_sampler = CurriculumEpochSampler(
        train_dataset,
        model=base_model,
        seed=_seed,
        shuffle=True,
    )
    # Instance attribute shadows Trainer._get_train_sampler; the optional
    # argument covers transformers versions that pass the dataset.
    trainer._get_train_sampler = lambda td=None: _stage_sampler
    _two_stage_cb._sampler = _stage_sampler
    rank0_print("[INFO] using CurriculumEpochSampler")

    def _build_ik_mode_dict(model):
        """Describe the trajectory-decoding configuration of ``model`` for ``ik_mode.json``.

        The file is written next to the weights (output dir and every
        checkpoint) so later loads can check compatibility;
        ``checkpoint._read_ckpt_ik_head_type`` reads ``ik_head_type`` to decide
        whether the IK head can be warm-started. The other fields document the
        run.

        Args:
            model: The (unwrapped) DriveIK model.

        Returns:
            JSON-serialisable dict of live model attributes: BEV horizons /
            prediction length, IK enablement, trajectory scale, head class and
            diffusion steps, and the training dtype.
        """
        # Head object, its diffusion step count (0 if absent) and class name
        # ("unknown" when no IK head is attached).
        _ik_head_obj = getattr(model, "ik_head", None)
        _ik_num_steps_val = int(getattr(_ik_head_obj, "num_steps", 0) or 0)
        _ik_type_val = type(_ik_head_obj).__name__ if _ik_head_obj is not None else "unknown"
        # "ik_enabled" is evaluated now, i.e. it depends on the model's current
        # ik_loss_weight (0 before the curriculum callback switches stages).
        return {
            "bev_horizon_steps": [int(x) for x in getattr(model, "bev_horizon_steps", [6])],
            "bev_pred_len": int(getattr(model, "bev_pred_len", 0) or 0),
            "ik_enabled": bool(
                _ik_head_obj is not None and float(getattr(model, "ik_loss_weight", 0.0) or 0.0) > 0
            ),
            "ik_traj_scale": float(getattr(model, "ik_traj_scale", 20.0) or 20.0),
            "ik_head_type": _ik_type_val,
            "ik_num_steps": _ik_num_steps_val if _ik_num_steps_val > 0 else None,
            "train_dtype": (
                "bf16"
                if bool(getattr(model, "_trained_bf16", False))
                else "fp16" if bool(getattr(model, "_trained_fp16", False)) else "fp32"
            ),
        }

    def _write_ik_mode(directory, model):
        """Write ``_build_ik_mode_dict(model)`` to ``<directory>/ik_mode.json``.

        Only the global rank 0 (or a non-distributed process) writes; the
        directory is created if needed. Errors are printed as a warning and
        otherwise ignored.
        """
        import json as _json

        # Single writer in distributed runs.
        _is_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if not _is_rank0:
            return
        try:
            os.makedirs(directory, exist_ok=True)
            _ik_mode = _build_ik_mode_dict(model)
            _ik_mode_path = os.path.join(directory, "ik_mode.json")
            with open(_ik_mode_path, "w") as _f:
                _json.dump(_ik_mode, _f, indent=2)
            rank0_print(f"[INFO] Saved IK mode to {_ik_mode_path}: {_ik_mode}")
        except Exception as _e:
            rank0_print(f"[WARN] Could not save ik_mode.json to {directory}: {_e}")

    # Record the configuration in output_dir right away. Note that it reflects
    # the pre-training state: ik_loss_weight is still 0 here (curriculum stage
    # 1), so this copy has "ik_enabled": false; it is not rewritten after
    # training.
    _write_ik_mode(args.output_dir, base_model)

    class _IKModeCheckpointCallback(TrainerCallback):
        """Also write ``ik_mode.json`` into every ``checkpoint-<step>`` directory."""

        def on_save(self, args, state, control, **kwargs):
            """Called by the Trainer after it saved a checkpoint (``args`` = TrainingArguments)."""
            _ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
            _write_ik_mode(_ckpt_dir, base_model)

    trainer.add_callback(_IKModeCheckpointCallback())
    # Optional --ddp-static-graph: switch the DDP wrapper to static-graph mode
    # (the log message calls this "SFT + GRPO double-mark protection"). The
    # wrapper only exists once Trainer.train() has prepared the model, so the
    # callback below retries at train begin and at every step until it succeeds.
    if bool(getattr(args, "ddp_static_graph", False)):
        try:
            from torch.nn.parallel import DistributedDataParallel as _DDP
        except Exception:
            _DDP = None

        def _try_enable_static_graph(_trainer):
            """Find the DDP wrapper around the model and call ``_set_static_graph()`` on it.

            Searches ``trainer.model_wrapped`` and the Accelerator's ``_models``,
            following ``.module`` / ``.model`` links up to 10 levels deep (cycles are
            skipped).

            Returns:
                True if static graph was enabled, False otherwise (no DDP wrapper
                found yet, DDP unavailable, or the call failed).
            """
            if _DDP is None:
                return False
            # Root objects that may be (or contain) the DDP wrapper.
            candidates = []
            mw = getattr(_trainer, "model_wrapped", None)
            if mw is not None:
                candidates.append(mw)
            acc = getattr(_trainer, "accelerator", None)
            if acc is not None:
                for mdl in getattr(acc, "_models", []) or []:
                    candidates.append(mdl)
            # Walk down each wrapper chain until a DDP instance is found.
            seen = set()
            for root in candidates:
                tgt = root
                for _ in range(10):
                    if tgt is None or id(tgt) in seen:
                        break
                    seen.add(id(tgt))
                    if isinstance(tgt, _DDP):
                        try:
                            tgt._set_static_graph()
                            rank0_print(
                                "[DDP] static_graph enabled (SFT + GRPO double-mark protection)."
                            )
                            return True
                        except Exception as _e:
                            rank0_print(f"[DDP] static_graph fail: {type(_e).__name__}: {_e}")
                            return False
                    nxt = getattr(tgt, "module", None)
                    if nxt is None:
                        nxt = getattr(tgt, "model", None)
                    tgt = nxt
            return False

        class _DDPStaticGraphCallback(TrainerCallback):
            """Retry ``_try_enable_static_graph`` until it succeeds once (class-level flag)."""

            _enabled = False

            def on_train_begin(self, args, state, control, **kwargs):
                """First attempt, once training has started."""
                if _DDPStaticGraphCallback._enabled:
                    return
                if _try_enable_static_graph(trainer):
                    _DDPStaticGraphCallback._enabled = True

            def on_step_begin(self, args, state, control, **kwargs):
                """Retry before each step; report a missing wrapper at step 0."""
                if _DDPStaticGraphCallback._enabled:
                    return
                if _try_enable_static_graph(trainer):
                    _DDPStaticGraphCallback._enabled = True
                elif int(state.global_step or 0) == 0:
                    rank0_print(
                        f"[DDP] static_graph: no DDP wrapper found (model_wrapped={type(getattr(trainer, 'model_wrapped', None)).__name__}); will retry on subsequent steps."
                    )

        trainer.add_callback(_DDPStaticGraphCallback())
    # If DeepSpeed is importable, globally replace DeepSpeedEngine.load_checkpoint
    # with a no-op (regardless of --deepspeed). On a DeepSpeed resume the model
    # weights therefore come only from from_pretrained(resume checkpoint) and the
    # DeepSpeed optimizer / scheduler state is NOT restored (it starts fresh).
    try:
        from deepspeed.runtime.engine import DeepSpeedEngine as _DSEngine

        def _lc_noop(self, load_dir, tag=None, **kw):
            """Stand-in for ``DeepSpeedEngine.load_checkpoint``: log and return ``(load_dir, {})``."""
            rank0_print(
                "[RESUME] ZeRO-3 load_checkpoint skipped — model weights pre-loaded via from_pretrained, optimizer resets."
            )
            return (load_dir, {})

        _DSEngine.load_checkpoint = _lc_noop
    except Exception as _e:
        rank0_print(f"[RESUME][WARN] Could not patch DeepSpeed: {_e}")
    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    # Train (resuming optimizer / scheduler / RNG / step counters when a
    # checkpoint was found). Under scripts/evaluate_checkpoint.py, Trainer.train is
    # replaced by an evaluate() call that writes eval_metrics.json, and save_model
    # below is a no-op.
    # With --fp16 (the SFT recipes), the GradScaler skips the first ~10 optimizer
    # steps while it calibrates the loss scale: grad_norm is logged as nan and the
    # learning rate stays 0 for those steps. This is expected.
    if resume_ckpt:
        trainer.train(resume_from_checkpoint=resume_ckpt)
    else:
        trainer.train()
    # Save the final weights, config and tokenizer files to output_dir.
    trainer.save_model(args.output_dir)
    rank0_print(f"[DONE] saved to {args.output_dir}")
