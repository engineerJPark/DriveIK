"""CLI arguments for the supported world-model and NAVSIM GRPO paths.

Every training/eval entry point builds its parser from this module:

* ``scripts/train_bev_latent_action_world_model.py`` (NAVSIM SFT) and
  ``scripts/evaluate_checkpoint.py`` use ``build_argparser()``;
* ``scripts/train_bev_latent_action_world_model_grpo.py`` (NAVSIM RFT) uses
  ``build_argparser()`` followed by ``add_grpo_args(parser)``.

This module imports only :mod:`argparse`, so ``--help`` works without
importing PyTorch. The concrete values used by the two released recipes are
set in ``scripts/train_common.sh``; the defaults below are the fallbacks for
options a recipe does not pass.

Options are declared as data: each row of ``BASE_ARGUMENTS`` /
``GRPO_ARGUMENTS`` is ``(flags, kwargs)`` and is passed verbatim to
``parser.add_argument(*flags, **kwargs)``. Unless ``dest`` is given, argparse
derives the attribute name from the first long flag (``--ik-num-steps`` ->
``args.ik_num_steps``). ``argparse.BooleanOptionalAction`` rows also accept a
``--no-<flag>`` form.
"""

import argparse

# One row per option keeps the supported training configuration easy to compare.
# fmt: off
BASE_ARGUMENTS = [
    # --- Distributed launch -------------------------------------------------
    # --local_rank is filled by legacy launchers; common._get_dist_info() falls
    # back to the LOCAL_RANK / RANK / WORLD_SIZE env vars set by torchrun.
    # --deepspeed is checked and registered (HfDeepSpeedConfig) by
    # common._maybe_init_deepspeed() before the model is instantiated; the
    # launcher passes it only for USE_ZERO3=1 multi-GPU SFT runs.
    (('--local_rank', '--local-rank'), {'dest': 'local_rank', 'type': int, 'default': -1, 'help': 'For torchrun/torch.distributed.launch/deepspeed'}),
    (('--deepspeed',), {'type': str, 'default': None, 'help': 'Path to DeepSpeed config json (enables ZeRO / model sharding).'}),
    # --- Model and input data -----------------------------------------------
    # --model-path: base LLM (Qwen2.5-0.5B-Instruct) for SFT, or the SFT
    #   checkpoint for the second stage / evaluation. A local output_dir with
    #   checkpoint-* subdirs is also accepted when restoring extra modules
    #   (see checkpoint._resolve_ckpt_dir_for_weight_loading).
    # --attn-implementation: passed to from_pretrained (all recipes use
    #   flash_attention_2).
    # --data: world-model jsonl (opendrivevla_bev_worldmodel_{train,val}.jsonl).
    # --sensor-root: camera image directory of the training split (NAVSIM
    #   sensor_blobs/trainval). The three front cameras listed in the --ann-file
    #   infos (info["cams"][cam]["data_path"], relative to this directory) are
    #   read per frame and encoded during training / evaluation by a frozen
    #   DINOv3 (--dinov3-model; drivevla/training/vision.py). Samples whose
    #   current-frame images are missing are dropped when the dataset index is
    #   built.
    # --dinov3-model: DINOv3 checkpoint (Hugging Face id or a
    #   local directory with config.json + model.safetensors); not saved with the
    #   DriveIK checkpoints.
    # --ann-file: temporal infos pkl (navsim_infos_temporal_*.pkl); provides
    #   the prev/next links between frames and the camera image paths (the
    #   NAVSIM infos carry no can_bus vectors).
    # --cached-info-pkl: cached_navsim_info.pkl; source of the textual ego
    #   status / history / command and of the GT future trajectory (see
    #   common.load_token_drivevla_prompt_meta_from_infos_pkl and
    #   data.load_token_to_gt_fut_traj_map).
    (('--model-path',), {'type': str, 'required': True}),
    (('--attn-implementation',), {'type': str, 'default': 'sdpa', 'choices': ['sdpa', 'eager', 'flash_attention_2']}),
    (('--data',), {'type': str, 'required': True, 'help': 'json/jsonl containing sample_id and (ideally) trajectory GT'}),
    (('--sensor-root',), {'type': str, 'required': True, 'help': 'Camera images of the training split (NAVSIM sensor_blobs/trainval); a frozen DINOv3 encodes them in the model forward.'}),
    (('--dinov3-model',), {'type': str, 'default': 'facebook/dinov3-vits16-pretrain-lvd1689m', 'help': 'DINOv3 checkpoint (HF id or local dir)'}),
    (('--ann-file',), {'type': str, 'default': None, 'required': True}),
    (('--cached-info-pkl',), {'type': str, 'default': None, 'help': 'Cached info pkl (sample_token->dict, e.g. cached_navsim_info.pkl) for prompt meta and GT trajectories'}),
    # --- Next-visual-state prediction horizons -------------------------------
    # --bev-horizon-steps: frame offsets of the future visual states the LLM
    #   predicts; the recipes pass a single horizon ("8" = 4 s).
    #   train.py stores the parsed list on the model
    #   (bev_horizon_steps / num_bev_horizons); the IK head receives one
    #   predicted token set per horizon.
    (('--bev-horizon-steps',), {'type': str, 'default': '2,4,6', 'help': 'Comma-separated frame horizons (0.5 s each) for multi-step BEV prediction'}),
    # --- Ego vector width / step conditioning (NAVSIM) -----------------------
    # The dataset always appends one extra channel, "k_norm", to the
    #   ego/can_bus vector fed to the <EGO> projector (step conditioning): the
    #   number of frames actually advanced to the first-horizon target frame
    #   (can be fewer than requested on sparse NAVSIM chains) divided by the
    #   largest horizon in --bev-horizon-steps.
    #   train.py therefore widens the first Linear of ego_projector from ego_dim
    #   to ego_dim + 1 inputs (new column zero-initialised) unless it already
    #   has that width (resume); any other width is rejected.
    # --ego-can-bus-dim sets ego_dim, the raw width WITHOUT that extra channel.
    #   0 (default) means: use config.ego_base_feature_dim from the checkpoint,
    #   else the loaded projector width, else config.ego_feature_dim (18).
    #   The NAVSIM recipes chain the widths 18 -> 19 -> 20:
    #     NAVSIM SFT: --ego-can-bus-dim 18 + step channel -> projector in=19;
    #     NAVSIM RFT: --ego-can-bus-dim 19 re-reads that 19-wide SFT projector
    #                 as the raw width and widens it again -> in=20.
    #   The optimizer-state padding in checkpoint.py handles the matching
    #   exp_avg / exp_avg_sq width change on resume.
    (('--ego-can-bus-dim',), {'type': int, 'default': 0, 'help': 'Raw can_bus feature width of the DATASET, WITHOUT the k_norm step-conditioning channel (navsim: 19)'}),
    # --- Next-visual-state loss ---------------------------------------------
    # --bev-pred-len: K, the number of <BEV_STATE> placeholder tokens the
    #   answer contains between <bev_start> and <bev_end>; their hidden states
    #   are regressed onto the projected DINOv3 tokens of the future frame. The
    #   recipes pass 540 (3 front cameras x 180 patches of a 288 x 160 image).
    # --bev-loss-weight: weight of that regression loss.
    (('--bev-pred-len',), {'type': int, 'default': 625}),
    (('--bev-loss-weight',), {'type': float, 'default': 1.0}),
    # --- Camera-conditioned query pool: object auxiliaries ------------------
    # The IK modules include a 4-query CameraConditionedQueryPool plus an
    #   ObjectQueryHead; the obj/box weights scale that head's per-query object
    #   BCE and box L1 losses on the current state and on the predicted next
    #   state. (The NAVSIM infos have no GT boxes, so the targets are always
    #   "no object".)
    (('--obj-loss-weight',), {'type': float, 'default': 0.5, 'help': 'Weight for current-state per-query object BCE loss.'}),
    (('--box-loss-weight',), {'type': float, 'default': 0.5, 'help': 'Weight for current-state per-query box L1 loss.'}),
    (('--obj-loss-weight-next',), {'type': float, 'default': 0.3, 'help': 'Weight for next-horizon per-query object BCE loss.'}),
    (('--box-loss-weight-next',), {'type': float, 'default': 0.3, 'help': 'Weight for next-horizon per-query box L1 loss.'}),
    # --- Text loss ----------------------------------------------------------
    # --qa-loss-weight: weight of the LM cross-entropy on the supervised text
    #   tokens; the <bev_start>...<bev_end> block is masked out of that loss.
    #   The prompt always uses the v3 template (llava/conversation.py), whose
    #   system prompt carries the <SCENE> and <EGO> placeholders.
    (('--qa-loss-weight',), {'type': float, 'default': 1.0}),
    # --- Inverse Kinematics (IK) head: loss weight ---------------------------
    # The IK head is the cross-attention conditional diffusion model
    # (heads.CrossAttentionDiffusionIK); each waypoint is (x, y), so the
    # diffusion target has 2 * ik_num_waypoints values.
    # --ik-loss-weight multiplies the IK diffusion loss once training starts
    # (curriculum stage 2); it is 0 at setup time (stage 1), which is also
    # the value checkpoint evaluation runs with.
    (('--ik-loss-weight',), {'type': float, 'default': 1.0}),
    # --- IK trajectory length and diffusion schedule ------------------------
    # --ik-num-waypoints: future waypoints decoded by the IK head (8 = 4 s at
    #   0.5 s; the NAVSIM recipes pass 8 explicitly).
    # --ik-num-steps: number of diffusion timesteps of the IK head's noise
    #   schedule (all recipes pass 50).
    (('--ik-num-waypoints',), {'type': int, 'default': 8, 'help': 'Number of future waypoints predicted by the IK head'}),
    (('--ik-num-steps',), {'type': int, 'default': 1000, 'help': 'diffusion timesteps'}),
    # --- IK head: training / debug sampling details --------------------------
    # --ik-debug-sample-steps: DDIM steps used when the model samples IK
    #   trajectories (debug printing, evaluation, GRPO reference pass).
    # --ik-diff-mc: Monte-Carlo expansion factor of the diffusion IK loss:
    #   each target is repeated this many times with independent
    #   (timestep, noise) draws (passed to the head as ``mc``).
    # --ik-stop-grad: initial value of model.ik_stop_grad (detach the IK
    #   conditioning, i.e. the current / predicted future visual tokens, so
    #   the IK loss does not back-propagate into the LLM). All recipes pass
    #   it; see trainer.TwoStageCurriculumCallback for the value used while
    #   training.
    # --ik-traj-scale: GT trajectories (metres) are divided by this before
    #   diffusion and samples are multiplied back.
    # --ik-attn-heads: attention heads of the CameraConditionedQueryPool.
    # --ik-decoder-{dim,heads,dropout}: CrossAttentionDiffusionIK d_model,
    #   attention heads and dropout.
    # --ik-longitudinal-weight / --ik-lateral-weight: per-axis weights passed
    #   as weight_xy=(w_longitudinal, w_lateral) to the diffusion IK loss. The
    #   head applies w_longitudinal to the y (forward) components and
    #   w_lateral to the x (rightward) components, following the v3 frame
    #   (x right, y forward); the "(x)" / "(y)" in the help strings below do
    #   not match that mapping.
    (('--ik-debug-sample-steps',), {'type': int, 'default': 50, 'help': '#diffusion sampling steps used ONLY for debug IK trajectory printing (DDIM eta=0)'}),
    (('--ik-diff-mc',), {'type': int, 'default': 8, 'help': 'Monte-Carlo expansion factor for diffusion IK training when batch_size=1'}),
    (('--ik-stop-grad',), {'action': argparse.BooleanOptionalAction, 'default': False, 'help': 'stop-grad from IK loss into LLM'}),
    (('--ik-traj-scale',), {'type': float, 'default': 20.0, 'help': 'Scale factor for IK trajectory normalization (meters)'}),
    (('--ik-attn-heads',), {'type': int, 'default': 4, 'help': '#heads of the camera-conditioned query pool.'}),
    (('--ik-decoder-dim',), {'type': int, 'default': 256, 'help': 'Decoder d_model'}),
    (('--ik-decoder-heads',), {'type': int, 'default': 8, 'help': '#attention heads for IK decoder.'}),
    (('--ik-decoder-dropout',), {'type': float, 'default': 0.0, 'help': 'Dropout for IK decoder.'}),
    (('--ik-longitudinal-weight',), {'type': float, 'default': 10.0, 'help': 'Per-dimension weight for longitudinal (x) component in IK loss.'}),
    (('--ik-lateral-weight',), {'type': float, 'default': 5.0, 'help': 'Per-dimension weight for lateral (y) component in IK loss.'}),
    # --- Optimisation (mapped onto transformers.TrainingArguments) -----------
    # --learning-rate applies to the LLM / projectors; --ik-learning-rate (if
    # set) to the IK modules (ik_head, ik_attn_pool) via
    # checkpoint._create_optimizer_with_ik_lr. The recipes use a cosine
    # schedule with warmup and --ik-learning-rate 1e-4.
    (('--output-dir',), {'type': str, 'required': True}),
    (('--num-train-epochs',), {'type': int, 'default': 1}),
    (('--per-device-train-batch-size',), {'type': int, 'default': 1}),
    (('--learning-rate',), {'type': float, 'default': 2e-05}),
    (('--ik-learning-rate',), {'type': float, 'default': None, 'help': 'Optional separate learning rate for IK modules (ik_head / ik_attn_pool)'}),
    (('--lr-scheduler-type',), {'type': str, 'default': 'linear', 'help': 'Learning rate scheduler type for transformers.TrainingArguments (e.g., linear, cosine, constant_with_warmup).'}),
    (('--warmup-ratio',), {'type': float, 'default': 0.0, 'help': 'Linear warmup ratio over total training steps (0..1)'}),
    (('--warmup-steps',), {'type': int, 'default': 0, 'help': 'Number of warmup steps'}),
    (('--weight-decay',), {'type': float, 'default': 0.0, 'help': 'Weight decay for AdamW.'}),
    (('--max-grad-norm',), {'type': float, 'default': 1.0, 'help': 'Gradient clipping norm (TrainingArguments.max_grad_norm).'}),
    (('--logging-steps',), {'type': int, 'default': 10}),
    (('--save-steps',), {'type': int, 'default': 20000}),
    # --- Validation ---------------------------------------------------------
    # World-model validation (trajectory ADE / L2, BEV loss, attention shares)
    # runs on --val-data with its own ann_file and camera images
    # (--val-sensor-root, NAVSIM navtest: sensor_blobs/test), capped at
    # --eval-max-wm-samples evenly spaced samples
    # (0 = full split).
    (('--val-data',), {'type': str, 'default': None, 'help': 'json/jsonl for validation set (same format as --data)'}),
    (('--val-ann-file',), {'type': str, 'default': None, 'help': 'ann_file for validation set'}),
    (('--val-sensor-root',), {'type': str, 'default': None, 'help': 'Camera images of the validation split (NAVSIM sensor_blobs/test); defaults to --sensor-root'}),
    (('--eval-steps',), {'type': int, 'default': 20000, 'help': 'Run validation every N training steps (only used when --val-data is provided).'}),
    (('--per-device-eval-batch-size',), {'type': int, 'default': 1, 'help': 'Batch size per device for validation.'}),
    (('--eval-max-wm-samples',), {'type': int, 'default': 500, 'help': 'Max WM val samples per eval (forward-only, fast)'}),
    # --- Precision, memory and DDP ------------------------------------------
    # fp16 is on by default (SFT recipe); the NAVSIM RFT recipe passes
    # --no-fp16 --bf16. --ddp-find-unused-parameters left at None is set by
    # train.py to (world_size > 1).
    (('--fp16',), {'action': argparse.BooleanOptionalAction, 'default': True}),
    (('--bf16',), {'action': argparse.BooleanOptionalAction, 'default': False}),
    (('--gradient-checkpointing',), {'action': argparse.BooleanOptionalAction, 'default': True}),
    (('--max-steps',), {'type': int, 'default': -1}),
    (('--gradient-accumulation-steps',), {'type': int, 'default': 1}),
    (('--dataloader-num-workers',), {'type': int, 'default': 2}),
    (('--ddp-find-unused-parameters',), {'action': argparse.BooleanOptionalAction, 'default': None, 'help': 'AUTO(None): enable when world_size>1'}),
    (('--ddp-static-graph',), {'action': 'store_true', 'default': False, 'help': 'Call DDP._set_static_graph() at train begin'}),
    (('--gradient-checkpointing-use-reentrant',), {'action': 'store_true', 'default': False, 'help': 'Use the legacy reentrant gradient-checkpointing implementation'}),
    # --- Debug printing (common.DebugTrainer) -------------------------------
    # Periodic decoded prompt / GT / teacher-forced prediction previews,
    # truncation warnings, and the one-time step-0 token/label diagnostic
    # (common.debug_print_one_batch).
    (('--debug-print-steps',), {'type': int, 'default': 0, 'help': 'Print decoded prompt/gt/pred preview every N steps (0 disables).'}),
    (('--debug-print-only-truncated',), {'action': argparse.BooleanOptionalAction, 'default': False, 'help': 'If true, only print when truncation happened.'}),
    (('--debug-print-max-chars',), {'type': int, 'default': 4000, 'help': 'Max characters to print for each preview block.'}),
    (('--warn-truncation',), {'action': argparse.BooleanOptionalAction, 'default': True, 'help': 'Warn when inputs are truncated to max length.'}),
    (('--warn-truncation-steps',), {'type': int, 'default': 500, 'help': 'Throttle truncation warnings: once per N steps (plus first few).'}),
    (('--debug-one-batch',), {'action': argparse.BooleanOptionalAction, 'default': False, 'help': 'Print ONE-BATCH diagnostics at step 0 (pad/eos, special ids, labels mask, shift alignment).'}),
    (('--debug-one-batch-max-show',), {'type': int, 'default': 160, 'help': 'Max tokens to show in the one-batch diagnostic window.'}),
    # --- Resume / checkpoint retention --------------------------------------
    # --resume-from-checkpoint also accepts "auto" / "latest" / "last" (newest
    # checkpoint-* under --output-dir; see
    # common._resolve_resume_from_checkpoint). The launcher passes "auto" when
    # $OUT already contains a checkpoint. --resume-load-model /
    # --resume-load-tokenizer choose whether weights / tokenizer come from that
    # checkpoint (default) or from --model-path.
    (('--resume-from-checkpoint',), {'type': str, 'default': None, 'help': 'Resume Trainer state from checkpoint dir (e.g., output_dir/checkpoint-115500)'}),
    (('--resume-load-model',), {'action': argparse.BooleanOptionalAction, 'default': True, 'help': 'If true and resume-from-checkpoint is set, try to load model weights/config from that checkpoint path first.'}),
    (('--resume-load-tokenizer',), {'action': argparse.BooleanOptionalAction, 'default': True, 'help': 'If true and resume-from-checkpoint is set, try to load tokenizer from that checkpoint path first.'}),
    (('--save-total-limit',), {'type': int, 'default': 1, 'help': 'Keep only the last N checkpoints (older ones will be deleted)'}),
]

# Options of the NAVSIM RFT stage (drivevla/training/grpo.py), added on top of
# BASE_ARGUMENTS by add_grpo_args(). All are read into the GRPO config object
# in grpo.py. Some help strings are in Korean; the comments give the English
# meaning.
GRPO_ARGUMENTS = [
    # --- Policy / reference model and loss mixing ----------------------------
    # --grpo-sft-ckpt: SFT checkpoint loaded (frozen) as the reference policy
    #   in the log-ratios of the GRPO loss; if it is the same path as
    #   --model-path the reference starts from the policy initialisation.
    # --grpo-group-size: G, the number of trajectories sampled from the IK
    #   diffusion head per input (stochastic DDIM, see --grpo-diff-*).
    # --grpo-loss-weight: weight of the GRPO term; the total loss is
    #   sft_loss + w * grpo_loss.
    (('--grpo-sft-ckpt',), {'type': str, 'default': None}),
    (('--grpo-group-size',), {'type': int, 'default': 8, 'help': 'dropout forward 샘플 수 G (default: 8)'}),
    (('--grpo-loss-weight',), {'type': float, 'default': 0.5}),
    # --- Proxy reward (used before / instead of PDM) ------------------------
    # The per-trajectory proxy reward is a weighted sum of components, each
    # added only when its weight is > 0:
    #   ep      ego progress, clamp(final forward y / --grpo-ep-max, 0, 1);
    #   comfort exp(-mean |2nd difference| / --grpo-comfort-sigma).
    # When the inline PDM scorer is active, the group rewards are replaced by
    # PDM scores (the proxy reward is kept only if the scorer raises; see
    # --grpo-pdm-inline-metric-cache-dir).
    (('--grpo-reward-ep-weight',), {'type': float, 'default': 0.5}),
    (('--grpo-reward-comfort-weight',), {'type': float, 'default': 0.2}),
    (('--grpo-ep-max',), {'type': float, 'default': 4.0}),
    (('--grpo-comfort-sigma',), {'type': float, 'default': 1.0}),
    # --- Stochastic diffusion sampling and log-probabilities ----------------
    # (Korean help strings, in English:)
    # --grpo-diff-eta: DDIM eta of the stochastic sampling chain (1.0 = fully
    #   stochastic). --grpo-diff-num-steps: length of the sampling chain.
    # --grpo-sigma-min-sample: floor on the per-step sampling sigma_t.
    # --grpo-sigma-min-logprob: floor on the sigma used in the log-probability
    #   denominator (kept separate from the sampling floor).
    # --grpo-gamma-denoising: per-denoising-step discount of the chain
    #   log-probability.
    # --grpo-xt-clip: +/- clip applied to the initial noise x_T and to the
    #   per-step injected noise (0 = no clip).
    # The help strings name the corresponding options of ReCogDrive.
    (('--grpo-diff-eta',), {'type': float, 'default': 1.0, 'help': 'Stochastic DDIM η'}),
    (('--grpo-diff-num-steps',), {'type': int, 'default': 20, 'help': 'DDPO sampling chain 길이 (default: 20)'}),
    (('--grpo-sigma-min-sample',), {'type': float, 'default': 0.0, 'help': 'ReCogDrive σ_min^exp: stochastic DDIM σ_t의 floor'}),
    (('--grpo-sigma-min-logprob',), {'type': float, 'default': 0.0, 'help': 'ReCogDrive min_logprob_denoising_std: log p 분모 σ floor (sampling σ_min 과 분리)'}),
    (('--grpo-gamma-denoising',), {'type': float, 'default': 1.0, 'help': 'ReCogDrive Eq.12 γ-discount'}),
    (('--grpo-xt-clip',), {'type': float, 'default': 0.0, 'help': 'ReCogDrive randn_clip_value: prior xT (및 per-step injection noise) 의 ±clip'}),
    # --- Diagnostics ----------------------------------------------------------
    # --grpo-diagnostic-steps: log extra diagnostics (trajectory / reward /
    #   log-prob spread, ...) during the first N steps (0 = off).
    (('--grpo-diagnostic-steps',), {'type': int, 'default': 0, 'help': '처음 N step 동안 진단 메트릭(traj/reward/log_p 분산 등) 추가 로깅 (default: 0)'}),
    # --- Inline PDM-score rewards -------------------------------------------
    # --grpo-pdm-inline-metric-cache-dir: NAVSIM metric cache (PDM_CACHE in
    #   the launcher). Setting it registers the scorer, which is then applied
    #   on every GRPO step (trajectories are scored in the nuPlan frame).
    #   Samples whose scene token is not in the cache skip the GRPO term for
    #   that step.
    (('--grpo-pdm-inline-metric-cache-dir',), {'type': str, 'default': '', 'help': 'NAVSIM MetricCache dir'}),
    # --- Pairwise GRPO objective --------------------------------------------
    # --grpo-beta: temperature of the pairwise preference loss built from the
    #   group rewards (best vs worst sample of the group, one pair).
    (('--grpo-beta',), {'type': float, 'default': 0.1, 'help': 'GRPO temperature β'}),
]

# fmt: on


def build_argparser():
    """Build the parser with every option in ``BASE_ARGUMENTS``.

    Used directly by the SFT entry point and by
    ``scripts/evaluate_checkpoint.py``; the GRPO entry point extends the result
    with :func:`add_grpo_args`.

    Returns:
        argparse.ArgumentParser: parser whose ``parse_args()`` yields the
        namespace consumed by ``drivevla.training.train.main``.
    """
    parser = argparse.ArgumentParser(
        description="Train a DINOv3 world model and diffusion IK head"
    )
    # Register every (flags, kwargs) row unchanged, in table order.
    for flags, options in BASE_ARGUMENTS:
        parser.add_argument(*flags, **options)
    return parser


def add_grpo_args(parser):
    """Add the NAVSIM diffusion-GRPO options to an existing parser.

    Args:
        parser: Parser returned by :func:`build_argparser`.

    Side effects:
        Adds an argument group titled "NAVSIM diffusion GRPO" (affects only
        the ``--help`` layout) containing every ``GRPO_ARGUMENTS`` row. Returns
        ``None``; the parser is modified in place.
    """
    group = parser.add_argument_group("NAVSIM diffusion GRPO")
    # Register every GRPO (flags, kwargs) row into that group.
    for flags, options in GRPO_ARGUMENTS:
        group.add_argument(*flags, **options)
