#!/usr/bin/env bash
# Shared launcher for the two supported NAVSIM training recipes.
#
# This file is not run on its own. Each stage launcher sources it and then calls
#
#     launch_training <recipe> "$@"
#
# with one of these recipes:
#
#   recipe        launcher                              stage (initialised from)
#   navsim_wm     train_navsim/DriveIK_NAVSIM_sft.sh      NAVSIM world model + IK head SFT
#                                                       (BASE_MODEL, default Qwen2.5-0.5B-Instruct)
#   navsim_grpo   train_navsim/DriveIK_NAVSIM_rft.sh      NAVSIM diffusion GRPO with inline PDM
#                                                       rewards (SFT_CKPT)
#
# launch_training builds one Python command line from a shared base (data paths, IK
# head, optimizer, logging) plus recipe-specific flags, chooses a plain single-process
# or a torch.distributed.run launch, and then either trains or, when EVAL_CKPT is set,
# evaluates that checkpoint with the same recipe options (scripts/evaluate_checkpoint.py).
# Arguments given to the stage launcher are appended last, so they override the recipe.
#
# Environment variables (see README.md for the full table):
#   DATA_ROOT                      annotation / prompt data root
#   SENSOR_ROOT, VAL_SENSOR_ROOT   NAVSIM camera images of the training / validation
#                                  split (sensor_blobs/trainval, sensor_blobs/test);
#                                  DINOv3 encodes them during training and evaluation
#   DINOV3_MODEL                   DINOv3 checkpoint (HF id or local directory)
#   BASE_MODEL                     initial LLM of the SFT recipe
#   SFT_CKPT                       starting checkpoint of navsim_grpo
#   PDM_CACHE, NUPLAN_MAPS_ROOT    required for navsim_grpo training
#   NAVSIM_DEVKIT_ROOT             NAVSIM source checkout (if the devkit is not installed)
#   EVAL_CKPT                      evaluate this checkpoint instead of training
#   OUT                            output directory
#   NGPU, MAX_GPUS, CUDA_VISIBLE_DEVICES, MASTER_PORT   GPU selection and rendezvous port
#   USE_ZERO3=1, DEEPSPEED_CFG     DeepSpeed ZeRO-3 for multi-GPU (non-GRPO) runs
#   ODVLA_PYTHON                   Python interpreter
#   WANDB_*                        Weights & Biases settings (offline by default)
#   DRY_RUN=1                      print the final command instead of running it

# Abort on any error, unset variable or failed pipeline stage. The launchers set this
# too; because this file is sourced, it applies to the launcher's shell.
set -euo pipefail

# Absolute path of the stage launcher that sourced this file (BASH_SOURCE[1]; index 0
# is this file). Its directory and name give the default W&B project, run id and output
# directory below. It is exported so the trainer's start-up banner
# (RunInfoProgressCallback in drivevla/training/train.py) prints which launcher ran.
LAUNCH_SCRIPT="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)/$(basename "${BASH_SOURCE[1]}")"
export LAUNCH_SCRIPT

# User-supplied paths (env vars below) may be relative to the caller's directory.
# Resolve them before switching to the repository root. A value that does not
# exist there (e.g. a Hugging Face model id) is passed through unchanged.
# Only set, non-absolute values that name an existing file or directory under the
# caller's directory are rewritten, in place (printf -v assigns to the variable whose
# name is in $_var), to an absolute path. The built-in defaults applied later (data,
# outputs/...) are relative to the repository root. OUT is not in this list, so a
# relative OUT is also taken relative to the repository root.
CALLER_PWD="$PWD"
for _var in DATA_ROOT SENSOR_ROOT VAL_SENSOR_ROOT DINOV3_MODEL BASE_MODEL \
  SFT_CKPT PDM_CACHE NUPLAN_MAPS_ROOT NAVSIM_DEVKIT_ROOT DEEPSPEED_CFG EVAL_CKPT; do
  _val="${!_var:-}"
  if [[ -n "$_val" && "$_val" != /* && -e "$CALLER_PWD/$_val" ]]; then
    printf -v "$_var" '%s' "$(cd "$CALLER_PWD" && realpath -- "$_val")"
  fi
done
unset _var _val
# Work from the repository root (the parent of scripts/), so the relative entry points,
# DeepSpeed config and default data/output paths below resolve the same way no matter
# where the launcher was started from.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Interpreter for training and for the small helper snippets (override: ODVLA_PYTHON).
PY="${ODVLA_PYTHON:-python}"
# Make the in-repo packages (drivevla, llava) importable without installing them.
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Runtime settings for distributed training and tokenization:
#   NCCL_TIMEOUT=300               always set to 300 (not taken from the environment)
#   NCCL_ASYNC_ERROR_HANDLING=1    a failed or timed-out NCCL collective aborts the
#                                  process instead of leaving every rank hanging
#   TOKENIZERS_PARALLELISM=false   avoid the Hugging Face tokenizers fork warning and
#                                  possible deadlocks when DataLoader workers fork
export NCCL_TIMEOUT=300
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# Weights & Biases defaults (each can be overridden from the environment):
#   - offline mode: run files stay local under WANDB_DIR and are uploaded after a
#     successful run by wandb_auto_sync_on_exit (below);
#   - no console capture, quiet client;
#   - never upload checkpoints or weight files;
#   - project = launcher directory name (train_navsim) and
#     run id = launcher file name without .sh (e.g. DriveIK_NAVSIM_rft). A fixed run id
#     plus WANDB_RESUME=allow lets an auto-resumed run continue the same W&B run.
# Note: in this release the Python trainer reports to TensorBoard only
# (TrainingArguments(report_to=["tensorboard"]) in drivevla/training/train.py, logs in
# $OUT/tb), so these W&B settings only take effect if a W&B logger is enabled there.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_CONSOLE=off WANDB_SILENT=true
export WANDB_IGNORE_GLOBS="checkpoint-*,*.pt,*.pth,*.ckpt,*.safetensors,*.bin"
export WANDB_PROJECT="${WANDB_PROJECT:-$(basename "$(dirname "$LAUNCH_SCRIPT")")}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-$(basename "$LAUNCH_SCRIPT" .sh)}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"
# Output directory. First record whether the caller set OUT ("1" if non-empty), because
# evaluation mode only replaces the default. The default is outputs/<run id>, e.g.
# outputs/DriveIK_NAVSIM_sft, relative to the repository root. W&B files go to
# $OUT/wandb unless WANDB_DIR is set.
OUT_SET_BY_USER="${OUT:+1}"
OUT="${OUT:-outputs/$WANDB_RUN_ID}"
export WANDB_DIR="${WANDB_DIR:-$OUT/wandb}"

# EXIT trap that launch_training installs right before the real run. After a successful
# run in offline mode it uploads the local W&B run directories (set WANDB_AUTO_SYNC=0 to
# skip this). It never turns a failed run into a successful exit status.
wandb_auto_sync_on_exit() {
  # Exit status of the launcher shell at the moment the trap fires.
  local rc=$?
  # Training failed: keep the local files, skip the upload, and return the original
  # status so the failure is not masked.
  if (( rc != 0 )); then
    echo "[WARN] Training failed (exit=$rc); skipping W&B sync." >&2
    return "$rc"
  fi
  # Upload only when enabled, in offline mode (evaluation sets WANDB_MODE=disabled), and
  # when the wandb CLI is installed.
  [[ "${WANDB_AUTO_SYNC:-1}" == 1 && "$WANDB_MODE" == offline ]] || return 0
  command -v wandb >/dev/null 2>&1 || return 0
  # Base `wandb sync` command; the entity is added only when WANDB_ENTITY is set.
  local -a sync=(wandb sync -p "$WANDB_PROJECT" --include-offline --skip-console)
  [[ -z "${WANDB_ENTITY:-}" ]] || sync+=(-e "$WANDB_ENTITY")
  # Collect this run id's offline run directories (one per launch, e.g. before and after
  # an auto-resume), or every offline run in WANDB_DIR if none matches. nullglob makes
  # an unmatched pattern expand to nothing instead of the literal pattern.
  shopt -s nullglob
  local -a fragments=("$WANDB_DIR"/offline-run-*-"$WANDB_RUN_ID")
  (( ${#fragments[@]} )) || fragments=("$WANDB_DIR"/offline-run-*)
  shopt -u nullglob
  # Upload the first fragment under the fixed run id and append the later ones to that
  # same run. Upload errors are ignored (|| true) so they cannot fail a finished run.
  local frag first=1
  for frag in "${fragments[@]}"; do
    if (( first )); then
      WANDB_MODE=online "${sync[@]}" --id "$WANDB_RUN_ID" "$frag" || true
      first=0
    else
      WANDB_MODE=online "${sync[@]}" --id "$WANDB_RUN_ID" --append "$frag" || true
    fi
  done
}

# Print the number of GPUs to use when NGPU is not given:
#   1. CUDA_VISIBLE_DEVICES is set: count its non-empty entries, ignoring "-1" (counted
#      with plain Python, so CUDA is not initialised);
#   2. otherwise count the GPUs listed by nvidia-smi (grep -c exits 1 when it counts 0;
#      `|| true` keeps the status 0 so the 0 reaches the GPU check in launch_training);
#   3. otherwise ask PyTorch.
# An empty CUDA_VISIBLE_DEVICES counts as unset and falls through to case 2.
detect_ngpu() {
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    "$PY" - <<'PY'
import os
print(len([i for i in os.environ["CUDA_VISIBLE_DEVICES"].split(",") if i.strip() and i.strip() != "-1"]))
PY
  elif command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c 'GPU ' || true
  else
    "$PY" -c 'import torch; print(torch.cuda.device_count())'
  fi
}

# require_dir <ENV_VAR> <path> <description>: fail early (skipped with DRY_RUN=1).
# Succeeds when <path> is an existing directory, or always with DRY_RUN=1 so commands
# can be printed without the data present. Otherwise it prints which variable to set
# and returns 1; callers write `require_dir ... || return 1`, so a missing input stops
# the launcher before Python starts.
require_dir() {
  [[ "${DRY_RUN:-0}" == 1 || -d "$2" ]] && return 0
  echo "[FATAL] $3 not found: '${2}'. Set $1=/path/to/it." >&2
  return 1
}

# launch_training <recipe> [extra args...]
# Build the command for one recipe and run it (or print it with DRY_RUN=1).
launch_training() {
  # Recipe name; the remaining arguments ("$@") are the caller's extra CLI arguments.
  local recipe=$1
  shift
  # Defaults shared by the recipes; the case statement below overrides them per recipe.
  #   dataset    data sub-directory and file prefix (navsim)
  #   epochs     number of training epochs
  #   horizon    --bev-horizon-steps: how many frames ahead the predicted visual state
  #              lies (8 frames = 4 s at 2 Hz)
  #   model      --model-path to initialise from (BASE_MODEL for the SFT recipe)
  #   lr         LLM learning rate (the IK head uses its own 1e-4, see below)
  #   warmup     warmup fraction of the cosine learning-rate schedule
  local dataset=navsim epochs=30 horizon=8
  local model="${BASE_MODEL:-Qwen/Qwen2.5-0.5B-Instruct}" lr=1e-5 warmup=0.145
  # Python entry point; replaced by the GRPO or the evaluation entry point below.
  local entry=scripts/train_bev_latent_action_world_model.py
  # Non-empty EVAL_CKPT switches to evaluation mode.
  local eval_ckpt="${EVAL_CKPT:-}"
  # Recipe-specific flags (also used in evaluation) and GRPO-only flags (training only).
  local -a extra=() grpo=()
  case "$recipe" in
    # Both NAVSIM recipes: NAVSIM rows and camera inputs, 30 epochs, and an 8-step horizon
    # with 8 IK waypoints (NAVSIM's 4 s trajectory at 0.5 s).
    #   --ik-num-waypoints 8     the IK head predicts 8 (x, y) waypoints
    #   --obj/box-loss-weight    per-query object BCE / box L1 losses of the 4-query
    #                            CameraConditionedQueryPool + ObjectQueryHead on the
    #                            current visual state (0.25 / 0.25) and, *-next, on
    #                            the predicted next state (0.15 / 0.15)
    # The model code always adds the "Previous action: <trajectory>" prompt line (the
    # previous frame's GT future trajectory) and the step-conditioning ego channel (a NAVSIM frame chain
    # can end before the full horizon; the number of steps actually reached, divided by
    # the horizon, is appended to the ego vector).
    navsim_wm|navsim_grpo)
      extra+=(--ik-num-waypoints 8
        --obj-loss-weight 0.25 --box-loss-weight 0.25
        --obj-loss-weight-next 0.15 --box-loss-weight-next 0.15)
      if [[ "$recipe" == navsim_wm ]]; then
        # NAVSIM SFT (DriveIK_NAVSIM_sft), trained from the base LLM.
        # Raw ego/can_bus width; the step-conditioning channel is added on top (18+1).
        # Four DataLoader workers: decoding and resizing the six 1920x1080 camera
        # images of a sample (current + horizon frame) takes ~0.25 s of CPU time, a
        # bit less than one training step. (Batches and their order do not depend on
        # the worker count.)
        extra+=(--ego-can-bus-dim 18 --dataloader-num-workers 4)
      else
        # NAVSIM RFT (DriveIK_NAVSIM_rft): 10 epochs of GRPO from the NAVSIM SFT
        # checkpoint, with a lower learning rate (5e-6) and 5% warmup.
        epochs=10 lr=5e-6 warmup=0.05
        model="${SFT_CKPT:-outputs/DriveIK_NAVSIM_sft}"
        # The SFT projector (18+1) is widened to 19+1 at the start of GRPO.
        # (--ego-can-bus-dim takes precedence over the width stored in the checkpoint,
        # so the 19-wide SFT projector is read as raw width 19 and gets one new,
        # zero-initialised step column; see the ego_dim logic in
        # drivevla/training/train.py.) The other flags:
        #   --dataloader-num-workers 0     load batches in the main process
        #   --per-device-*-batch-size 1    one scene per GPU and step: GRPO expands it
        #                                  into a group of trajectory samples scored
        #                                  against that scene's PDM metric cache entry
        #   --no-fp16 --bf16               bf16 instead of the SFT stage's fp16 + loss
        #                                  scaling (the frozen reference is bf16 too)
        extra+=(--ego-can-bus-dim 19 --dataloader-num-workers 0
          --per-device-train-batch-size 1 --per-device-eval-batch-size 1
          --no-fp16 --bf16)
        # GRPO training only. Evaluation keeps the recipe flags above but uses the
        # plain evaluation entry point and needs none of the GRPO inputs below.
        if [[ -z "$eval_ckpt" ]]; then
          # GRPO entry point: base arguments plus the "NAVSIM diffusion GRPO" argument
          # group, run by drivevla.training.grpo.run.
          entry=scripts/train_bev_latent_action_world_model_grpo.py
          # Use GPUs 0-3 unless the caller chose GPUs.
          export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
          # NAVSIM devkit: either installed as a package or given as a source checkout.
          # A checkout is exported and put on PYTHONPATH so the inline PDM scorer
          # (drivevla/training/pdm.py) can import the navsim package from it.
          if [[ -n "${NAVSIM_DEVKIT_ROOT:-}" ]]; then
            export NAVSIM_DEVKIT_ROOT
            export PYTHONPATH="$PWD:$NAVSIM_DEVKIT_ROOT:${PYTHONPATH:-}"
          fi
          # Put the active conda environment's libraries (when CONDA_PREFIX is set) and
          # the system CUDA / x86_64 library directories on the dynamic loader path for
          # the native extensions imported by the NAVSIM / nuPlan stack.
          export LD_LIBRARY_PATH="${CONDA_PREFIX:+$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64:}/usr/local/cuda/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"
          # Inputs GRPO cannot run without: the SFT checkpoint (initial policy and
          # frozen reference), the NAVSIM PDM metric cache (rewards) and the nuPlan
          # maps, which the nuPlan map API finds through NUPLAN_MAPS_ROOT.
          local pdm_cache="${PDM_CACHE:-}"
          require_dir SFT_CKPT "$model" "NAVSIM SFT checkpoint" || return 1
          require_dir PDM_CACHE "$pdm_cache" "NAVSIM PDM metric cache" || return 1
          require_dir NUPLAN_MAPS_ROOT "${NUPLAN_MAPS_ROOT:-}" "nuPlan maps root" || return 1
          export NUPLAN_MAPS_ROOT="${NUPLAN_MAPS_ROOT:-}"
          # GRPO flags (definitions: GRPO_ARGUMENTS in drivevla/training/config.py,
          # implementation: drivevla/training/grpo.py). First line: DDP tolerates
          # parameters that receive no gradient in a step; the frozen reference policy
          # comes from the same SFT checkpoint as --model-path (so it is a copy of the
          # initial policy); G = 4 trajectories are sampled per scene.
          grpo+=(--ddp-find-unused-parameters --grpo-sft-ckpt "$model" --grpo-group-size 4
          # Total loss = SFT loss + 1.0 * GRPO loss. Trajectories are sampled from the
          # diffusion IK head with stochastic DDIM (eta = 1) over a 20-step chain.
          --grpo-loss-weight 1.0 --grpo-diff-eta 1.0 --grpo-diff-num-steps 20
          # Noise floors (ReCogDrive-style): minimum sigma_t when sampling and minimum
          # std in the per-step log-probability denominator.
          --grpo-sigma-min-sample 0.04 --grpo-sigma-min-logprob 0.05
          # Discount 0.6 on the per-denoising-step log-probabilities, clip the prior
          # x_T and the injected noise to +-5, and log extra diagnostics (reward and
          # log-prob spread) for the first 1000 steps.
          --grpo-gamma-denoising 0.6 --grpo-xt-clip 5.0 --grpo-diagnostic-steps 1000
          # Pairwise preference objective on the best and worst sample of each group:
          # L = -log sigmoid(0.2 * [(log pi - log pi_ref)(best) - (log pi - log pi_ref)(worst)]).
          --grpo-beta 0.2
          # Proxy reward weight of ego progress (comfort keeps its default 0.2). The
          # proxy is a fallback only: the PDM score replaces it whenever the scorer
          # returns one, a scene the scorer cannot find skips that step's GRPO term,
          # and only an exception inside the scorer falls back to the proxy. The
          # frozen reference policy pi_ref (bf16) used in the log-ratios above is
          # always built.
          --grpo-reward-ep-weight 1.0
          # Inline NAVSIM PDM scoring of the sampled trajectories (in the nuPlan
          # frame) against the metric cache (an EPDMS-style aggregate), on every GRPO
          # step.
          --grpo-pdm-inline-metric-cache-dir "$pdm_cache")
        fi
      fi
      ;;
    *) echo "Unknown training recipe: $recipe" >&2; return 1 ;;
  esac

  # Evaluation mode (EVAL_CKPT set). The recipe flags chosen above are kept, so the
  # checkpoint runs with the data pipeline, prompt (previous action source) and model
  # options (ego width, precision) it was trained with.
  if [[ -n "$eval_ckpt" ]]; then
    # Inference/evaluation: same data, prompt and model options as the recipe,
    # but load EVAL_CKPT and run trajectory evaluation instead of training.
    # scripts/evaluate_checkpoint.py replaces Trainer.train with Trainer.evaluate.
    require_dir EVAL_CKPT "$eval_ckpt" "Checkpoint to evaluate" || return 1
    model="$eval_ckpt"
    entry=scripts/evaluate_checkpoint.py
    # No W&B for evaluation (this also turns the post-run upload into a no-op).
    export WANDB_MODE=disabled
    # Unless OUT was given, write to outputs/eval/<checkpoint name>_<launcher name>,
    # so an evaluation never touches a training output directory.
    [[ -n "${OUT_SET_BY_USER:-}" ]] || OUT="outputs/eval/$(basename "$eval_ckpt")_$WANDB_RUN_ID"
    WANDB_DIR="$OUT/wandb"
  fi

  # Number of processes (one per GPU): NGPU if given, else detected. Evaluation always
  # uses one process (the evaluation loop runs on rank 0 only). MAX_GPUS caps the count.
  # Anything that is not a positive integer (e.g. 0 GPUs detected) stops the launcher.
  local ngpu="${NGPU:-$(detect_ngpu)}"
  [[ -z "$eval_ckpt" ]] || ngpu=1
  if [[ -n "${MAX_GPUS:-}" ]] && (( ngpu > MAX_GPUS )); then ngpu=$MAX_GPUS; fi
  [[ "$ngpu" =~ ^[1-9][0-9]*$ ]] || { echo '[FATAL] No GPU detected; check CUDA_VISIBLE_DEVICES or NGPU.' >&2; return 1; }
  # Launch prefix: plain `python <entry>` for one process. Multi-GPU runs, and the GRPO
  # entry point even on one GPU, go through torch.distributed.run (single node, one
  # process per GPU, rendezvous on MASTER_PORT, default 29500), so GRPO always runs with
  # the torchrun environment (RANK, LOCAL_RANK, WORLD_SIZE).
  local -a launch=("$PY")
  if (( ngpu > 1 )) || [[ "$entry" == *grpo.py ]]; then
    launch+=(-m torch.distributed.run --standalone --nnodes=1
      --nproc_per_node="$ngpu" --master_port="${MASTER_PORT:-29500}")
  fi
  launch+=("$entry")
  # Optional DeepSpeed ZeRO-3 (USE_ZERO3=1) for multi-GPU runs. The default config
  # (configs/deepspeed/zero3_noprefetch.json) shards parameters with prefetching
  # disabled; DEEPSPEED_CFG selects another file. Never used for navsim_grpo: GRPO
  # builds its frozen reference policy as a deep copy of the model, which its code does
  # not support for ZeRO-3-sharded parameters (it would run without the reference).
  # Note that train.py skips DeepSpeed's own checkpoint
  # loading, so an auto-resumed ZeRO-3 run reloads the weights but restarts the optimizer.
  if (( ngpu > 1 )) && [[ "${USE_ZERO3:-0}" == 1 && "$recipe" != navsim_grpo ]]; then
    local ds_config="${DEEPSPEED_CFG:-configs/deepspeed/zero3_noprefetch.json}"
    [[ -f "$ds_config" ]] || { echo "[FATAL] DeepSpeed config missing: $ds_config" >&2; return 1; }
    extra+=(--deepspeed "$ds_config")
  fi
  # Checkpoint and logging interval in optimizer steps: every 10000 steps for the SFT
  # recipes (the final model is always saved when training ends); GRPO saves every 5000
  # and logs every 50 steps.
  local save_steps=10000 logging_steps=10000
  [[ "$recipe" != navsim_grpo ]] || { save_steps=5000; logging_steps=50; }
  # Data root. The default is relative to the repository root (the current directory).
  local data="${DATA_ROOT:-data}"
  # Visual input of the training and validation splits: the NAVSIM camera images
  # (SENSOR_ROOT: sensor_blobs/trainval for navtrain, VAL_SENSOR_ROOT:
  # sensor_blobs/test for navtest), which the frozen DINOv3 (DINOV3_MODEL) encodes
  # inside the model forward (drivevla/training/vision.py).
  require_dir SENSOR_ROOT "${SENSOR_ROOT:-}" \
    "Camera images of the training split (NAVSIM sensor_blobs/trainval)" || return 1
  require_dir VAL_SENSOR_ROOT "${VAL_SENSOR_ROOT:-}" \
    "Camera images of the validation split (NAVSIM sensor_blobs/test)" || return 1
  local -a visual=(--sensor-root "${SENSOR_ROOT:-}" --val-sensor-root "${VAL_SENSOR_ROOT:-}"
    --dinov3-model "${DINOV3_MODEL:-facebook/dinov3-vits16-pretrain-lvd1689m}")
  # Base arguments of every recipe, for training and evaluation (flag definitions:
  # drivevla/training/config.py). Recipe flags, GRPO flags and the caller's arguments are
  # appended after these and win when an option repeats (argparse keeps the last value).
  # Curriculum (fixed in the code, trainer.TwoStageCurriculumCallback): training
  # switches to stage 2 (IK loss at full weight) at the first training step. Entering
  # stage 2 sets model.ik_stop_grad=False, so during SFT training the IK loss also
  # reaches the LLM despite --ik-stop-grad below; GRPO turns stop-grad back on in every
  # compute_loss.
  local -a args=(
    # Model to start from (base LLM, SFT checkpoint or EVAL_CKPT).
    --model-path "$model"
    # Training rows (sample ids and GT trajectories).
    --data "$data/$dataset/opendrivevla_bev_worldmodel_train.jsonl"
    # Cached per-sample info used to build the prompt (ego status, history, driving
    # command) and the GT future trajectory.
    --cached-info-pkl "$data/$dataset/cached_${dataset}_info.pkl"
    # Validation rows.
    --val-data "$data/$dataset/opendrivevla_bev_worldmodel_val.jsonl"
    # Validation temporal infos (frame chain, ego can_bus) and the epoch count.
    --val-ann-file "$data/$dataset/${dataset}_infos_temporal_val.pkl"
    --num-train-epochs "$epochs"
    # Use FlashAttention-2 (samples with missing camera images / features are always
    # dropped).
    --attn-implementation flash_attention_2
    # Predict the visual state $horizon frames ahead. IK head: cross-attention
    # conditional diffusion model over the current and predicted visual tokens.
    # --ik-stop-grad detaches the LLM features fed to the IK head (see the curriculum
    # note above for training).
    --bev-horizon-steps "$horizon" --ik-stop-grad
    # IK loss weight; the IK head is conditioned on the LLM's predicted next visual
    # state, not the ground-truth one; 50 diffusion timesteps.
    --ik-loss-weight 1.0 --ik-num-steps 50
    # LLM learning rate, a separate 1e-4 learning rate for the IK modules, and the
    # weight of the next-visual-state prediction loss.
    --learning-rate "$lr" --ik-learning-rate 1e-4 --bev-loss-weight 1.0
    # Cosine schedule with warmup and gradient-norm clipping at 1.0.
    --lr-scheduler-type cosine --warmup-ratio "$warmup" --max-grad-norm 1.0
    # Visual input of both splits (see above) and the temporal infos of the training
    # split (frame chain and camera image paths).
    "${visual[@]}"
    --ann-file "$data/$dataset/${dataset}_infos_temporal_train.pkl"
    # Output directory; 540 <BEV_STATE> placeholder tokens per predicted visual state;
    # checkpoint interval, keeping only the newest checkpoint-*.
    --output-dir "$OUT" --bev-pred-len 540 --save-steps "$save_steps" --save-total-limit 1
    # Logging interval, and a decoded prompt / GT / prediction preview every 2000 steps
    # (at most 4000 characters per block).
    --logging-steps "$logging_steps" --debug-print-steps 2000 --debug-print-max-chars 4000)
  # Auto-resume (training only): if $OUT already holds a checkpoint-* directory, pass
  # --resume-from-checkpoint auto and train.py restarts from the newest one. Read as:
  # stop if evaluating; else stop if there is no checkpoint; else append the flag.
  [[ -n "$eval_ckpt" ]] || ! compgen -G "$OUT/checkpoint-*" >/dev/null \
    || args+=(--resume-from-checkpoint auto)
  # DRY_RUN=1: print the complete shell-quoted command (printf %q) on one line and stop.
  # Nothing is created or run; the launcher tests parse this output.
  if [[ "${DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${launch[@]}" "${args[@]}" "${extra[@]}" "${grpo[@]}" "$@"
    printf '\n'
    return 0
  fi
  # Real run: create the output and W&B directories, register the W&B upload as an EXIT
  # trap (it also fires when the training command fails under set -e, and then keeps
  # the failing status), and run. Arguments go in the order base, recipe, GRPO, caller,
  # so the caller's values override the recipe.
  mkdir -p "$OUT" "$WANDB_DIR"
  trap wandb_auto_sync_on_exit EXIT
  "${launch[@]}" "${args[@]}" "${extra[@]}" "${grpo[@]}" "$@"
}
