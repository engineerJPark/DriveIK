#!/usr/bin/env bash
# DriveIK on NAVSIM, stage 2: reinforcement fine-tuning (RFT) with diffusion GRPO.
#
# Continues from the NAVSIM SFT checkpoint (SFT_CKPT) for 10 epochs. For each scene the
# diffusion IK head samples a group of 4 trajectories, the NAVSIM PDM scorer rates them
# inline against the PDM metric cache (an EPDMS-style score), and a pairwise preference
# loss on the best and worst sample, measured relative to a frozen copy of the SFT
# policy, is added to the SFT loss. bf16, batch size 1 per GPU, GPUs 0-3 by default,
# launched through torch.distributed.run. Produces DriveIK_NAVSIM_rft.
#
# Recipe: navsim_grpo in scripts/train_common.sh, where every flag is set and explained.
#
# Environment variables:
#   DATA_ROOT           annotations and prompt data (default: data)
#   SENSOR_ROOT         NAVSIM camera images of the training split (sensor_blobs/trainval)
#   VAL_SENSOR_ROOT     NAVSIM camera images of the validation split (sensor_blobs/test)
#   DINOV3_MODEL        DINOv3 checkpoint (default facebook/dinov3-vits16-pretrain-lvd1689m)
#   SFT_CKPT            NAVSIM SFT checkpoint to start from
#                       (default: outputs/DriveIK_NAVSIM_sft)
#   PDM_CACHE           NAVSIM PDM metric cache (required for training)
#   NUPLAN_MAPS_ROOT    nuPlan maps directory (required for training)
#   NAVSIM_DEVKIT_ROOT  NAVSIM source checkout; omit if the devkit is installed
#   OUT                 output directory (default: outputs/DriveIK_NAVSIM_rft)
#   EVAL_CKPT           evaluate this checkpoint instead of training
#   Optional: CUDA_VISIBLE_DEVICES (default 0,1,2,3) / NGPU / MAX_GPUS, ODVLA_PYTHON,
#   MASTER_PORT, DRY_RUN=1 (see README.md). ZeRO-3 is not used for this stage.
# Training resumes automatically from the newest checkpoint-* in OUT.
#
# Training (after train_navsim/DriveIK_NAVSIM_sft.sh has finished):
#   DATA_ROOT=/data/driveik SENSOR_ROOT=/data/navsim/sensor_blobs/trainval \
#     VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
#     SFT_CKPT=outputs/DriveIK_NAVSIM_sft \
#     PDM_CACHE=/data/navsim/metric_cache_navtrain \
#     NUPLAN_MAPS_ROOT=/data/nuplan/maps \
#     NAVSIM_DEVKIT_ROOT=/opt/navsim \
#     bash train_navsim/DriveIK_NAVSIM_rft.sh
#
# Inference / evaluation of a checkpoint from this stage (one GPU; no PDM cache, maps or
# devkit needed; metrics are written to
# outputs/eval/<checkpoint>_DriveIK_NAVSIM_rft/eval_metrics.json):
#   DATA_ROOT=/data/driveik SENSOR_ROOT=/data/navsim/sensor_blobs/trainval \
#     VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
#     EVAL_CKPT=checkpoints/DriveIK_NAVSIM_rft \
#     bash train_navsim/DriveIK_NAVSIM_rft.sh
#
# Extra arguments are appended after the recipe and override it, e.g. `--max-steps 5`.

# Abort on any error, unset variable or failed pipeline stage.
set -euo pipefail
# Load the shared launcher; the repository root is found from this file's location, so
# the script can be started from any directory.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/train_common.sh"
# Build and run the navsim_grpo recipe, forwarding any extra command-line arguments.
launch_training navsim_grpo "$@"
