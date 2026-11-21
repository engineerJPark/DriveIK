#!/usr/bin/env bash
# DriveIK on NAVSIM, stage 1: SFT of the world model + Inverse Kinematics (IK) head.
#
# Starts from the base LLM (BASE_MODEL, default Qwen/Qwen2.5-0.5B-Instruct) and trains it
# on NAVSIM front-camera images, encoded on the fly by a frozen DINOv3, to predict the
# visual state 8 steps ahead,
# while the cross-attention diffusion IK head decodes the 8-waypoint (4 s) trajectory
# from the current and predicted visual states. It uses step
# conditioning on the ego vector (18 raw can_bus channels + 1 step channel) and the
# camera-conditioned object-query pool. 30 epochs, fp16 mixed precision, previous-action
# prompt line on. The result (DriveIK_NAVSIM_sft) is the starting point of
# train_navsim/DriveIK_NAVSIM_rft.sh.
#
# Recipe: navsim_wm in scripts/train_common.sh, where every flag is set and explained.
#
# Environment variables:
#   DATA_ROOT        annotations and prompt data (default: data; files under $DATA_ROOT/navsim/)
#   SENSOR_ROOT      NAVSIM camera images of the training split (sensor_blobs/trainval)
#   VAL_SENSOR_ROOT  NAVSIM camera images of the validation split (sensor_blobs/test)
#   DINOV3_MODEL     DINOv3 checkpoint (default facebook/dinov3-vits16-pretrain-lvd1689m)
#   BASE_MODEL    initial LLM (Hugging Face id or local path)
#   OUT           output directory (default: outputs/DriveIK_NAVSIM_sft)
#   EVAL_CKPT     evaluate this checkpoint instead of training
#   Optional: CUDA_VISIBLE_DEVICES / NGPU / MAX_GPUS, USE_ZERO3=1, ODVLA_PYTHON,
#   MASTER_PORT, DRY_RUN=1 (see README.md).
# Training resumes automatically from the newest checkpoint-* in OUT.
#
# Training:
#   DATA_ROOT=/data/driveik SENSOR_ROOT=/data/navsim/sensor_blobs/trainval \
#     VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
#     bash train_navsim/DriveIK_NAVSIM_sft.sh
#
# Inference / evaluation of a checkpoint from this stage (one GPU; metrics are written
# to outputs/eval/<checkpoint>_DriveIK_NAVSIM_sft/eval_metrics.json). Use this launcher
# only for NAVSIM SFT checkpoints; an RFT checkpoint needs DriveIK_NAVSIM_rft.sh:
#   DATA_ROOT=/data/driveik SENSOR_ROOT=/data/navsim/sensor_blobs/trainval \
#     VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
#     EVAL_CKPT=checkpoints/DriveIK_NAVSIM_sft \
#     bash train_navsim/DriveIK_NAVSIM_sft.sh
#
# Extra arguments are appended after the recipe and override it, e.g. `--max-steps 5`.

# Abort on any error, unset variable or failed pipeline stage.
set -euo pipefail
# Load the shared launcher; the repository root is found from this file's location, so
# the script can be started from any directory.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/scripts/train_common.sh"
# Build and run the navsim_wm recipe, forwarding any extra command-line arguments.
launch_training navsim_wm "$@"
