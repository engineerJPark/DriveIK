<div align="center">

# Grounding Driving VLA via Inverse Kinematics

**NeurIPS 2026**

Junsung Park, Hyunjung Shim — KAIST, Kim Jaechul Graduate School of AI

<a href="https://engineerjpark.github.io/NeurIPS-2026-DriveIK/"><img src="https://img.shields.io/badge/Project-Page-1f72b8?logo=googlechrome&logoColor=white" alt="Project page"></a>
<a href="https://arxiv.org/abs/2605.21061"><img src="https://img.shields.io/badge/arXiv-2605.21061-b31b1b?logo=arxiv&logoColor=white" alt="Paper (arXiv)"></a>
<a href="https://github.com/engineerJPark/DriveIK"><img src="https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white" alt="Code (GitHub)"></a>
<a href="https://huggingface.co/engineerJPark/DriveIK"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-ffc107" alt="Checkpoints (Hugging Face)"></a>

</div>

This repository contains the official NAVSIM training and evaluation code for **DriveIK**.

## Overview

Representative driving vision-language-action (VLA) models often plan blindly. If a
vehicle is inserted directly in front of the ego car, their trajectories barely change.
We trace this to a shortcut: on ego-predictable training examples, the planner can rely
on ego status and the driving command instead of the scene.

DriveIK anchors trajectory decoding on visual states, in two ways:

1. **Next visual state prediction.** The LLM must predict the future visual scene. This
   gives dense visual supervision and suppresses the shortcut.
2. **Inverse Kinematics Network.** A separate cross-attention conditional diffusion model
   decodes the trajectory from *only* the current and predicted future visual states, so
   it cannot fall back on ego status or text.

With a 0.5B LLM (Qwen2.5-0.5B-Instruct), cameras only, and no LiDAR or BEV encoder,
DriveIK reaches state-of-the-art planning on NAVSIM-v1 and NAVSIM-v2.

| Benchmark | Metric | OpenDriveVLA\* (0.5B) | DriveIK SFT | DriveIK SFT→GRPO |
| --- | --- | --- | --- | --- |
| NAVSIM-v1 | PDMS ↑ | 73.2 | 90.3 | **92.2** |
| NAVSIM-v2 | EPDMS ↑ | 70.2 | 89.3 | **90.6** |

OpenDriveVLA\* shares DriveIK's training setup; see the paper for full tables and
ablations.

## Repository scope

```
train_navsim/        DriveIK_NAVSIM_sft.sh, DriveIK_NAVSIM_rft.sh
scripts/             train_common.sh (shared launcher), Python entry points
drivevla/training/   data, model + IK head, trainer, checkpoints, GRPO + PDM rewards
llava/               Qwen backbone that consumes the DINOv3 camera tokens, v3 prompts
tests/               CPU regression tests
```

## Installation

You need Python ≥ 3.10 and a CUDA build of PyTorch.

```bash
pip install -e '.[train]'          # add '.[zero3]' for DeepSpeed ZeRO-3
pip install flash-attn --no-build-isolation
```

All recipes use `flash_attention_2`, so install a FlashAttention build that matches your
PyTorch/CUDA version.

The camera encoder is the gated Hugging Face checkpoint
[`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
Accept its license on the model page and run `huggingface-cli login` once; after the
first download it is read from the local Hugging Face cache. The repository ships its
own DINOv3 ViT, which loads this checkpoint and gives the same outputs as Hugging Face
`DINOv3ViTModel`, so Transformers ≥ 4.56 is not required.

NAVSIM GRPO training also needs the
[NAVSIM](https://github.com/autonomousvision/navsim) and nuPlan devkits, either installed
or pointed to with `NAVSIM_DEVKIT_ROOT`.

## Data

Every input path is set with an environment variable, so data can live anywhere.
Relative paths are resolved from the directory you launch from.

| Variable | Default | Contents |
| --- | --- | --- |
| `DATA_ROOT` | `data` | Annotations and prompt data (layout below) |
| `SENSOR_ROOT` | required | NAVSIM camera images of the training split (`sensor_blobs/trainval`) |
| `VAL_SENSOR_ROOT` | required | NAVSIM camera images of the validation split, navtest (`sensor_blobs/test`) |
| `DINOV3_MODEL` | `facebook/dinov3-vits16-pretrain-lvd1689m` | DINOv3 checkpoint (Hugging Face id or local directory) |

```
$DATA_ROOT/
  navsim/    opendrivevla_bev_worldmodel_{train,val}.jsonl
             cached_navsim_info.pkl
             navsim_infos_temporal_{train,val}.pkl
$SENSOR_ROOT/ , $VAL_SENSOR_ROOT/
  <log>/CAM_L0/*.jpg, <log>/CAM_F0/*.jpg, <log>/CAM_R0/*.jpg
```

The model reads the three front cameras (`CAM_L0`, `CAM_F0`, `CAM_R0`) at the paths
listed in `navsim_infos_temporal_*.pkl` and encodes them with a frozen DINOv3 ViT-S/16
inside the forward pass of training and evaluation. Each camera is resized to 288×160
(height × width) and ImageNet-normalized, and the 180 patch tokens of DINOv3's final
LayerNorm output (without CLS and register tokens) are kept per camera: 540 × 384 tokens
per frame in fp16. The DINOv3 weights are not part of the DriveIK checkpoints.

## Checkpoints

| Checkpoint | Stage | Initialized from | Epochs |
| --- | --- | --- | --- |
| [`DriveIK_NAVSIM_sft`](https://huggingface.co/engineerJPark/DriveIK/tree/main/DriveIK_NAVSIM_sft) | NAVSIM world model + IK | Qwen2.5-0.5B-Instruct | 30 |
| [`DriveIK_NAVSIM_rft`](https://huggingface.co/engineerJPark/DriveIK/tree/main/DriveIK_NAVSIM_rft) | NAVSIM GRPO with inline PDM rewards | `DriveIK_NAVSIM_sft` | 10 |

Download them into `checkpoints/`:

```bash
huggingface-cli download engineerJPark/DriveIK --local-dir checkpoints
# or a single stage:
huggingface-cli download engineerJPark/DriveIK --include 'DriveIK_NAVSIM_rft/*' --local-dir checkpoints
```

This gives `checkpoints/DriveIK_NAVSIM_sft/` and `checkpoints/DriveIK_NAVSIM_rft/`. Each
folder holds the full model: the LLM, the visual and ego projectors, the IK head and the
tokenizer. The DINOv3 encoder is not included; it is loaded from `DINOV3_MODEL`. Pass a
folder to `EVAL_CKPT` to evaluate it (see below), or the SFT folder to `SFT_CKPT` to run
the RFT stage from the released SFT model.

## Training

Run each launcher with `bash`. The SFT stage must finish before the RFT stage starts.

**SFT** (world model + IK, from Qwen2.5-0.5B-Instruct):

```bash
DATA_ROOT=/data/driveik \
  SENSOR_ROOT=/data/navsim/sensor_blobs/trainval VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
  bash train_navsim/DriveIK_NAVSIM_sft.sh
```

**RFT** (GRPO with inline PDM rewards, from the SFT checkpoint):

```bash
DATA_ROOT=/data/driveik \
  SENSOR_ROOT=/data/navsim/sensor_blobs/trainval VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
  SFT_CKPT=outputs/DriveIK_NAVSIM_sft \
  PDM_CACHE=/data/navsim/metric_cache_navtrain \
  NUPLAN_MAPS_ROOT=/data/nuplan/maps \
  NAVSIM_DEVKIT_ROOT=/opt/navsim \
  bash train_navsim/DriveIK_NAVSIM_rft.sh
```

To start RFT from the released SFT model, set `SFT_CKPT=checkpoints/DriveIK_NAVSIM_sft`.

Runs resume automatically from the newest `checkpoint-*` in `OUT`. Training curves are
logged to TensorBoard in `$OUT/tb` (`tensorboard --logdir $OUT/tb`).

## Inference and evaluation

Set `EVAL_CKPT` and run the launcher of the same stage as the checkpoint.

**SFT checkpoint:**

```bash
DATA_ROOT=/data/driveik \
  SENSOR_ROOT=/data/navsim/sensor_blobs/trainval VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
  EVAL_CKPT=checkpoints/DriveIK_NAVSIM_sft \
  bash train_navsim/DriveIK_NAVSIM_sft.sh
```

**RFT checkpoint:**

```bash
DATA_ROOT=/data/driveik \
  SENSOR_ROOT=/data/navsim/sensor_blobs/trainval VAL_SENSOR_ROOT=/data/navsim/sensor_blobs/test \
  EVAL_CKPT=checkpoints/DriveIK_NAVSIM_rft \
  bash train_navsim/DriveIK_NAVSIM_rft.sh
```

Metrics are printed and written to `$OUT/eval_metrics.json`. The default `OUT` is
`outputs/eval/<checkpoint>_<launcher>`. By default the script evaluates 500 evenly
spaced validation samples; pass `--eval-max-wm-samples 0` to use the full split. NAVSIM PDMS/EPDMS come from the NAVSIM devkit.

## Citation

```bibtex
@inproceedings{park2026driveik,
  title     = {Grounding Driving VLA via Inverse Kinematics},
  author    = {Park, Junsung and Shim, Hyunjung},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgements

This codebase builds on [OpenDriveVLA](https://drivevla.github.io/) and
[LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT). We thank the authors for releasing
their code.

## License

Apache License 2.0. See [LICENSE](LICENSE).
