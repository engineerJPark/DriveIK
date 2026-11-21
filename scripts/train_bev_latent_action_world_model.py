#!/usr/bin/env python
"""Training entry point; parse --help before importing PyTorch.

Used by the supervised recipe in scripts/train_common.sh (NAVSIM SFT). It parses the world-model / IK arguments defined in
drivevla/training/config.py and hands them to drivevla.training.train.main, which
builds the tokenizer, the Qwen-based world model with its diffusion IK head, the
datasets and the Trainer, and then trains and saves the model. The NAVSIM GRPO stage
uses scripts/train_bev_latent_action_world_model_grpo.py instead.

Example (normally started through a launcher, which supplies all required flags):

    python scripts/train_bev_latent_action_world_model.py --help
"""

import sys
from pathlib import Path

# Make the repository root importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivevla.training.config import build_argparser

if __name__ == "__main__":
    # Parse first: config.py only needs argparse, so --help and argument errors return
    # without importing PyTorch / transformers.
    args = build_argparser().parse_args()
    # Import the heavy training module only once the arguments are valid.
    from drivevla.training.train import main

    main(args)
