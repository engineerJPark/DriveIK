#!/usr/bin/env python
"""Training entry point; parse --help before importing PyTorch.

Used by the NAVSIM RFT recipe (navsim_grpo in scripts/train_common.sh). It accepts
every base training argument plus the "NAVSIM diffusion GRPO" argument group and runs
drivevla.training.grpo.run, which performs the normal training setup with a GRPO
trainer: on each step the diffusion IK head samples a group of trajectories, the inline
NAVSIM PDM scorer rewards them, and a pairwise preference loss against a frozen
reference policy is added to the SFT loss. The launcher always starts it through
torch.distributed.run, even on one GPU.

Example (normally started through the launcher, which supplies all required flags):

    python scripts/train_bev_latent_action_world_model_grpo.py --help
"""

import sys
from pathlib import Path

# Make the repository root importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivevla.training.config import build_argparser

if __name__ == "__main__":
    # Base world-model / IK arguments, extended with the GRPO argument group, parsed
    # before PyTorch is imported so --help and argument errors are fast.
    parser = build_argparser()
    from drivevla.training.config import add_grpo_args

    add_grpo_args(parser)
    args = parser.parse_args()
    # Import the GRPO trainer (and PyTorch) only once the arguments are valid.
    from drivevla.training.grpo import run

    run(args)
