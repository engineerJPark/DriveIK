#!/usr/bin/env python
"""Checkpoint inference/evaluation entry point.

Builds the model, data and prompts exactly as the training recipe does, loads the
checkpoint given by --model-path, and runs the trainer's evaluation instead of
training: the diffusion IK head samples future trajectories on the validation split
and ADE / per-horizon L2 (meters) are reported. Use it through a launcher:

    EVAL_CKPT=/path/to/checkpoint bash train_navsim/DriveIK_NAVSIM_sft.sh

With EVAL_CKPT set, scripts/train_common.sh passes the recipe's normal arguments with
--model-path set to EVAL_CKPT, leaves out the GRPO-only flags and auto-resume, runs a
single process, and by default writes to outputs/eval/<checkpoint>_<launcher>/.

How it works: drivevla.training.train.main() performs the complete training setup
(tokenizer, model with IK head, datasets, Trainer and callbacks) and ends with
trainer.train() followed by trainer.save_model(). Before calling main(), this script
replaces both methods on transformers.Trainer (and therefore on every trainer subclass
used here): train() runs one evaluation pass, prints the metrics and writes them to
<output-dir>/eval_metrics.json, and save_model() does nothing, so no weights are
written. The reported metrics are listed in README.md ("Inference and evaluation").
"""

import json
import sys
from pathlib import Path

# Make the repository root importable when the script is run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from drivevla.training.config import build_argparser


def main():
    """Parse the training CLI, patch Trainer.train/save_model, and run the training setup."""
    # Same argument parser as training (base arguments only; evaluation never receives
    # the GRPO flags). Parsing first keeps --help and argument errors fast.
    args = build_argparser().parse_args()

    # Heavy imports only after the arguments are known to be valid.
    import torch
    import transformers

    from drivevla.training import train as train_module

    def evaluate_instead_of_train(self, *_args, **_kwargs):
        """Stand-in for Trainer.train: evaluate once, save the metrics, and fail if no
        trajectory could be evaluated. Training arguments such as
        resume_from_checkpoint are ignored."""
        # Accelerate applies this autocast inside Trainer.train(); the model weights
        # stay in fp32, and FlashAttention needs fp16/bf16 activations.
        # bf16 recipes (NAVSIM RFT) use bf16, all others fp16.
        dtype = torch.bfloat16 if self.args.bf16 else torch.float16
        with torch.autocast("cuda", dtype=dtype):
            metrics = self.evaluate()
        # Write the metrics as JSON into the run's output directory.
        out = Path(self.args.output_dir) / "eval_metrics.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(metrics, indent=2))
        print(f"[EVAL] metrics written to {out}", flush=True)
        # The trajectory evaluation skips samples whose forward pass raises; if that
        # happened for every sample (eval_wm/num_samples == 0), fail instead of
        # reporting empty metrics.
        if any(k.endswith("wm/num_samples") and not v for k, v in metrics.items()):
            raise SystemExit("[EVAL][FATAL] no trajectory was evaluated; see the forward errors above.")

    # Swap training for evaluation and disable checkpoint saving, then run the normal
    # training setup, which ends by calling the patched methods.
    transformers.Trainer.train = evaluate_instead_of_train
    transformers.Trainer.save_model = lambda self, *a, **kw: None
    train_module.main(args)


if __name__ == "__main__":
    main()
