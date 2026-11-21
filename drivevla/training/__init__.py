"""DINOv3 camera world-model and NAVSIM GRPO training.

This package holds the training code behind the two DriveIK NAVSIM stages
(SFT / RFT). The launchers in ``train_navsim/`` call ``scripts/train_common.sh``, which runs one of the
thin entry points in ``scripts/``; those parse the CLI with
:mod:`drivevla.training.config` and then call into this package.

Modules:

* ``config``     -- argparse definitions (``build_argparser`` / ``add_grpo_args``).
* ``compat``     -- import-time shims for the installed PyTorch / DeepSpeed /
  Accelerate / Transformers versions (imported by ``train``).
* ``common``     -- distributed-env helpers, resume resolution, NAVSIM
  info-pkl readers that build prompt metadata and ``can_bus`` vectors,
  checkpoint tensor readers, and the ``DebugTrainer`` diagnostics base class.
* ``vision``     -- the frozen DINOv3 camera encoder (raw images -> the 540 x 384
  visual-state tokens).
* ``prompts``    -- the v3 user-prompt builder (``<SCENE>``/``<EGO>`` live in the
  system prompt, so the user prompt only carries text).
* ``checkpoint`` -- optimizer construction with a separate IK learning rate,
  shape-tolerant optimizer/scheduler resume, and restoration of dynamically
  attached modules (IK head, attention pool, projectors) from checkpoints.
* ``data``, ``model``, ``heads``, ``trainer``, ``train``, ``grpo``, ``pdm`` --
  dataset/collator, the world-model forward/loss mixin, the diffusion IK head
  and pooling modules, Trainer subclasses and curriculum, the SFT ``main``,
  and the GRPO (RFT) stage with inline PDM-score rewards.

The package ``__init__`` itself imports nothing, so importing
``drivevla.training.config`` (for ``--help``) does not pull in PyTorch.
"""
