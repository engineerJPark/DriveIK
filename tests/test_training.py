"""CPU regression tests for the retained training paths.

Run from the repository root (no GPU, dataset or checkpoint is needed):

    CUDA_VISIBLE_DEVICES= python -m unittest tests.test_training -v

LaunchTests runs the real bash launchers with DRY_RUN=1, so scripts/train_common.sh
prints the final command instead of running it, and checks that command (recipe
settings, path handling, evaluation mode, GPU / resume / ZeRO-3 logic) by parsing it
with the real argument parser from drivevla/training/config.py.

ModelTests covers the model side: the cross-attention diffusion IK head, the GRPO
best-vs-worst preference loss, and one complete forward/backward/save step per recipe
on camera images (tests/smoke_training.py, run in a subprocess).

VisionTests covers the DINOv3 camera encoder (drivevla/training/vision.py): the
image preprocessing and the ViT are checked against torchvision / Hugging Face
DINOv3ViTModel when those are installed (Transformers >= 4.56 for DINOv3), and the
encoder keeps only the patch tokens.
"""

import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

# Make the repository root importable and load the real CLI definitions.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from drivevla.training.config import build_argparser, add_grpo_args

# The two stage launchers, in the order NAVSIM SFT, NAVSIM RFT; the tests index this
# list.
NAV = ROOT / "train_navsim"
LAUNCHERS = [NAV / "DriveIK_NAVSIM_sft.sh", NAV / "DriveIK_NAVSIM_rft.sh"]


def script_args(cmd):
    """The arguments after the entry script (drops the interpreter / torchrun prefix)."""
    return cmd[next(i for i, c in enumerate(cmd) if c.endswith(".py")) + 1 :]


class LaunchTests(unittest.TestCase):
    """Launcher tests based on DRY_RUN=1 output; no Python training process is started
    unless a test explicitly sets DRY_RUN=0."""

    def command(self, launcher, out, **env):
        """Dry-run `launcher` from /tmp and return the printed command as argv tokens.

        Defaults: DRY_RUN=1, NGPU=2 (so the multi-GPU torchrun path is taken) and
        OUT=out; keyword arguments add or override environment variables. Raises if
        the launcher exits non-zero.
        """
        result = subprocess.run(
            ["bash", str(launcher)],
            cwd="/tmp",
            text=True,
            capture_output=True,
            env={**os.environ, "DRY_RUN": "1", "NGPU": "2", "OUT": str(out), **env},
            check=True,
        )
        return shlex.split(result.stdout)

    def test_recipes_keep_training_settings(self):
        """Each launcher still produces its published recipe: per-recipe epochs, BEV
        horizon, learning rate and warmup; the shared IK and object-loss settings; the
        previous-action source; the NAVSIM data files; and the ego can_bus width (18 for
        SFT, 19 for RFT)."""
        # (num_train_epochs, bev_horizon_steps, learning_rate, warmup_ratio) for the
        # launchers in LAUNCHERS order.
        expected = [
            (30, 8, 1e-5, 0.145),
            (10, 8, 5e-6, 0.05),
        ]
        with tempfile.TemporaryDirectory() as out:
            for i, launcher in enumerate(LAUNCHERS):
                with self.subTest(recipe=launcher.name):
                    # Parse the arguments after the entry script (this drops the
                    # interpreter and torchrun prefix). The RFT command also carries the
                    # GRPO argument group, so its parser needs those arguments.
                    cmd = self.command(launcher, out)
                    parser = build_argparser()
                    if i == 1:
                        add_grpo_args(parser)
                    ns = parser.parse_args(script_args(cmd))
                    # Recipe-specific schedule.
                    self.assertEqual(
                        (
                            ns.num_train_epochs,
                            int(ns.bev_horizon_steps),
                            ns.learning_rate,
                            ns.warmup_ratio,
                        ),
                        expected[i],
                    )
                    # NAVSIM data files.
                    self.assertTrue(ns.data.endswith("navsim/opendrivevla_bev_worldmodel_train.jsonl"))
                    self.assertTrue(ns.ann_file.endswith("navsim/navsim_infos_temporal_train.pkl"))
                    self.assertTrue(ns.val_ann_file.endswith("navsim/navsim_infos_temporal_val.pkl"))
                    self.assertTrue(ns.cached_info_pkl.endswith("navsim/cached_navsim_info.pkl"))
                    self.assertEqual(ns.ik_num_waypoints, 8)
                    # Settings shared by every recipe: 50-step diffusion IK head with its
                    # own learning rate and stop-grad flag, the camera-query object /
                    # box loss weights and one kept checkpoint.
                    self.assertEqual(ns.ik_num_steps, 50)
                    self.assertEqual(ns.ik_learning_rate, 1e-4)
                    self.assertTrue(ns.ik_stop_grad)
                    self.assertEqual(
                        (ns.obj_loss_weight, ns.box_loss_weight, ns.obj_loss_weight_next, ns.box_loss_weight_next),
                        (0.25, 0.25, 0.15, 0.15),
                    )
                    self.assertEqual(ns.save_total_limit, 1)
                    # Ego input: the raw can_bus width (the step channel is added on top).
                    self.assertEqual(ns.ego_can_bus_dim, [18, 19][i])
                    self.assertEqual(ns.dataloader_num_workers, [4, 0][i])

    def test_visual_input_selection(self):
        """The launchers feed the camera images (SENSOR_ROOT / VAL_SENSOR_ROOT) to the
        DINOv3 encoder (default checkpoint, or DINOV3_MODEL). Relative image roots are
        resolved against the caller's directory."""
        with tempfile.TemporaryDirectory() as tmp:
            for name in ["trainval", "test"]:
                (Path(tmp) / name).mkdir()
            env = {k: v for k, v in os.environ.items() if k != "DINOV3_MODEL"}
            for launcher in LAUNCHERS:
                with self.subTest(recipe=launcher.name):
                    run = lambda **e: shlex.split(subprocess.run(
                        ["bash", str(launcher)], cwd=tmp, text=True, capture_output=True, check=True,
                        env={**env, "DRY_RUN": "1", "NGPU": "1", "OUT": tmp, **e}).stdout)
                    cmd = run(SENSOR_ROOT="trainval", VAL_SENSOR_ROOT="test")
                    arg = lambda flag: cmd[cmd.index(flag) + 1]
                    self.assertEqual(arg("--sensor-root"), f"{tmp}/trainval")
                    self.assertEqual(arg("--val-sensor-root"), f"{tmp}/test")
                    self.assertEqual(arg("--dinov3-model"), "facebook/dinov3-vits16-pretrain-lvd1689m")
                    cmd = run(SENSOR_ROOT="trainval", VAL_SENSOR_ROOT="test", DINOV3_MODEL="my/dino")
                    self.assertEqual(cmd[cmd.index("--dinov3-model") + 1], "my/dino")

    def test_paths_come_from_environment(self):
        """Relative DATA_ROOT / SENSOR_ROOT / SFT_CKPT are resolved against the
        caller's directory, and a BASE_MODEL that is not an existing path (a Hugging
        Face id) is passed through unchanged."""
        with tempfile.TemporaryDirectory() as tmp:
            # Create the relative targets and launch the RFT recipe from that directory.
            for name in ["data", "trainval", "test", "sft"]:
                (Path(tmp) / name).mkdir()
            env = dict(DATA_ROOT="data", SENSOR_ROOT="trainval", VAL_SENSOR_ROOT="test",
                       SFT_CKPT="sft", BASE_MODEL="my/model")
            result = subprocess.run(
                ["bash", str(LAUNCHERS[1])],
                cwd=tmp,  # relative paths are resolved from the caller's directory
                text=True,
                capture_output=True,
                env={**os.environ, "DRY_RUN": "1", "NGPU": "1", "OUT": tmp, **env},
                check=True,
            )
            # Every relative path became an absolute path under the caller's directory.
            cmd = shlex.split(result.stdout)
            arg = lambda flag: cmd[cmd.index(flag) + 1]
            self.assertEqual(arg("--model-path"), f"{tmp}/sft")
            self.assertEqual(arg("--data"), f"{tmp}/data/navsim/opendrivevla_bev_worldmodel_train.jsonl")
            self.assertEqual(arg("--sensor-root"), f"{tmp}/trainval")
            self.assertEqual(arg("--val-sensor-root"), f"{tmp}/test")
            # The SFT recipe forwards a non-path BASE_MODEL as is.
            cmd = self.command(LAUNCHERS[0], tmp, BASE_MODEL="my/model")
            self.assertEqual(cmd[cmd.index("--model-path") + 1], "my/model")

    def test_eval_mode_reuses_recipe_options(self):
        """With EVAL_CKPT every launcher runs scripts/evaluate_checkpoint.py in a single
        process (no torchrun even with NGPU=4), never auto-resumes (although OUT holds
        a checkpoint), passes no GRPO flags (the base parser accepts the command), uses
        the checkpoint as --model-path and keeps the recipe's previous-action source."""
        with tempfile.TemporaryDirectory() as ckpt, tempfile.TemporaryDirectory() as out:
            for launcher in LAUNCHERS:
                with self.subTest(recipe=launcher.name):
                    # A checkpoint-* in OUT would trigger auto-resume during training.
                    (Path(out) / "checkpoint-10").mkdir(exist_ok=True)
                    cmd = self.command(launcher, out, EVAL_CKPT=ckpt, NGPU="4")
                    # cmd[0] is the interpreter; the entry point follows directly.
                    self.assertTrue(cmd[1].endswith("scripts/evaluate_checkpoint.py"), cmd[:3])
                    self.assertNotIn("torch.distributed.run", cmd)
                    self.assertNotIn("--resume-from-checkpoint", cmd)
                    self.assertFalse(any(c.startswith("--grpo") for c in cmd))
                    ns = build_argparser().parse_args(script_args(cmd))
                    self.assertEqual(ns.model_path, ckpt)

    def test_default_output_dirs(self):
        """Without OUT, training writes to outputs/<launcher name> and evaluation to
        outputs/eval/<checkpoint name>_<launcher name>."""
        # Remove OUT / WANDB_DIR from the environment so the launcher defaults apply.
        env = {k: v for k, v in os.environ.items() if k not in ("OUT", "WANDB_DIR")}
        with tempfile.TemporaryDirectory() as ckpt:
            for extra, expected in [({}, "outputs/DriveIK_NAVSIM_sft"),
                                    ({"EVAL_CKPT": ckpt}, f"outputs/eval/{Path(ckpt).name}_DriveIK_NAVSIM_sft")]:
                result = subprocess.run(
                    ["bash", str(LAUNCHERS[0])], cwd="/tmp", text=True, capture_output=True,
                    env={**env, "DRY_RUN": "1", "NGPU": "1", **extra}, check=True,
                )
                cmd = shlex.split(result.stdout)
                self.assertEqual(cmd[cmd.index("--output-dir") + 1], expected)

    def test_missing_required_paths_fail_early(self):
        """Real (DRY_RUN=0) launches stop before Python starts and name the variable to
        set: the RFT launcher without PDM_CACHE, and with a missing SFT_CKPT; the SFT
        launcher without camera images."""
        with tempfile.TemporaryDirectory() as tmp:
            # SFT: no SENSOR_ROOT, then a missing VAL_SENSOR_ROOT.
            env0 = {k: v for k, v in os.environ.items()
                    if k not in ("SENSOR_ROOT", "VAL_SENSOR_ROOT")}
            p = subprocess.run(
                ["bash", str(LAUNCHERS[0])], capture_output=True, text=True,
                env={**env0, "NGPU": "1", "OUT": tmp, "DRY_RUN": "0"},
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Set SENSOR_ROOT=", p.stderr)
            p = subprocess.run(
                ["bash", str(LAUNCHERS[0])], capture_output=True, text=True,
                env={**env0, "NGPU": "1", "OUT": tmp, "DRY_RUN": "0", "SENSOR_ROOT": tmp},
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Set VAL_SENSOR_ROOT=", p.stderr)
            # RFT: the SFT checkpoint exists, but PDM_CACHE / NUPLAN_MAPS_ROOT are unset.
            env = {k: v for k, v in os.environ.items() if k not in ("PDM_CACHE", "NUPLAN_MAPS_ROOT")}
            p = subprocess.run(
                ["bash", str(LAUNCHERS[1])],
                capture_output=True,
                text=True,
                env={**env, "NGPU": "1", "OUT": tmp, "SFT_CKPT": tmp, "DRY_RUN": "0"},
            )
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("Set PDM_CACHE=", p.stderr)
            # RFT: SFT_CKPT points to a directory that does not exist.
            p = subprocess.run(
                ["bash", str(LAUNCHERS[1])],
                capture_output=True,
                text=True,
                env={**env, "NGPU": "1", "OUT": tmp, "SFT_CKPT": f"{tmp}/missing", "DRY_RUN": "0"},
            )
            self.assertIn("Set SFT_CKPT=", p.stderr)

    def test_gpu_launch_resume_and_zero3(self):
        """One GPU runs the entry point directly and an existing checkpoint-* in OUT adds
        --resume-from-checkpoint auto; MAX_GPUS caps NGPU for torchrun, and
        USE_ZERO3=1 adds the DeepSpeed config on multi-GPU runs."""
        with tempfile.TemporaryDirectory() as out:
            # Single GPU + existing checkpoint: plain python launch with auto-resume.
            (Path(out) / "checkpoint-10").mkdir()
            cmd = self.command(LAUNCHERS[0], out, NGPU="1")
            self.assertNotIn("torch.distributed.run", cmd)
            self.assertEqual(cmd[cmd.index("--resume-from-checkpoint") + 1], "auto")
            # 8 requested GPUs capped to 2 processes, with ZeRO-3 enabled.
            cmd = self.command(LAUNCHERS[0], out, NGPU="8", MAX_GPUS="2", USE_ZERO3="1")
            self.assertIn("--nproc_per_node=2", cmd)
            self.assertIn("--deepspeed", cmd)

    def test_failure_status_is_preserved(self):
        """The exit status of a failing training command (7, from a fake interpreter)
        is returned by the launcher; the EXIT trap must not mask it."""
        with tempfile.TemporaryDirectory() as tmp:
            # Fake "python" that exits with status 7, used through ODVLA_PYTHON.
            fake = Path(tmp) / "python"
            fake.write_text("#!/bin/sh\nexit 7\n")
            fake.chmod(0o755)
            # Real (non-dry) single-GPU launch with the W&B upload disabled.
            p = subprocess.run(
                ["bash", str(LAUNCHERS[0])],
                capture_output=True,
                text=True,
                env={
                    **os.environ,
                    "NGPU": "1",
                    "ODVLA_PYTHON": str(fake),
                    "OUT": tmp,
                    "SENSOR_ROOT": tmp,
                    "VAL_SENSOR_ROOT": tmp,
                    "WANDB_AUTO_SYNC": "0",
                    "DRY_RUN": "0",
                },
            )
            self.assertEqual(p.returncode, 7)


class ModelTests(unittest.TestCase):
    """CPU tests of the model-side code: the diffusion IK head, the GRPO preference
    loss, and a full one-step training run for each recipe family."""

    def test_diffusion_chain_and_gradients(self):
        """CrossAttentionDiffusionIK trains and samples: the denoising loss is finite and
        reaches the parameters, DDIM sampling returns a finite (1, 12) trajectory
        (6 waypoints x 2; the head is size-agnostic), and a state_dict round trip reproduces the samples exactly
        under the same seed."""
        import torch
        from drivevla.training.heads import CrossAttentionDiffusionIK

        # Tiny head: 16-wide input tokens (LLM hidden size), d_model 16, 4 heads and
        # 10 diffusion steps. Inputs: 9 current visual tokens, one future-state token
        # set and a flattened 6 x 2 trajectory target.
        torch.set_num_threads(1)
        torch.manual_seed(4)
        head = CrossAttentionDiffusionIK(16, d_model=16, num_heads=4, num_steps=10)
        current = torch.randn(1, 9, 16)
        future = [torch.randn_like(current)]
        target = torch.randn(1, 12)
        # Training loss: finite, and at least one parameter gets a non-zero gradient.
        loss = head.diffusion_loss(current, future, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in head.parameters())
        )
        # Sampling with a shorter 5-step chain returns a finite trajectory.
        sample = head.sample(current, future, num_steps=5)
        self.assertEqual(tuple(sample.shape), (1, 12))
        self.assertTrue(torch.isfinite(sample).all())
        # A freshly built head loaded from the state_dict (strict) produces bit-identical
        # samples for the same random seed, i.e. sampling depends only on the weights
        # and the seed.
        restored = CrossAttentionDiffusionIK(16, d_model=16, num_heads=4, num_steps=10)
        restored.load_state_dict(head.state_dict(), strict=True)
        torch.manual_seed(7)
        a = head.sample(current, future, num_steps=5)
        torch.manual_seed(7)
        b = restored.sample(current, future, num_steps=5)
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_grpo_best_worst_reference_and_ties(self):
        """GRPO preference loss on a hand-made group of 4 samples.

        The best (reward 1.0) and the worst (reward 0.0) sample form the only pair, so
        the loss is -log sigmoid(beta * delta) with
        delta = (0.9 - 0.2) - (-0.2 - 0.5) = 1.4 and beta = 0.2, i.e. -log sigmoid(0.28).
        Only the pair receives gradients (winner up, loser down, middle samples none),
        the reference log-probabilities get no gradient, and a group whose rewards are
        all equal is skipped with zero loss.
        """
        import torch
        from drivevla.training.grpo import GRPOLatentActionWMTrainer

        # Build the trainer without __init__ (no model or Trainer needed) and stub what
        # _compute_grpo_loss uses: beta, the reference refresh hook, and fixed
        # reference log-probabilities.
        trainer = object.__new__(GRPOLatentActionWMTrainer)
        trainer._grpo_cfg = SimpleNamespace(grpo_beta=0.2)
        trainer._maybe_refresh_ref_model = lambda model: None
        ref = torch.tensor([0.2, 0.3, 0.4, 0.5], requires_grad=True)
        trainer._compute_ref_logprobs_for_chain = lambda inputs, chain: ref
        # Policy log-probabilities and rewards of the G = 4 samples.
        logp = torch.tensor([0.9, 0.3, 0.4, -0.2], requires_grad=True)
        rewards = torch.tensor([1.0, 0.5, 0.2, 0.0])
        loss, info = trainer._compute_grpo_loss(logp, rewards, {}, [], None)
        self.assertAlmostEqual(loss.item(), -math.log(1 / (1 + math.exp(-0.28))), places=6)
        # Minimising the loss raises the winner's log-prob (negative gradient) and lowers
        # the loser's (positive gradient); a middle sample gets zero gradient and the
        # reference is detached.
        loss.backward()
        self.assertLess(logp.grad[0], 0)
        self.assertGreater(logp.grad[3], 0)
        self.assertEqual(logp.grad[1].item(), 0)
        self.assertIsNone(ref.grad)
        # Identical rewards carry no preference signal: zero loss, logged as skipped.
        skipped, info = trainer._compute_grpo_loss(logp, torch.ones(4), {}, [], None)
        self.assertEqual(skipped.item(), 0)
        self.assertEqual(info["grpo/skipped_no_var"], 1)

    def test_full_training_forward_backward_and_save(self):
        """Run tests/smoke_training.py for the nav / grpo profiles (camera images
        through a tiny DINOv3) in a subprocess (offline Hugging Face hub, 90 s timeout
        each); see that script for the checks. On failure the tail of its combined
        output is shown."""
        for profile in ["nav", "grpo"]:
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as tmp:
                p = subprocess.run(
                    [sys.executable, str(ROOT / "tests/smoke_training.py"), tmp, profile],
                    cwd=ROOT,
                    env={**os.environ, "HF_HUB_OFFLINE": "1", "PYTHONDONTWRITEBYTECODE": "1"},
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    timeout=90,
                )
                self.assertEqual(p.returncode, 0, p.stdout[-6000:])


def _tiny_dinov3_config(**over):
    """Config of a 2-layer, 8-channel DINOv3 ViT with 4 registers and 16 px patches."""
    cfg = dict(model_type="dinov3_vit", hidden_size=8, intermediate_size=16, num_hidden_layers=2,
               num_attention_heads=2, num_register_tokens=4, patch_size=16, num_channels=3,
               layer_norm_eps=1e-5, rope_theta=100.0, hidden_act="gelu", query_bias=True,
               key_bias=False, value_bias=True, proj_bias=True, mlp_bias=True,
               layerscale_value=1.0, use_gated_mlp=False, image_size=224)
    cfg.update(over)
    return cfg


class VisionTests(unittest.TestCase):
    """The online DINOv3 camera encoder must give the tokens of the released feature
    caches: same preprocessing, same backbone, patch tokens only."""

    @staticmethod
    def _random_vit(cfg, seed=0):
        """A DINOv3ViT with all parameters drawn from N(0, 0.5^2)."""
        import torch
        from drivevla.training.vision import DINOv3ViT

        torch.manual_seed(seed)
        vit = DINOv3ViT(cfg)
        for p in vit.parameters():
            torch.nn.init.normal_(p, std=0.5)
        return vit.eval()

    def _jpeg(self, tmp, name, size=(64, 36), seed=0):
        """Write a random RGB JPEG of `size` (width, height) and return its path."""
        import numpy as np
        from PIL import Image

        path = Path(tmp) / name
        rgb = np.random.default_rng(seed).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
        Image.fromarray(rgb).save(path, quality=90)
        return path

    def test_preprocessing_matches_torchvision(self):
        """load_camera_image is bit-identical to the cache-extraction pipeline
        (torchvision to_tensor -> antialiased resize to 288 x 160 -> ImageNet
        normalize), for down- and upscaling."""
        try:
            import torchvision.transforms.functional as TF
        except ImportError:
            self.skipTest("torchvision not installed")
        import torch
        from PIL import Image
        from drivevla.training.vision import IMAGENET_MEAN, IMAGENET_STD, IMAGE_HW, load_camera_image

        with tempfile.TemporaryDirectory() as tmp:
            for i, size in enumerate([(640, 360), (64, 36)]):
                path = self._jpeg(tmp, f"{i}.jpg", size=size, seed=i)
                ref = TF.normalize(
                    TF.resize(TF.to_tensor(Image.open(path).convert("RGB")), list(IMAGE_HW), antialias=True),
                    list(IMAGENET_MEAN), list(IMAGENET_STD),
                )
                out = load_camera_image(str(path))
                self.assertEqual(tuple(out.shape), (3, *IMAGE_HW))
                torch.testing.assert_close(out, ref, rtol=0, atol=0)

    def test_vit_matches_huggingface_dinov3(self):
        """DINOv3ViT loads a Hugging Face DINOv3ViTModel state_dict strictly and returns
        its last_hidden_state (CLS, registers and patches) for a non-square input."""
        import torch

        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError:
            self.skipTest("transformers without DINOv3 (needs >= 4.56)")
        from drivevla.training.vision import DINOv3ViT

        cfg = _tiny_dinov3_config()
        torch.manual_seed(0)
        hf = DINOv3ViTModel(DINOv3ViTConfig(**{k: v for k, v in cfg.items() if k != "model_type"}))
        for p in hf.parameters():
            torch.nn.init.normal_(p, std=0.5)
        hf.eval()
        ours = DINOv3ViT(cfg).eval()
        ours.load_state_dict(hf.state_dict(), strict=True)
        x = torch.randn(3, 3, 288, 160)
        with torch.no_grad():
            torch.testing.assert_close(ours(x), hf(pixel_values=x).last_hidden_state, rtol=1e-5, atol=1e-5)

    def test_encoder_keeps_patch_tokens_only(self):
        """CameraEncoder loads a local HF-layout checkpoint, drops the CLS and register
        tokens (540 = 3 cameras x 180 patches per frame, cameras concatenated in
        order), runs in float32 on CPU, rejects images of the wrong size, and is shared
        (not copied) by deepcopy."""
        import copy
        import json
        import torch
        from safetensors.torch import save_file
        from drivevla.training.vision import CameraEncoder

        cfg = _tiny_dinov3_config()
        vit = self._random_vit(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "config.json").write_text(json.dumps(cfg))
            save_file({k: v.contiguous() for k, v in vit.state_dict().items()}, str(Path(tmp) / "model.safetensors"))
            enc = CameraEncoder(tmp)
        self.assertEqual((enc.hidden_size, enc.patches_per_camera, enc.num_tokens), (8, 180, 540))
        images = torch.randn(2, 3, 3, 288, 160)
        out = enc(images, torch.device("cpu"))
        self.assertEqual((tuple(out.shape), out.dtype), ((2, 540, 8), torch.float32))
        with torch.no_grad():
            ref = vit(images.flatten(0, 1))[:, 1 + 4 :, :].reshape(2, 540, 8)
        torch.testing.assert_close(out, ref, rtol=0, atol=0)
        with self.assertRaises(ValueError):
            enc(torch.randn(1, 3, 3, 224, 224), torch.device("cpu"))
        self.assertIs(copy.deepcopy({"enc": enc})["enc"], enc)


if __name__ == "__main__":
    unittest.main()
