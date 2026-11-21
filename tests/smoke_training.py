"""End-to-end CPU smoke test: one real training step for one recipe.

Usage (run by tests/test_training.py, ModelTests.test_full_training_forward_backward_and_save):

    python tests/smoke_training.py <work_dir> [nav|grpo]

The script fabricates a tiny 10-frame NAVSIM-style driving sequence under
<work_dir>/data (temporal infos, front-camera JPEGs, a tiny random DINOv3 checkpoint
and world-model rows with trajectories), builds the real
BEVLatentActionWorldModelDataset in the NAVSIM configuration, and then runs the real
training entry point
(drivevla.training.train.main, or drivevla.training.grpo.run for "grpo") with three
pieces patched:

  * the tokenizer is a small word-level tokenizer holding the DriveVLA special tokens;
  * LlavaQwenForCausalLM.from_pretrained returns a randomly initialised 1-layer,
    32-wide Qwen model;
  * LatentActionDebugTrainer.train (inherited by the GRPO trainer) runs exactly one
    forward/backward step (one_step) with extra checks.

train.main then saves the model as usual, and the saved safetensors are compared with
the weights captured during the step. Any failed check raises (non-zero exit); on
success the script prints a PARITY_OK line.

Both profiles read the camera images (--sensor-root); the frozen DINOv3 runs in the
model forward, and no DINOv3 weight may end up in the saved model.

Profiles (second argument):
  nav   NAVSIM SFT recipe flags (8-step horizon and waypoints, object / box loss
        weights of the camera-conditioned query pool); the default. Also checks that
        the forward encodes exactly what the camera encoder makes of the images.
  grpo  nav + GRPO flags, run through grpo.run with a fake inline PDM scorer; the
        frozen reference policy must share the camera encoder instead of copying it
"""

import os, sys, json, pickle, types
from pathlib import Path
from unittest.mock import patch

# Paths and process setup: repository root, the work directory and the profile given
# on the command line (the profile is echoed in the final PARITY_OK line).
root = Path(__file__).resolve().parents[1]
dest = Path(sys.argv[1])
dest.mkdir(exist_ok=True)
profile = sys.argv[2] if len(sys.argv) > 2 else "nav"
assert profile in ("nav", "grpo"), profile
grpo = profile == "grpo"
# Import from the repository (and scripts/) without installation; never contact the
# Hugging Face hub and never start W&B.
sys.path[:0] = [str(root), str(root / "scripts")]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["WANDB_MODE"] = "disabled"
import torch, numpy as np
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast
from llava.constants import DRIVEVLA_SPECIAL_TOKENS
from llava.model.language_model.llava_qwen import LlavaQwenConfig, LlavaQwenForCausalLM
import drivevla.training.train as train
from drivevla.training.data import BEVLatentActionWorldModelDataset as Data
from drivevla.training.vision import CAMERAS, IMAGE_HW, DINOv3ViT
from PIL import Image
from safetensors.torch import save_file

import accelerate.utils.other

# Make accelerate treat DeepSpeed as unavailable, so its model unwrapping
# (extract_model_from_parallel) never imports DeepSpeed in this CPU-only test; use one
# CPU thread.
accelerate.utils.other.is_deepspeed_available = lambda: False
torch.set_num_threads(1)
# Synthetic dataset directory with the camera image directory (NAVSIM sensor-blob
# layout: <log>/<camera>/<id>.jpg) inside it.
work = dest / "data"
work.mkdir(exist_ok=True)
sensors = work / "sensor_blobs"
# Tiny random DINOv3 checkpoint (Hugging Face layout: config.json + model.safetensors):
# 1 layer, 8 channels, 2 heads, 4 registers, 16 px patches, so a 288 x 160 camera gives
# the recipe's 180 patch tokens and a frame 540 tokens of width 8.
dino_dir = work / "dinov3"
dino_dir.mkdir(exist_ok=True)
dino_cfg = dict(model_type="dinov3_vit", hidden_size=8, intermediate_size=16, num_hidden_layers=1,
                num_attention_heads=2, num_register_tokens=4, patch_size=16, num_channels=3,
                layer_norm_eps=1e-5, rope_theta=100.0, hidden_act="gelu", key_bias=False)
torch.manual_seed(5)
_dino = DINOv3ViT(dino_cfg)
for _p in _dino.parameters():
    torch.nn.init.normal_(_p, std=0.5)
(dino_dir / "config.json").write_text(json.dumps(dino_cfg))
save_file({k: v.contiguous() for k, v in _dino.state_dict().items()}, str(dino_dir / "model.safetensors"))
rng = np.random.default_rng(3)
# Ten frames linked into one chain through prev/next tokens (32-hex tokens).
infos = []
rows = []
for i in range(10):
    tok = f"{i:032x}"
    # Temporal info of frame i: an 18-d can_bus vector, a 9 x 2 future trajectory, ego
    # features and history, a one-hot driving command, and one GT car box 10 m ahead.
    infos.append(
        {
            "token": tok,
            "prev_token": f"{i-1:032x}" if i else "",
            "next_token": f"{i+1:032x}" if i < 9 else "",
            "can_bus": np.arange(18, dtype=np.float32) / 10,
            "gt_ego_fut_trajs": np.arange(18, dtype=np.float32).reshape(9, 2) / 10,
            "gt_ego_lcf_feat": np.arange(9, dtype=np.float32),
            "gt_ego_his_diff": np.ones((4, 2), dtype=np.float32),
            "gt_ego_his_trajs": np.ones((4, 2), dtype=np.float32),
            "gt_ego_fut_cmd": np.array([0, 0, 1]),
            "gt_boxes": np.array([[0, 10, 0, 2, 4, 1.5, 0]], dtype=np.float32),
            "gt_names": np.array(["car"]),
            "gt_ids": np.array(["actor"]),
            # Front-camera images, relative to the sensor root.
            "cams": {c: {"data_path": f"log0/{c}/{tok}.jpg"} for c in CAMERAS},
        }
    )
    # Small random JPEG per camera (the loader resizes it to IMAGE_HW).
    for c in CAMERAS:
        (sensors / "log0" / c).mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 256, (36, 64, 3), dtype=np.uint8)).save(
            sensors / "log0" / c / f"{tok}.jpg", quality=90
        )
    # World-model row of frame i in the NAVSIM JSONL format, with an 8-waypoint
    # trajectory answer.
    rows.append(
        {
            "id": tok + "_wm",
            "sample_id": tok,
            "task_type": "wm",
            "conversations": [
                {"from": "human", "value": "drive"},
                {
                    "from": "gpt",
                    "value": "<traj_start>[(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)]<traj_end>",
                },
            ],
        }
    )
(work / "rows.json").write_text(json.dumps(rows))
# Temporal infos file ({"infos": [...]}, like navsim_infos_temporal_*.pkl) and the
# token -> info mapping used for prompt metadata (like cached_navsim_info.pkl).
with (work / "infos.pkl").open("wb") as f:
    pickle.dump({"infos": infos}, f)
with (work / "cached.pkl").open("wb") as f:
    pickle.dump({r["token"]: r for r in infos}, f)
# Word-level tokenizer whose vocabulary is just the control tokens, the DriveVLA special
# tokens, <BEV_STATE> and the Qwen chat markers; everything else maps to [UNK].
vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
for s in DRIVEVLA_SPECIAL_TOKENS + ["<BEV_STATE>", "<|im_start|>", "<|im_end|>"]:
    vocab.setdefault(s, len(vocab))
tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
tok.pre_tokenizer = WhitespaceSplit()
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tok,
    unk_token="[UNK]",
    pad_token="[PAD]",
    eos_token="[EOS]",
    additional_special_tokens=list(vocab)[3:],
    model_max_length=4096,
)
# Dataset arguments mirroring the NAVSIM recipes: 8-step horizon and waypoints, 19-wide
# ego vector plus the step channel, and camera-query object targets. The visual input
# is the camera images (540 DINOv3 tokens per frame).
bev_pred_len = 540
kwargs = dict(
    data_path=str(work / "rows.json"),
    sensor_root=str(sensors),
    ann_file=str(work / "infos.pkl"),
    cached_info_pkl=str(work / "cached.pkl"),
    bev_pred_len=bev_pred_len,
    bev_horizon_steps="8",
    ik_num_waypoints=8,
    ego_dim=19,
    use_camera_query_targets=True,
)
nav = Data(tokenizer, **kwargs)
# Build every item once (exercises __getitem__ end to end). The training step below
# adds its forward / gradient / weight snapshots to this dict.
records = {
    "nav": [
        (
            x["input_ids"],
            x["labels"],
            x["gt_traj"],
            {k: v for k, v in x.items() if k.startswith("wm_")},
            x["uniad_data"],
            x.get("query_box_targets"),
        )
        for x in nav
    ],
}
# Every item has an 8 x 2 IK target and, except the first frame, a "Previous action"
# line with the previous frame's GT trajectory text.
assert all(r[2] is not None and tuple(r[2].shape) == (8, 2) for r in records["nav"])
assert "Previous action: <traj_start>[No Previous Trajectory]<traj_end>" in records["nav"][0][3]["wm_question"]
assert "<traj_start>[(0,1),(0,2),(0,3),(0,4),(0,5),(0,6),(0,7),(0,8)]<traj_end>" in records["nav"][1][3]["wm_question"]
# Every frame carries the three preprocessed front-camera images.
_x = nav[0]
assert tuple(_x["uniad_pth"]["camera_images"].shape) == (3, 3, *IMAGE_HW)
assert all(tuple(p["camera_images"].shape) == (3, 3, *IMAGE_HW) for p in _x["next_uniad_pths"])


# Run the complete training setup with a tiny local Qwen, then one real forward/backward.
def model_from_pretrained(*a, **kw):
    """Replacement for LlavaQwenForCausalLM.from_pretrained: a seeded, randomly
    initialised 1-layer Qwen (hidden size 32, 4 heads, 2 KV heads) sized to the test
    tokenizer, with eager attention so it runs on CPU. All arguments are ignored."""
    torch.manual_seed(13)
    cfg = LlavaQwenConfig(
        vocab_size=len(tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        pad_token_id=1,
        eos_token_id=2,
        attention_dropout=0.0,
    )
    cfg._attn_implementation = "eager"
    return LlavaQwenForCausalLM(cfg)


def one_step(self, *a, **kw):
    """Replacement for Trainer.train: one checked forward/backward step.

    Computes the full training loss on the first training sample, checks it (and, for
    GRPO, that exactly one preference pair was formed with a working reference policy),
    back-propagates it while recording outputs, gradients and weights in `records`, and
    finally checks that the IK loss alone only reaches the IK head (stop-grad).
    """
    # GRPO profile: swap the inline PDM scorer, which needs a NAVSIM metric cache, for a
    # deterministic fake.
    if hasattr(self, "_grpo_cfg"):

        class FakePDM:
            """Stand-in for InlinePDMScorer: every scene is known, and the G samples get
            strictly increasing scores from 0 to 1, so best and worst always differ."""

            def has_token(self, token):
                """Every scene token is treated as present in the metric cache."""
                return True

            def score_batch(self, samples, token):
                """Score sample i of G as i / (G - 1)."""
                return [{"score": float(i) / (len(samples) - 1)} for i in range(len(samples))]

            def aggregate_pdm_score(self, score):
                """Use the per-sample score directly as the aggregate PDM score."""
                return score["score"]

        self._pdm_inline = FakePDM()

    # Callbacks do not run because Trainer.train is replaced, so do what the curriculum
    # callback would do: switch to stage 2 with the IK loss enabled. model.ik_stop_grad
    # keeps the value from --ik-stop-grad (True), which the final check relies on.
    self.model.curriculum_stage = 2
    self.model.ik_loss_weight = 1.0
    # Collate the first training sample into a batch and compute the full loss
    # (LM + next visual state + IK, plus the GRPO term for the grpo profile).
    batch = self.data_collator([self.train_dataset[0]])
    # SFT: the forward's frame encoding gives exactly the camera encoder's tokens of
    # the images (current frame and horizon frame, encoded in one batch), and leaves
    # the batch untouched.
    if not grpo:
        cur, nxt = self.model._encode_camera_frames(batch["uniad_pth"], batch["next_uniad_pths"])
        assert tuple(cur["bev_tokens"].shape) == (540, 8) and "camera_images" not in cur
        assert "camera_images" in batch["uniad_pth"]
        enc = self.model.camera_encoder
        ref = enc(torch.stack([batch["uniad_pth"]["camera_images"], batch["next_uniad_pths"][0]["camera_images"]]), "cpu")
        torch.testing.assert_close(cur["bev_tokens"], ref[0], rtol=0, atol=0)
        torch.testing.assert_close(nxt[0]["bev_tokens"], ref[1], rtol=0, atol=0)
    torch.manual_seed(71)
    loss, outputs = self.compute_loss(self.model, batch, return_outputs=True)
    assert torch.isfinite(loss), loss
    # GRPO: the frozen reference (a deep copy of the policy) shares the encoder.
    if grpo:
        assert self._ref_model is not None
        assert self._ref_model.camera_encoder is self.model.camera_encoder
    # GRPO: exactly one best-vs-worst pair was used, and the reference policy produced
    # log-probabilities (no "ref_missing" fallback to zeros).
    if hasattr(self, "_grpo_cfg"):
        logs = [r for r in self.state.log_history if "grpo/n_pairs" in r]
        assert logs and logs[-1]["grpo/n_pairs"] == 1, self.state.log_history
        assert "grpo/ref_missing" not in logs[-1], logs[-1]
    # Back-propagate the total loss (keeping the graph for the IK-only pass below) and
    # snapshot the tensor outputs, the gradients and the weights.
    loss.backward(retain_graph=True)
    records["forward"] = {k: v.detach().clone() for k, v in outputs.items() if torch.is_tensor(v)}
    records["grads"] = {
        k: p.grad.detach().clone() for k, p in self.model.named_parameters() if p.grad is not None
    }
    records["state"] = {k: p.detach().clone() for k, p in self.model.state_dict().items()}
    # Stop-grad check: the IK loss alone produces gradients in the IK head and none in
    # any other parameter, i.e. the LLM features given to the IK head are detached.
    self.model.zero_grad(set_to_none=True)
    outputs["ik_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in self.model.ik_head.parameters()
    )
    assert all(
        p.grad is None or p.grad.abs().sum() == 0
        for k, p in self.model.named_parameters()
        if not k.startswith("ik_head.")
    )
    # Mimic Trainer.train's TrainOutput (train.main does not use the return value).
    return types.SimpleNamespace(metrics={})


# Command line of the NAVSIM SFT profile: the synthetic data, a tiny IK head (10
# diffusion steps, 16-wide decoder with 4 heads), fp32 without gradient checkpointing
# and eager attention for CPU, exactly one step, and the recipes' stop-grad and NAVSIM
# flags. argv[0] is only the program name.
argv = [
    "train",
    # Model (patched loader) and synthetic data.
    "--model-path",
    "dummy",
    "--data",
    str(work / "rows.json"),
    "--sensor-root",
    str(sensors),
    "--dinov3-model",
    str(dino_dir),
    "--ann-file",
    str(work / "infos.pkl"),
    "--cached-info-pkl",
    str(work / "cached.pkl"),
    "--output-dir",
    str(dest / "model"),
    # Visual-state settings matching the DINOv3 tokens.
    "--bev-pred-len",
    str(bev_pred_len),
    "--bev-horizon-steps",
    "8",
    # Tiny diffusion IK head.
    "--ik-num-steps",
    "10",
    "--ik-decoder-dim",
    "16",
    "--ik-decoder-heads",
    "4",
    # CPU-friendly precision / memory settings and a single step.
    "--no-fp16",
    "--no-gradient-checkpointing",
    "--attn-implementation",
    "eager",
    "--max-steps",
    "1",
    # Recipe flags shared by all launchers.
    "--ik-stop-grad",
    # NAVSIM recipe flags.
    "--ik-num-waypoints",
    "8",
    "--obj-loss-weight",
    "0.25",
    "--box-loss-weight",
    "0.25",
    "--obj-loss-weight-next",
    "0.15",
    "--box-loss-weight-next",
    "0.15",
]
# The SFT profile calls train.main() directly; it parses the patched sys.argv.
run_training = train.main
# grpo profile: add the RFT recipe's GRPO flags (with a shorter 5-step sampling chain)
# and run grpo.run on them. ODVLA_STRICT_REWARD=1 makes an exception in the reward /
# GRPO loss fail the test instead of being logged and degrading the step to SFT only.
if grpo:
    from drivevla.training.config import build_argparser, add_grpo_args
    from drivevla.training.grpo import run

    os.environ["ODVLA_STRICT_REWARD"] = "1"
    argv += [
        "--grpo-group-size",
        "4",
        "--grpo-loss-weight",
        "1",
        "--grpo-diff-num-steps",
        "5",
        "--grpo-sigma-min-sample",
        "0.04",
        "--grpo-sigma-min-logprob",
        "0.05",
        "--grpo-beta",
        "0.2",
    ]
    parser = build_argparser()
    add_grpo_args(parser)
    run_training = lambda: run(parser.parse_args(argv[1:]))
# Run the real setup with the tokenizer and model loaders, Trainer.train and sys.argv
# patched; train.main then saves the model to <work_dir>/model.
with (
    patch.object(train.AutoTokenizer, "from_pretrained", return_value=tokenizer),
    patch.object(LlavaQwenForCausalLM, "from_pretrained", side_effect=model_from_pretrained),
    patch.object(train.LatentActionDebugTrainer, "train", one_step),
    patch.object(sys, "argv", argv),
):
    run_training()
# The dataset kept all 10 driving rows.
assert len(nav) == 10
# The model has an IK head, the final save wrote model.safetensors, and the saved
# tensors are bit-identical to the weights captured during the step.
assert any(k.startswith("ik_head.") for k in records["state"])
assert (dest / "model/model.safetensors").is_file()
from safetensors.torch import load_file

restored = load_file(str(dest / "model/model.safetensors"))
for k, v in records["state"].items():
    torch.testing.assert_close(v, restored[k], rtol=0, atol=0)
# The frozen DINOv3 is not part of the model's weights: nothing of it is saved.
assert not any("register_tokens" in k or "camera_encoder" in k or "layer_scale" in k for k in restored)
# Success line: label, dataset sizes, number of state tensors and of parameters that
# received gradients.
print("PARITY_OK", profile, len(nav), len(records["state"]), len(records["grads"]))
