"""Camera inputs, temporal links and collation for NAVSIM.

This module is the data side of DriveIK training and evaluation. It turns the
world-model JSONL annotations, the per-frame infos pickles and the camera images
into single training samples for the
LLaVA-style Qwen driving VLA and its Inverse Kinematics (IK) diffusion head.

Pipeline overview
-----------------
* **Visual states.** The visual state of a frame (sample token) is a ``(540, 384)``
  set of frozen DINOv3 ViT-S/16 tokens: three front cameras (CAM_FRONT_LEFT,
  CAM_FRONT, CAM_FRONT_RIGHT) x 180 patches, 384 channels. The dataset delivers the
  three camera images, read from ``sensor_root`` with the paths in the infos pickle
  and preprocessed (``{"camera_images": (3, 3, 288, 160)}``); the model forward runs
  DINOv3 on them (:mod:`vision`) and projects the tokens with ``mm_projector_bev``
  (see :func:`ensure_mm_projectors`) into the LLM hidden size.
* **Temporal links.** The ``ann_file`` infos pickle provides ``token -> prev`` and
  ``token -> next`` links (:func:`load_temporal_links_and_cameras`). The dataset
  follows ``next`` links up to ``k`` hops (``--bev-horizon-steps``) through frames
  that have their visual input to find the frame whose visual state becomes the
  *next-visual-state* regression target. NAVSIM frames are 0.5 s apart, so the
  recipe's 8 hops = 4 s.
* **Prompt.** The user turn is built with the v3 prompt builder
  (``prompts.build_drivevla_prompt_question_v3``): ego status, historical trajectory,
  mission goal and previous action. The ``<SCENE>`` and
  ``<EGO>`` placeholders live in the v3 *system* prompt (``llava/conversation.py``).
  The assistant turn is one ``<bev_start> <BEV_STATE> x K <bev_end>`` block per
  horizon (K = ``bev_pred_len``, 540 in the recipes).
* **IK supervision.** ``gt_traj`` is the ground-truth future ego trajectory
  ``(ik_num_waypoints, 2)`` in meters at 0.5 s spacing (8 waypoints / 4 s in the
  NAVSIM recipes), read from the cached info pickle (``gt_ego_fut_trajs``).
* **Collation.** :class:`DataCollatorForLatentActionWM` supports batch size 1 only. It
  truncates to the effective max length, pads, builds the attention mask and forwards
  every side input (visual inputs, ego vector, GT trajectory, query targets, debug
  metadata) to the model.

Per-sample step conditioning: the ego vector gets one extra channel
``actual_k / max_k`` (hops actually reached / requested horizon). The NAVSIM infos
carry no ``can_bus``, so the rest of the ego vector is zeros; the ego status
reaches the model as prompt text.

Camera-query object targets (``use_camera_query_targets``, training set only): the
NAVSIM infos have no ``gt_boxes``, so the ``ObjectQueryHead`` targets are constant
"no object" tensors for every frame.
"""

from __future__ import annotations

# Standard library.
import json
import pickle
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

# Third-party.
import torch
import torch.nn as nn
import transformers

# Special-token strings shared with the tokenizer / model (llava/constants.py):
# trajectory and BEV-state block delimiters, and the label value (-100) that the
# cross-entropy loss ignores.
from llava.constants import (
    IGNORE_INDEX,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    DEFAULT_BEV_START_TOKEN,
    DEFAULT_BEV_END_TOKEN,
)

# Conversation templates (only "qwen_bev_latent_action_world_model_v3" exists), the
# tokenizer helper that maps <SCENE>/<EGO>/... placeholders to their negative
# placeholder ids, the model class (type hint only) and the projector factory.
from llava.conversation import conv_templates
from llava.mm_utils import tokenizer_uniad_token
from llava.model import LlavaQwenForCausalLM
from llava.model.multimodal_projector.builder import build_vision_projector
from llava.utils import rank0_print

# `wm` holds the shared prompt-meta loader.
from . import common as wm
from .prompts import build_drivevla_prompt_question_v3
from .vision import CAMERAS, load_camera_images

# Placeholder token repeated K (= bev_pred_len) times inside each <bev_start> ...
# <bev_end> block of the assistant answer. The LLM hidden states at these positions
# are regressed onto the projected DINOv3 tokens of the future frame. train.py
# adds it to the tokenizer and requires it to be a single token.
DEFAULT_BEV_STATE_TOKEN = "<BEV_STATE>"


def _extract_gt_fut_traj_tensor_from_info(
    info: Dict[str, Any], n: int = 6
) -> Optional[torch.Tensor]:
    """Extract a (N,2) future trajectory tensor from an info dict.

    Info dicts store `gt_ego_fut_trajs` with shape (T,2) or (T,>=2). When the table
    has one row more than requested, row 0 is taken to be the current-position
    anchor and skipped (indices [1..N]); the NAVSIM infos store (8, 2) without an
    anchor, so n=8 returns rows 0..7. The anchor is detected only by length
    (``T >= n + 1``); the row values are not checked.

    Args:
        info: per-token cached info dict (must contain `gt_ego_fut_trajs`).
        n:    number of future waypoints to extract (the NAVSIM recipes pass 8 =
              4 s @ 0.5 s).

    Returns:
        Tensor[n,2] on CPU (float32), or None if unavailable/too short. Units are
        meters in the dataset's ego frame; only the first two columns (x, y) are kept.
    """
    # Accept only dicts; read the plural key first, then the singular spelling.
    if not isinstance(info, dict):
        return None
    fut = info.get("gt_ego_fut_trajs", None)
    if fut is None:
        fut = info.get("gt_ego_fut_traj", None)
    if fut is None:
        return None
    # Convert to float32; anything non-numeric / ragged is treated as missing.
    try:
        t = torch.as_tensor(fut, dtype=torch.float32)
    except Exception:
        return None
    # Must be a (T, >=2) table of waypoints.
    if t.ndim != 2 or t.shape[1] < 2:
        return None
    n = int(n)
    # T >= n+1: assume row 0 is the current-position anchor and skip it.
    # T == n:   no anchor, take the first n rows.
    # T < n:    too short for the requested horizon.
    if t.shape[0] >= n + 1:
        pts = t[1 : n + 1, :2]
    elif t.shape[0] >= n:
        pts = t[:n, :2]
    else:
        return None
    # Final shape guard: exactly (n, 2).
    if pts.shape[0] != n or pts.shape[1] != 2:
        return None
    return pts.contiguous().cpu()


def load_token_to_gt_fut_traj_map(
    cached_info_pkl: Optional[str], *, n: int = 6
) -> Dict[str, torch.Tensor]:
    """Load {sample_token -> Tensor[n,2]} GT future trajectory for IK.

    The trajectories come from the cached info pkl (``cached_navsim_info.pkl``,
    a flat ``{sample_token -> info dict}``), which covers every NAVSIM train / val
    token.

    Args:
        cached_info_pkl: the cached-info pickle (``--cached-info-pkl``).
        n: number of future waypoints (8 in the NAVSIM recipes).

    Returns:
        ``{token -> float32 CPU Tensor[n, 2]}`` (meters, dataset ego frame). Entries
        whose key is not a non-empty string, whose value is not a dict, or that have
        no usable future trajectory are skipped.

    Side effects:
        Unpickles the cached-info file; logs failures and a final summary on rank 0.
        Never raises for load errors (a missing / unreadable file gives ``{}``).
    """
    out: Dict[str, torch.Tensor] = {}
    cached_path = None
    try:
        cp = Path(str(cached_info_pkl)) if cached_info_pkl else None
        if cp is not None and cp.exists():
            with cp.open("rb") as f:
                obj = pickle.load(f)
            # Flat {sample_token -> info dict}; any other pickle type is ignored.
            if isinstance(obj, dict):
                for tok, info in obj.items():
                    if not isinstance(tok, str) or not tok:
                        continue
                    if not isinstance(info, dict):
                        continue
                    pts = _extract_gt_fut_traj_tensor_from_info(info, n=n)
                    if pts is not None:
                        out[tok] = pts
            if out:
                cached_path = str(cp)
    except Exception as e:
        rank0_print(
            f"[WARN] Failed to load GT fut traj from cached_info_pkl={cached_info_pkl}: {type(e).__name__}: {e}"
        )
    # Summary: total tokens and the pickle they came from.
    rank0_print(
        f"[DATA] Loaded token->GT future traj for IK: tokens={len(out)} | cached={len(out)}{('@' + cached_path if cached_path else '')}"
    )
    return out


def _get_model_max_pos(config) -> Optional[int]:
    """Return the model's maximum sequence length from its HF config.

    Args:
        config: a ``transformers`` config (or ``None``).

    Returns:
        The first positive int among ``max_position_embeddings``, ``max_seq_len``,
        ``seq_length`` and ``n_positions`` (checked in that order), else ``None``.
    """
    for k in ["max_position_embeddings", "max_seq_len", "seq_length", "n_positions"]:
        v = getattr(config, k, None)
        if isinstance(v, int) and v > 0:
            return int(v)
    return None


def _compute_effective_max_len(
    tokenizer, model
) -> Tuple[Optional[int], Optional[int], Optional[int]]:
    """Compute the token length the collator truncates to.

    Args:
        tokenizer: HF tokenizer; ``model_max_length`` is used if it is a real limit.
        model: the model; its ``config`` provides the position-embedding limit.

    Returns:
        ``(tok_max_effective, model_max, effective)``: the tokenizer limit (``None``
        if unset or > 100000, which is HF's "unlimited" sentinel ~1e30), the model
        limit, and the smaller of the two (``None`` if neither is known). train.py
        passes ``effective`` to :class:`DataCollatorForLatentActionWM` as
        ``max_length``.
    """
    # Tokenizer limit; values above 100000 are treated as "no limit".
    tok_max = getattr(tokenizer, "model_max_length", None)
    if isinstance(tok_max, int):
        tok_max_effective = None if tok_max > 100000 else tok_max
    else:
        tok_max_effective = None
    # Model limit from the config.
    model_max = _get_model_max_pos(getattr(model, "config", None))
    # The tighter of the known limits.
    cands = [v for v in [tok_max_effective, model_max] if isinstance(v, int) and v > 0]
    effective = min(cands) if len(cands) > 0 else None
    return (tok_max_effective, model_max, effective)


def _read_json_or_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSON or JSONL file (or a directory holding exactly one) into a list.

    Args:
        path: file path, or a directory that contains exactly one ``*.json`` /
            ``*.jsonl`` file.

    Returns:
        List of records. A ``.jsonl`` file gives one record per non-empty line. A JSON
        file holding a dict gives ``[dict]``; holding a list gives that list. If a
        non-``.jsonl`` file fails to parse as JSON, it is re-read as JSONL (with a
        warning).

    Raises:
        FileNotFoundError: the path does not exist, or the directory has zero or
            several candidate files.
    """
    # Directory input: resolve to its single JSON/JSONL file.
    p = Path(path)
    if p.is_dir():
        cands = sorted(list(p.glob("*.jsonl"))) + sorted(list(p.glob("*.json")))
        if len(cands) == 1:
            p = cands[0]
        else:
            raise FileNotFoundError(f"Ambiguous dir: {p} | candidates={len(cands)}")
    if not p.exists():
        raise FileNotFoundError(f"data file not found: {p}")
    # JSONL by extension: one JSON object per non-blank line.
    if p.suffix.lower() == ".jsonl":
        rows: List[Dict[str, Any]] = []
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    # Otherwise try a single JSON document first.
    try:
        obj = json.loads(p.read_text())
        if isinstance(obj, dict):
            return [obj]
        return list(obj)
    # A ".json" file that is actually line-delimited: fall back to JSONL parsing.
    except json.JSONDecodeError:
        rows: List[Dict[str, Any]] = []
        with p.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        rank0_print(f"[WARN] Failed to parse as JSON; fell back to JSONL: {p}")
        return rows


def _infer_sample_token(row: Dict[str, Any]) -> str:
    """Return the frame (sample) token a data row refers to.

    Args:
        row: one world-model JSONL row.

    Returns:
        The first non-empty string among ``sample_id``, ``sample_token``, ``token``.
        Otherwise the prefix before the first ``_`` of ``id`` (e.g.
        ``"<token>_wm"`` -> ``"<token>"``). ``""`` if nothing is found.
    """
    # Explicit token fields first.
    for k in ("sample_id", "sample_token", "token"):
        v = row.get(k, None)
        if isinstance(v, str) and v:
            return v
    # Fall back to the "<token>_<suffix>" id convention.
    rid = row.get("id") or ""
    if isinstance(rid, str) and rid:
        return rid.split("_", 1)[0]
    return ""


def _extract_question_answer(row: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(question, answer)`` text of a row.

    Args:
        row: a record in LLaVA chat form (``conversations`` list; the first two
            turns are taken as question / answer).

    Returns:
        ``(question, answer)`` as strings; ``("", "")`` if there is no
        ``conversations`` list. For world-model JSONL rows the answer is the
        ``<traj_start>...<traj_end>`` ground-truth trajectory text.
    """
    # LLaVA-style conversation: turn 0 = human, turn 1 = assistant.
    convs = row.get("conversations", None)
    if isinstance(convs, list) and len(convs) >= 2:
        q = convs[0].get("value", "") if isinstance(convs[0], dict) else ""
        a = convs[1].get("value", "") if isinstance(convs[1], dict) else ""
        return (str(q or ""), str(a or ""))
    return ("", "")


def _extract_traj_only(action: str) -> str:
    """Cut out the ``<traj_start>...<traj_end>`` span (tags included) from a string.

    Args:
        action: text that may contain a trajectory span among other content.

    Returns:
        The stripped span from the first ``<traj_start>`` through the first
        ``<traj_end>``; if the tags are missing or out of order, the whole stripped
        input. ``None`` becomes ``""``.
    """
    a = "" if action is None else str(action)
    s = a.find(DEFAULT_TRAJ_START_TOKEN)
    e = a.find(DEFAULT_TRAJ_END_TOKEN)
    # Both tags present in the right order: include the end tag in the slice.
    if s >= 0 and e > s:
        e = e + len(DEFAULT_TRAJ_END_TOKEN)
        return a[s:e].strip()
    return a.strip()


def _repeat_bev_state_tokens(k: int, token: str = DEFAULT_BEV_STATE_TOKEN) -> str:
    """Return ``token`` repeated ``k`` times, space separated (the body of one BEV
    block, e.g. 540 ``<BEV_STATE>`` placeholders)."""
    return " ".join([token] * int(k))


def load_token_prev_next_map_from_infos_pkl(
    ann_file: str,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Load {token->prev} and {token->next} from an infos pkl.

    The NAVSIM infos pickle names the links ``prev_token`` / ``next_token``.

    Args:
        ann_file: infos pickle ``{"infos": [...]}`` (``navsim_infos_temporal_*.pkl``).

    Returns:
        ``(token_to_prev, token_to_next)``. Every frame with a string token gets an
        entry in both maps; a missing / non-string link is stored as ``""`` (start or
        end of a scene / log).

    Raises:
        FileNotFoundError: ``ann_file`` does not exist.
        ValueError: the pickle has no infos list.
    """
    prev, nxt, _ = load_temporal_links_and_cameras(ann_file, cameras=())
    return (prev, nxt)


def load_temporal_links_and_cameras(
    ann_file: str, cameras: Sequence[str]
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Tuple[str, ...]]]:
    """Load the prev/next links and the camera image paths in one pass over the pkl.

    Args:
        ann_file: infos pickle ``{"infos": [...]}`` (``navsim_infos_temporal_*.pkl``).
        cameras: camera names to collect from ``info["cams"][name]["data_path"]``
            (paths relative to the split's sensor-blob directory, e.g.
            ``<log>/CAM_F0/<id>.jpg``). Empty: collect none.

    Returns:
        ``(token_to_prev, token_to_next, token_to_cams)`` as in
        :func:`load_token_prev_next_map_from_infos_pkl`, plus ``token -> one relative
        path per camera`` for frames that list every requested camera.

    Raises:
        FileNotFoundError: ``ann_file`` does not exist.
        ValueError: the pickle has no infos list.
    """
    # Load the pickle and locate the infos list.
    p = Path(ann_file)
    if not p.exists():
        raise FileNotFoundError(f"ann_file not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)
    infos = obj.get("infos") if isinstance(obj, dict) else None
    if not isinstance(infos, list):
        raise ValueError(f"Unsupported ann_file format: {p}")
    if len(infos) == 0:
        rank0_print(f"[WARN] ann_file has empty infos list: {p} — returning empty prev/next maps")
        return ({}, {}, {})
    token_to_prev: Dict[str, str] = {}
    token_to_next: Dict[str, str] = {}
    token_to_cams: Dict[str, Tuple[str, ...]] = {}
    cameras = tuple(cameras)
    no_prev = 0
    no_next = 0
    no_cams = 0
    # One entry per frame; count frames without a prev / next link for the log.
    for info in infos:
        if not isinstance(info, dict):
            continue
        tok = info.get("token") or info.get("sample_token")
        prv = info.get("prev_token") or ""
        nxt = info.get("next_token") or ""
        if not isinstance(tok, str) or tok == "":
            continue
        # Non-string links are treated as missing.
        if not isinstance(prv, str):
            prv = ""
        if not isinstance(nxt, str):
            nxt = ""
        if prv == "":
            no_prev += 1
        if nxt == "":
            no_next += 1
        token_to_prev[tok] = prv
        token_to_next[tok] = nxt
        if cameras:
            cams = info.get("cams") or {}
            rel = [
                cams[c].get("data_path", "") if isinstance(cams.get(c), dict) else ""
                for c in cameras
            ]
            if all(isinstance(r, str) and r for r in rel):
                token_to_cams[tok] = tuple(rel)
            else:
                no_cams += 1
    cam_msg = f", camera paths={len(token_to_cams)} (missing={no_cams})" if cameras else ""
    rank0_print(
        f"[DATA] Loaded token->prev/next from {p} : {len(token_to_next)} tokens (no-prev={no_prev}, no-next={no_next}){cam_msg}"
    )
    return (token_to_prev, token_to_next, token_to_cams)


def _projector_matches(module: Optional[nn.Module], in_features: int, out_features: int) -> bool:
    """Check whether an existing projector maps ``in_features -> out_features``.

    Args:
        module: an ``nn.Linear``, an ``nn.Sequential`` whose first element is an
            ``nn.Linear`` (MLP projectors), or ``None``.
        in_features: required input width (DINOv3 channel count).
        out_features: required output width (LLM hidden size).

    Returns:
        True if the (first) Linear's ``in_features`` and the (last) Linear's
        ``out_features`` match. False for ``None``, other module types (e.g. the
        ``qwen2_5_rms`` projector, which starts with an RMSNorm) or a mismatch.
    """
    if module is None:
        return False
    # Single linear projector.
    if isinstance(module, nn.Linear):
        return module.in_features == in_features and module.out_features == out_features
    # MLP projector: compare the first Linear's input and the last Linear's output.
    if isinstance(module, nn.Sequential) and len(module) > 0 and isinstance(module[0], nn.Linear):
        first = module[0]
        last = None
        # Last Linear layer (skipping trailing activations / norms).
        for m in reversed(module):
            if isinstance(m, nn.Linear):
                last = m
                break
        if last is None:
            return False
        return first.in_features == in_features and last.out_features == out_features
    return False


def ensure_mm_projectors(model: LlavaQwenForCausalLM, mm_hidden_size: int) -> None:
    """Make sure the model has a visual projector sized for the DINOv3 tokens.

    Called once by train.py after the DINOv3 encoder is loaded. After the call,
    ``mm_projector_bev`` maps ``mm_hidden_size`` (DINOv3 channels, e.g. 384) to
    ``config.hidden_size`` (LLM width), and ``mm_projector_scene`` / ``mm_projector``
    are attribute aliases of that same module, so every code path that projects
    scene features uses one set of weights.

    Args:
        model: the LLaVA-Qwen model (outer CausalLM; ``get_model()`` returns the inner
            module that owns the projectors).
        mm_hidden_size: channel count C of the DINOv3 tokens (``CameraEncoder.hidden_size``).

    Side effects:
        * Sets ``config.use_mm_proj = True`` and ``config.mm_hidden_size``; sets
          ``config.mm_projector_type = "linear"`` if the config has none.
        * Deletes any existing ``mm_projector`` (registered submodule or plain
          attribute).
        * If ``mm_projector_bev`` is not a registered submodule: reuses a registered
          ``mm_projector_scene`` with matching shape (stored as a plain attribute, so
          its weights stay registered only under ``mm_projector_scene``), otherwise
          builds and registers a new projector via ``build_vision_projector``.
        * Moves registered projectors to the device/dtype of the first model parameter.
        * Freezes ``mm_projector_track`` / ``mm_projector_map`` (not used by DriveIK).

    Raises:
        RuntimeError: no ``mm_projector_bev`` could be found or created.
    """
    # Record the projector input width in the config (build_vision_projector reads
    # mm_hidden_size / hidden_size / mm_projector_type from it; the config is also
    # saved with checkpoints).
    cfg = model.config
    cfg.use_mm_proj = True
    cfg.mm_hidden_size = int(mm_hidden_size)
    if not hasattr(cfg, "mm_projector_type"):
        cfg.mm_projector_type = "linear"
    mm = model.get_model()
    # Remove the generic LLaVA projector (registered or plain attribute); it is
    # re-pointed to the BEV projector below.
    if "mm_projector" in getattr(mm, "_modules", {}):
        del mm._modules["mm_projector"]
    if "mm_projector" in mm.__dict__:
        del mm.__dict__["mm_projector"]
    # Get a BEV projector: reuse a shape-compatible scene projector (as an
    # unregistered alias, so its parameters are not duplicated in the state dict) or
    # build and register a fresh one (config.mm_projector_type, "linear" by default).
    if "mm_projector_bev" not in mm._modules:
        scene = mm._modules.get("mm_projector_scene", None)
        if _projector_matches(scene, cfg.mm_hidden_size, cfg.hidden_size):
            mm.__dict__["mm_projector_bev"] = scene
        else:
            mm.mm_projector_bev = build_vision_projector(cfg)
    # Look it up in either place (registered submodule or plain attribute).
    bev_proj = mm._modules.get("mm_projector_bev", None) or mm.__dict__.get(
        "mm_projector_bev", None
    )
    if bev_proj is None:
        raise RuntimeError("Failed to create mm_projector_bev")
    # Alias the scene / generic projector names to the BEV projector. Writing into
    # __dict__ makes plain attribute access (mm.mm_projector_scene) return bev_proj
    # without registering it a second time.
    mm.__dict__["mm_projector_scene"] = bev_proj
    mm.__dict__["mm_projector"] = bev_proj
    # Match device / dtype of the rest of the model (nn.Module.to is in place, so the
    # aliases above follow). StopIteration = model without parameters -> skip.
    try:
        ref = next(model.parameters())
        device, dtype = (ref.device, ref.dtype)
        for name in [
            "mm_projector_bev",
            "mm_projector_scene",
            "mm_projector_track",
            "mm_projector_map",
        ]:
            if name in mm._modules and isinstance(mm._modules[name], nn.Module):
                mm._modules[name] = mm._modules[name].to(device=device, dtype=dtype)
    except StopIteration:
        pass
    # Track / map projectors are unused here (the v3 prompt has no <TRACK>/<MAP>
    # inputs): freeze them.
    for proj_name in ["mm_projector_track", "mm_projector_map"]:
        proj = mm._modules.get(proj_name, None)
        if proj is not None:
            for p in proj.parameters():
                p.requires_grad = False


def _zero_traj_str() -> str:
    """Placeholder used in the prompt when there is no previous action.

    Returns:
        ``"<traj_start>[No Previous Trajectory]<traj_end>"``: a text marker, not a
        list of zero waypoints.
    """
    z6 = "[No Previous Trajectory]"
    return f"{DEFAULT_TRAJ_START_TOKEN}{z6}{DEFAULT_TRAJ_END_TOKEN}"


def _format_prev_action(prev_action: str) -> str:
    """Normalise a trajectory string to a single ``<traj_start>...<traj_end>`` span.

    Used for the "Previous action:" prompt line and to normalise the current-frame
    GT trajectory text before parsing it in ``__getitem__``.

    Args:
        prev_action: trajectory text, with or without the trajectory tags; may be
            empty / ``None``.

    Returns:
        ``_zero_traj_str()`` for empty input; the tagged span cut out by
        :func:`_extract_traj_only` if both tags are present; otherwise the stripped
        text wrapped in the tags.
    """
    a = (prev_action or "").strip()
    # No previous action (start of a scene, or blanked all-zero trajectory).
    if not a:
        return _zero_traj_str()
    # Already tagged: keep only the tagged span.
    if DEFAULT_TRAJ_START_TOKEN in a and DEFAULT_TRAJ_END_TOKEN in a:
        return _extract_traj_only(a)
    # Bare coordinate list: add the tags.
    return f"{DEFAULT_TRAJ_START_TOKEN}{a}{DEFAULT_TRAJ_END_TOKEN}"


class BEVLatentActionWorldModelDataset(torch.utils.data.Dataset):
    """Transition dataset for latent-action world model.

    Each item provides:
      * current token, prev token, next token
      * current visual input (uniad_pth)
      * one next visual input per horizon (next_uniad_pths)
      * gt trajectory for current token (gt_traj)  [used for IK training]

    Each frame's visual input is ``{"camera_images": float32 (3, 3, 288, 160)}``, the
    three front cameras read from ``<sensor_root>/<info["cams"][cam]["data_path"]>``
    and preprocessed by :func:`vision.load_camera_images`; the model forward encodes
    them with the frozen DINOv3 (:class:`vision.CameraEncoder`).

    Conversation:
      User      : current scene + ego + prev action
      Assistant : <bev_start> <BEV_STATE>*K <bev_end>

    Details:
      * The current scene (``<SCENE>``) and ego vector (``<EGO>``) placeholders are in
        the v3 system prompt; the user turn holds the ego-status text, historical
        trajectory, mission goal and "Previous action:" line (see
        :meth:`_build_user_prompt`).
      * There is one ``<bev_start> ... <bev_end>`` block per entry of
        ``bev_horizon_steps`` (one block of 8 hops in the NAVSIM recipes). The
        hidden states at the ``<BEV_STATE>`` positions are regressed onto the
        projected features of the frame reached by following up to that many
        next-links through frames with visual input (``next_uniad_pths``).
      * ``gt_traj`` (``(ik_num_waypoints, 2)`` float32, meters, 0.5 s spacing) is the
        IK diffusion target. It is ``None`` when no trajectory is known.
      * Every item is one transition; the dataset is meant for batch size 1 (see
        :class:`DataCollatorForLatentActionWM`).
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        data_path: str,
        ann_file: str,
        *,
        sensor_root: str,
        bev_pred_len: int = 625,
        ego_dim: int = 18,
        bev_horizon_steps: str = "2,4,6",
        cached_info_pkl: Optional[str] = None,
        ik_num_waypoints: int = 8,
        use_camera_query_targets: bool = False,
    ):
        """Load annotations, link frames in time and pre-compute per-token lookups.

        Args:
            tokenizer: HF tokenizer with the DriveIK special tokens added.
            data_path: world-model JSON/JSONL (``opendrivevla_bev_worldmodel_*.jsonl``).
                Each row has ``sample_id``, ``task_type`` (``"wm"``) and a
                ``conversations`` pair whose answer is the GT trajectory text
                ``<traj_start>[(x1,y1),...]<traj_end>``.
            ann_file: infos pickle (``*_infos_temporal_{train,val}.pkl``) giving the
                prev/next links and the camera image paths.
            sensor_root: camera image directory of this split (NAVSIM
                ``sensor_blobs/trainval`` or ``sensor_blobs/test``); the paths in the
                infos pickle are relative to it.
            bev_pred_len: K, number of ``<BEV_STATE>`` placeholders per BEV block
                (the recipes pass 540 = number of DINOv3 tokens per frame).
            ego_dim: width of the ego vector before the step channel (the NAVSIM
                infos carry no can_bus, so these ``ego_dim`` values are zeros).
            bev_horizon_steps: comma-separated hop counts, one BEV block per entry
                (the NAVSIM recipes pass ``"8"``; frames are 0.5 s apart).
                Empty -> ``[2, 4, 6]``.
            cached_info_pkl: ``cached_navsim_info.pkl``; source of the prompt meta
                and the GT trajectories.
            ik_num_waypoints: GT trajectory length for IK (8 in the NAVSIM recipes).
            use_camera_query_targets: emit the 4-query object box targets for the
                ``ObjectQueryHead`` (training set only). The NAVSIM infos have no
                ``gt_boxes``, so they are always "no object".

        The prompt always has a "Previous action" line showing the previous
        frame's GT future trajectory (see ``__getitem__``).

        The dataset index keeps every row whose current frame has its visual input
        (all three camera images); horizon targets are walked
        through such frames only (a row is never dropped for a short chain), and the
        ego vector always gets the step channel.

        Raises:
            ValueError: no ``sensor_root`` given.
            FileNotFoundError: no row has its camera images under ``sensor_root``.

        Side effects:
            Reads the JSONL and the infos / cached-info pickles; logs statistics on
            rank 0.
        """
        # Camera image directory of this split (DINOv3 runs in the model forward).
        if not sensor_root:
            raise ValueError("sensor_root (camera image directory) is required.")
        self.sensor_root = str(sensor_root)
        # token -> whether its visual input exists (filled lazily; the index build
        # below asks for the same frames up to max(bev_horizon_steps) times).
        self._frame_available: Dict[str, bool] = {}
        # Basic configuration.
        self.tokenizer = tokenizer
        self.bev_pred_len = int(bev_pred_len)
        # IK target length (at least 1) and camera-query target switch.
        self.ik_num_waypoints = max(1, int(ik_num_waypoints))
        self.use_camera_query_targets = bool(use_camera_query_targets)
        self.cached_info_pkl = str(cached_info_pkl) if cached_info_pkl else None
        # Parse the BEV horizons ("8" -> [8], "2,4,6" -> [2, 4, 6]); empty input
        # falls back to [2, 4, 6] and negative values are clamped to 0.
        self.bev_horizon_steps = [
            int(x) for x in str(bev_horizon_steps).split(",") if str(x).strip() != ""
        ]
        if len(self.bev_horizon_steps) == 0:
            self.bev_horizon_steps = [2, 4, 6]
        self.bev_horizon_steps = [max(0, int(k)) for k in self.bev_horizon_steps]
        # Normaliser for the step-conditioning channel: k_norm = actual_k / max_k,
        # so k_norm is in [0, 1].
        self._ego_k_denom = max(1, max(self.bev_horizon_steps))
        self.num_bev_horizons = int(len(self.bev_horizon_steps))
        # Read the world-model rows and index their GT trajectory text by token:
        # sample_id_to_traj[token] = "<traj_start>[(x,y),...]<traj_end>". It serves as
        # the previous frame's "Previous action" text. Rows
        # without a tagged trajectory answer are skipped.
        rows = _read_json_or_jsonl(data_path)
        self.sample_id_to_traj: Dict[str, str] = {}
        for r in rows:
            if not isinstance(r, dict):
                continue
            sid = _infer_sample_token(r)
            if not sid:
                continue
            _q, a = _extract_question_answer(r)
            a = (a or "").strip()
            if not a:
                continue
            if DEFAULT_TRAJ_START_TOKEN not in a or DEFAULT_TRAJ_END_TOKEN not in a:
                continue
            traj = _extract_traj_only(a)
            if traj:
                self.sample_id_to_traj[sid] = traj
        self.rows = rows
        # Temporal links between frames (token -> previous / next token) and
        # token -> relative image path of each of vision.CAMERAS.
        self.token_to_prev, self.token_to_next, self.token_to_cams = (
            load_temporal_links_and_cameras(ann_file, cameras=CAMERAS)
        )
        # Build the transition index. Each entry is a 5-tuple: (row index into
        # self.rows, prev token, current token, comma-joined horizon target tokens,
        # comma-joined hop counts actually reached).
        self.items: List[Tuple[int, str, str, str, str]] = []
        # Filtering counters, logged after the loop.
        stats = {
            "total": 0,
            "no_token": 0,
            "missing_cur_frame": 0,
            "kept": 0,
            "kept_zero_hop_horizon": 0,
        }
        for i, r in enumerate(self.rows):
            stats["total"] += 1
            if not isinstance(r, dict):
                stats["no_token"] += 1
                continue
            cur_tok = _infer_sample_token(r)
            if not cur_tok:
                stats["no_token"] += 1
                continue
            # Previous frame ("" at the start of a scene); used for the
            # "Previous action" prompt line.
            prev_tok = self.token_to_prev.get(cur_tok, "")
            # Only a missing current-frame visual input drops a row.
            if not self._has_frame(cur_tok):
                stats["missing_cur_frame"] += 1
                continue
            # For each horizon k, follow next-links only through frames that have
            # their visual input, up to k hops. If not even one hop is possible, the
            # target is the current frame itself (actual_k = 0).
            results = [self._advance_next_token(cur_tok, k) for k in self.bev_horizon_steps]
            next_toks = [t for (t, _ak) in results]
            actual_ks = [int(ak) for (_t, ak) in results]
            # Count kept rows with a zero-hop horizon (target = current frame).
            if any((ak == 0 for ak in actual_ks)):
                stats["kept_zero_hop_horizon"] += 1
            # Keep the transition (tokens / hop counts stored as CSV strings).
            self.items.append(
                (
                    i,
                    prev_tok,
                    cur_tok,
                    ",".join(next_toks),
                    ",".join([str(ak) for ak in actual_ks]),
                )
            )
            stats["kept"] += 1
        self.stats = stats
        src = f"camera images under {self.sensor_root}"
        rank0_print(f"[DATA] LatentAction WM transitions: {stats} | transitions={len(self.items)} | visual input: {src}")
        if stats["kept"] == 0 and stats["total"] > 0:
            raise FileNotFoundError(
                f"No {data_path} row has its visual input ({src}). Check that the "
                "directory belongs to this split (NAVSIM train: sensor_blobs/trainval, "
                "navtest: sensor_blobs/test)."
            )
        # Ego vector for the <EGO> embedding: navsim_infos_temporal_*.pkl has no
        # can_bus, so every token uses the zero vector (float32 (ego_dim,)); the ego
        # status still reaches the model as prompt text.
        self.ego_dim = int(ego_dim)
        self._zero_can_bus = torch.zeros((self.ego_dim,), dtype=torch.float32)
        # Per-token prompt text (ego_msg, his_msg, cmd_msg): ego-status line,
        # historical trajectory "[(x,y),...]" (4 past points) and mission goal
        # ("turn left" / "turn right" / "keep forward"), from the cached info pickle.
        try:
            self.token_to_prompt_meta = wm.load_token_drivevla_prompt_meta_from_infos_pkl(
                self.cached_info_pkl
            )
        except Exception as e:
            rank0_print(
                f"[WARN] Failed to load prompt meta (ego/his/cmd) for DriveVLA prompts: {type(e).__name__}: {e}"
            )
            self.token_to_prompt_meta = {}
        # GT future trajectories for IK, {token -> (ik_num_waypoints, 2)}.
        try:
            self.token_to_gt_fut_traj = load_token_to_gt_fut_traj_map(
                self.cached_info_pkl, n=self.ik_num_waypoints
            )
        except Exception as e:
            rank0_print(f"[WARN] Failed to load token->GT future traj map: {type(e).__name__}: {e}")
            self.token_to_gt_fut_traj = {}

    def __len__(self) -> int:
        """Number of kept transitions (not the number of JSONL rows)."""
        return len(self.items)

    def _camera_paths(self, token: str) -> Optional[List[Path]]:
        """Absolute image paths of ``token``'s cameras (``vision.CAMERAS`` order), or
        ``None`` if the infos pickle lists no paths for it."""
        rel = self.token_to_cams.get(token)
        if rel is None:
            return None
        return [Path(self.sensor_root) / r for r in rel]

    def _has_frame(self, token: str) -> bool:
        """True if every camera image of ``token`` exists. Memoised per token."""
        hit = self._frame_available.get(token)
        if hit is None:
            paths = self._camera_paths(token)
            hit = paths is not None and all(p.is_file() for p in paths)
            self._frame_available[token] = hit
        return hit

    def _advance_next_token(self, cur_tok: str, k: int) -> Tuple[str, int]:
        """Walk the chain up to k hops, stopping at the first token without
        visual input or without a next link. Returns (target_token, actual_k).

        actual_k = 0 indicates the chain could not advance even one step
        (or k<1); in that case target_token == cur_tok and the caller is
        expected to use cur_tok's own visual state as the BEV target (autoencoder
        / identity supervision).

        Args:
            cur_tok: starting frame token.
            k: requested number of hops (``None`` -> 1).

        Returns:
            ``(target_token, actual_k)`` with ``0 <= actual_k <= k``. With 0.5 s
            frame spacing the target is ``0.5 * actual_k`` seconds ahead.
        """
        k = int(k) if k is not None else 1
        tok = str(cur_tok or "")
        # Nothing to walk: empty token or non-positive horizon.
        if not tok or k < 1:
            return (tok, 0)
        seen = set()
        actual = 0
        # Hop forward while the next frame exists and has its visual input; stop
        # early on a cycle in the next-links, a missing link or missing input.
        for _ in range(k):
            if tok in seen:
                break
            seen.add(tok)
            nxt = self.token_to_next.get(tok, "")
            if not nxt:
                break
            if not self._has_frame(nxt):
                break
            tok = nxt
            actual += 1
        return (tok, actual)

    def _load_pth(self, token: str) -> Dict[str, Any]:
        """Load the visual input of ``token`` on CPU.

        Returns:
            ``{"camera_images": float32 (3, 3, 288, 160)}``, the preprocessed front
            cameras (encoded by the model forward).

        Raises:
            FileNotFoundError: no camera paths for the token in the infos pickle.
        """
        paths = self._camera_paths(token)
        if paths is None:
            raise FileNotFoundError(f"no camera paths for token={token} in the infos pickle")
        return {"camera_images": load_camera_images(paths)}

    def _build_user_prompt(self, cur_tok: str, prev_action: str) -> str:
        """Build the v3 user prompt; scene and ego embeddings live in the system prompt.

        Args:
            cur_tok: current frame token (selects the prompt meta).
            prev_action: previous frame's trajectory text ("" if none).

        Returns:
            Newline-joined user text, in order: "Ego status: ...", "Historical
            trajectory (last 2 seconds): [...]", "Mission goal: ...", "Previous
            action: <traj_start>...<traj_end>", "Planning trajectory:
            <trajectory>".
        """
        # Prompt meta of the current frame; missing / malformed -> empty ego and
        # command text and an empty "[]" history.
        ego_msg, his_msg, cmd_msg = ("", "[]", "")
        meta = self.token_to_prompt_meta.get(cur_tok, None)
        if isinstance(meta, (tuple, list)) and len(meta) == 3:
            ego_msg, his_msg, cmd_msg = (
                str(meta[0] or ""),
                str(meta[1] or "[]"),
                str(meta[2] or ""),
            )
        # "Previous action:" line (always present). An empty previous action becomes
        # "<traj_start>[No Previous Trajectory]<traj_end>".
        extra_lines = [f"Previous action: {_format_prev_action(prev_action)}"]
        # No <EGO> token in the user turn (it is in the v3 system prompt); the
        # builder ends with the "Planning trajectory: <trajectory>" line.
        return build_drivevla_prompt_question_v3(
            ego_msg=ego_msg,
            his_msg=his_msg,
            cmd_msg=cmd_msg,
            extra_lines=extra_lines,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Build one training sample (one transition).

        Args:
            idx: index into ``self.items``.

        Returns:
            Dict consumed by :class:`DataCollatorForLatentActionWM`. Main keys
            (H = number of BEV horizons, L = token length):

            * ``input_ids`` / ``labels``: LongTensor ``(L,)``. Labels are
              ``IGNORE_INDEX`` (-100) over the prompt prefix (system + user +
              assistant header) and equal to ``input_ids`` over the assistant answer
              (BEV block(s), closing ``<|im_end|>``). The
              model uses the ``<BEV_STATE>`` label positions to locate the hidden
              states for the BEV regression and masks BEV spans out of the LM loss.
            * ``uniad_pth``: current frame visual input (``camera_images``
              (3, 3, 288, 160)).
            * ``uniad_data``: ``{"img_metas": [{"can_bus": Tensor(ego_dim + 1)}]}``,
              the ego vector for the ``<EGO>`` embedding.
            * ``next_uniad_pths``: list of H visual inputs of the horizon frames.
            * ``gt_traj``: float32 ``(ik_num_waypoints, 2)`` IK target, or ``None``.
            * ``query_box_targets`` (4, 6), ``query_has_obj`` (4,),
              ``query_box_targets_next`` (1, 4, 6), ``query_has_obj_next`` (1, 4):
              only with ``use_camera_query_targets``.
            * ``wm_*``: debug / logging metadata (tokens, task type, prompts,
              answer preview).
        """
        # Unpack the transition: horizon target tokens and the hop counts actually
        # reached, H entries each.
        row_idx, prev_tok, cur_tok, next_tok_csv, actual_ks_csv = self.items[idx]
        next_toks = next_tok_csv.split(",")
        actual_ks = [int(x) for x in actual_ks_csv.split(",")]
        row = self.rows[row_idx]
        task_type = str(row.get("task_type", "") or "").lower()
        prev_action = self.sample_id_to_traj.get(prev_tok, "") if prev_tok else ""
        # A previous trajectory whose numbers are all ~0 (|v| < 1e-3 m) is treated as
        # "no previous action" and shown as "[No Previous Trajectory]".
        if prev_action:
            _pa_nums = re.findall("-?\\d+\\.?\\d*", prev_action)
            if _pa_nums and max((abs(float(_v)) for _v in _pa_nums)) < 0.001:
                prev_action = ""
        # User turn text.
        q = self._build_user_prompt(cur_tok, prev_action)
        # Assistant answer: one "<bev_start> <BEV_STATE> x K <bev_end>" block per
        # horizon (newline separated).
        bev_blocks = []
        for _k in self.bev_horizon_steps:
            bev_blocks.append(
                f"{DEFAULT_BEV_START_TOKEN} {_repeat_bev_state_tokens(self.bev_pred_len)} {DEFAULT_BEV_END_TOKEN}".strip()
            )
        a = "\n".join(bev_blocks)
        # Conversation template (fixed to v3).
        tpl = conv_templates["qwen_bev_latent_action_world_model_v3"]
        # Full text = system + user turn + assistant turn (ChatML); prefix text = the
        # same up to and including the "<|im_start|>assistant\n" header.
        conv_full = tpl.copy()
        conv_full.messages = []
        conv_full.append_message(conv_full.roles[0], q)
        conv_full.append_message(conv_full.roles[1], a)
        full_prompt = conv_full.get_prompt()
        conv_pref = tpl.copy()
        conv_pref.messages = []
        conv_pref.append_message(conv_pref.roles[0], q)
        conv_pref.append_message(conv_pref.roles[1], None)
        pref_prompt = conv_pref.get_prompt()
        # Tokenize (placeholders such as <SCENE> / <EGO> become negative ids) and mask
        # the prefix in the labels, so only the assistant answer is supervised. This
        # assumes the prefix tokenizes to a prefix of the full token sequence.
        input_ids = tokenizer_uniad_token(full_prompt, self.tokenizer, return_tensors="pt")
        pref_ids = tokenizer_uniad_token(pref_prompt, self.tokenizer, return_tensors="pt")
        labels = input_ids.clone()
        labels[: pref_ids.numel()] = IGNORE_INDEX
        # Visual inputs: current frame and one per horizon (the horizon frames were
        # chosen among frames with visual input in __init__; loading raises if a file is
        # missing).
        cur_pth = self._load_pth(cur_tok)
        next_pths = [self._load_pth(nt) for nt in next_toks]
        # IK target: the cached-info future trajectory (ik_num_waypoints x 2; 8
        # waypoints for NAVSIM). Note: the tensor is shared with the lookup map, not
        # copied.
        gt_traj = None
        t = self.token_to_gt_fut_traj.get(cur_tok, None)
        if torch.is_tensor(t):
            gt_traj = t.to(dtype=torch.float32)
        # Ego vector for the <EGO> embedding: zeros (ego_dim,) followed by the step
        # channel k_norm = actual hops of the *first* horizon / max horizon, in
        # [0, 1]; it encodes how many hops ahead the BEV target is (e.g. NAVSIM:
        # 8 hops -> 1.0, 5 hops -> 0.625, 0 hops -> 0.0). Shape (ego_dim + 1,).
        k_norm = float(actual_ks[0]) / float(self._ego_k_denom)
        can_bus = torch.cat(
            [self._zero_can_bus, torch.tensor([k_norm], dtype=torch.float32)], dim=0
        )
        # Wrapped the way the LLaVA arch reads it: uniad_data["img_metas"][-1]["can_bus"].
        uniad_data = {"img_metas": [{"can_bus": can_bus}]}
        # Assemble the sample (see the docstring for the key meanings).
        out = {
            # Token ids / labels (L,).
            "input_ids": input_ids,
            "labels": labels,
            # Current visual state, ego vector and horizon visual states.
            "uniad_pth": cur_pth,
            "uniad_data": uniad_data,
            "next_uniad_pths": next_pths,
            # IK target (or None).
            "gt_traj": gt_traj,
        }
        # Camera-query object targets for the ObjectQueryHead: rows are (cx, cy,
        # length, width, sin_yaw, cos_yaw) for queries Q0 front / Q1 front-left /
        # Q2 front-right / Q3 nearest in all three. The NAVSIM infos have no
        # gt_boxes, so every frame gets "no object" targets (zero boxes, False
        # flags) for the current frame and for one future step.
        if self.use_camera_query_targets:
            out["query_box_targets"] = torch.zeros((4, 6), dtype=torch.float32)
            out["query_has_obj"] = torch.zeros(4, dtype=torch.bool)
            out["query_box_targets_next"] = torch.zeros((1, 4, 6), dtype=torch.float32)
            out["query_has_obj_next"] = torch.zeros((1, 4), dtype=torch.bool)
        # Debug / logging metadata (copied into wm_meta by the collator). wm_sample_id
        # also serves as the scene token for GRPO PDM scoring (_scene_token).
        out["wm_sample_id"] = cur_tok
        out["wm_prev_id"] = prev_tok
        out["wm_next_id"] = next_toks
        out["wm_task_type"] = task_type
        out["wm_conv_template"] = "qwen_bev_latent_action_world_model_v3"
        out["wm_system"] = tpl.system
        out["wm_pref_prompt"] = pref_prompt
        out["wm_question"] = q
        # Short human-readable answer preview for logs ("<BEV_STATE>*K" instead of K
        # repeated tokens).
        bev_preview = "\n".join(
            [
                f"{DEFAULT_BEV_START_TOKEN} {DEFAULT_BEV_STATE_TOKEN}*{self.bev_pred_len} {DEFAULT_BEV_END_TOKEN}"
                for _ in self.bev_horizon_steps
            ]
        )
        out["wm_answer_preview"] = bev_preview.strip()
        return out


@dataclass
class DataCollatorForLatentActionWM:
    """Collate one :class:`BEVLatentActionWorldModelDataset` sample into a batch.

    Only batch size 1 is supported: the side inputs (visual inputs, per-horizon lists,
    metadata) are passed through unbatched and only the token / label tensors and
    the query-target tensors get a leading batch dimension.

    Attributes:
        tokenizer: provides ``pad_token_id``, ``padding_side`` and (fallback)
            ``model_max_length``.
        max_length: truncation length; train.py passes the effective max length
            from :func:`_compute_effective_max_len`. ``None`` -> tokenizer
            ``model_max_length`` if it is an int, else no truncation.
    """

    tokenizer: transformers.PreTrainedTokenizer
    max_length: Optional[int] = None

    def pad_sequence(
        self, xs: List[torch.Tensor], batch_first: bool, padding_value: int
    ) -> torch.Tensor:
        """Pad 1-D tensors to a common length on the tokenizer's padding side.

        Args:
            xs: list of 1-D tensors.
            batch_first: passed to ``torch.nn.utils.rnn.pad_sequence``.
            padding_value: fill value (pad token id, or ``IGNORE_INDEX`` for labels).

        Returns:
            ``(B, L_max)`` tensor if ``batch_first``. With batch size 1 nothing is
            actually padded.
        """
        # Left padding is done by reversing each sequence, right-padding, and
        # reversing the time axis back.
        if self.tokenizer.padding_side == "left":
            xs = [torch.flip(x, [0]) for x in xs]
        out = torch.nn.utils.rnn.pad_sequence(
            xs, batch_first=batch_first, padding_value=padding_value
        )
        if self.tokenizer.padding_side == "left":
            out = torch.flip(out, [1])
        return out

    def __call__(self, instances: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        """Build the model inputs for one sample.

        Args:
            instances: a sequence with exactly one dataset item.

        Returns:
            Dict of model forward kwargs: ``input_ids`` / ``labels`` /
            ``attention_mask`` ``(1, L)``, the visual inputs and ego data, the IK
            target ``gt_traj`` ``(ik_num_waypoints, 2)`` (unbatched, may be ``None``),
            ``_scene_token`` (current token, used by GRPO for PDM scoring),
            ``wm_meta`` (debug / logging metadata read by the trainer;
            keys the model forward does not name end up in its ``**kwargs``) and,
            when present, the query targets with a leading batch dim of 1.

        Raises:
            AssertionError: more than one instance.
        """
        assert len(instances) == 1, "batch_size=1 only (matches UniAD collator assumption)"
        inst = instances[0]
        inp = inst["input_ids"]
        lab = inst["labels"]
        # Truncation length: explicit max_length, else the tokenizer limit, else the
        # sequence's own length (no truncation). Tokens beyond it are cut from the
        # end (which could cut BEV placeholders or the answer if a sequence were
        # that long).
        orig_len = int(inp.numel())
        tok_max = getattr(self.tokenizer, "model_max_length", None)
        max_len = (
            int(self.max_length)
            if self.max_length is not None
            else int(tok_max) if isinstance(tok_max, int) else orig_len
        )
        was_truncated = orig_len > max_len
        inp = inp[:max_len]
        lab = lab[:max_len]
        # Fallback pad id (mutates the tokenizer). train.py normally sets a pad token
        # before training.
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = 0
        # Add the batch dim: (1, L). Labels pad with IGNORE_INDEX.
        input_ids = self.pad_sequence(
            [inp], batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = self.pad_sequence([lab], batch_first=True, padding_value=IGNORE_INDEX)
        # Model inputs. The attention mask is "token != pad id", so any real token
        # equal to the pad id would also be masked. Side inputs are passed through
        # unbatched.
        out = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
            # Visual states and ego vector (current frame + horizon frames).
            "uniad_pth": inst.get("uniad_pth"),
            "uniad_data": inst.get("uniad_data"),
            "next_uniad_pths": inst.get("next_uniad_pths"),
            # IK target.
            "gt_traj": inst.get("gt_traj"),
            # Frame token for GRPO PDM scoring (metric-cache lookup).
            "_scene_token": str(inst.get("wm_sample_id", "") or ""),
            # Debug / logging metadata (truncation info, tokens, prompts, answers).
            "wm_meta": {
                "orig_len": orig_len,
                "max_len": max_len,
                "was_truncated": bool(was_truncated),
                "sample_id": inst.get("wm_sample_id", None),
                "prev_id": inst.get("wm_prev_id", None),
                "next_id": inst.get("wm_next_id", None),
                "task_type": inst.get("wm_task_type", None),
                "conv_template": inst.get("wm_conv_template", None),
                "system": inst.get("wm_system", None),
                "pref_prompt": inst.get("wm_pref_prompt", None),
                "question": inst.get("wm_question", None),
                "answer_preview": inst.get("wm_answer_preview", None),
            },
        }
        # Camera-query targets get a leading batch dim: (1, 4, 6) / (1, 4) for the
        # current frame, (1, 1, 4, 6) / (1, 1, 4) for the future step.
        if "query_box_targets" in inst:
            out["query_box_targets"] = inst["query_box_targets"].unsqueeze(0)
            out["query_has_obj"] = inst["query_has_obj"].unsqueeze(0)
            out["query_box_targets_next"] = inst["query_box_targets_next"].unsqueeze(0)
            out["query_has_obj_next"] = inst["query_has_obj_next"].unsqueeze(0)
        return out
