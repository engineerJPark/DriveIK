"""v3 prompt builder for the BEV latent-action world model.

In the v3 conversation template, <SCENE> and <EGO> tokens are already placed
inside the **system prompt** (at the very beginning), so they must NOT appear
again in the user prompt.

This module provides ``build_drivevla_prompt_question_v3``, which is identical
to ``train_bev_world_model.build_drivevla_prompt_question`` except that it
omits the Scene-information / Ego-token lines.

(``train_bev_world_model`` is the earlier, non-v3 training script this code
was derived from; it is not part of this release.)

Where it fits in the pipeline:

* The system prompt of the ``qwen_bev_latent_action_world_model_v3``
  conversation template (``llava/conversation.py``) already contains
  ``<scene_start><SCENE><scene_end>`` and ``<ego_start><EGO><ego_end>``; the
  model replaces those placeholders with projected DINOv3 camera tokens
  ("bev_tokens") and the projected ego/can_bus vector.
* ``drivevla.training.data`` (``BEVLatentActionWorldModelDataset``) calls this
  builder with the per-sample textual ego status, history, driving command
  and the "Previous action: ..." line to produce the *user* turn of the
  conversation.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from llava.constants import DEFAULT_TRAJ_TOKEN


def build_drivevla_prompt_question_v3(
    *,
    ego_msg: str = "",
    his_msg: str = "[]",
    cmd_msg: str = "",
    extra_lines: Optional[Sequence[str]] = None,
) -> str:
    """Build an OpenDriveVLA-style *user* prompt **without** <SCENE>/<EGO> tokens.

    In v3 these tokens live in the system prompt, so the user prompt only
    contains textual ego-status, historical trajectory, mission goal, etc.

    All arguments are keyword-only.

    Args:
        ego_msg: Textual ego status, e.g. the string produced by
            ``common._format_drivevla_prompt_meta_from_info`` ("- Velocity
            (vx,vy): (...) - Heading Angular Velocity ..."). Leading/trailing
            whitespace is stripped; ``None``/empty yields a bare "Ego status:".
        his_msg: Historical trajectory string such as
            ``"[(x1,y1),(x2,y2),(x3,y3),(x4,y4)]"`` (ego frame, metres); an
            empty string is rendered as ``"[]"``.
        cmd_msg: Driving command text ("turn right" / "turn left" /
            "keep forward" or empty) for the "Mission goal" line.
        extra_lines: Optional additional lines inserted after the mission
            goal (the dataset passes its "Previous action: ..." line here).
            ``None`` items and lines that are empty after stripping are
            skipped.

    Returns:
        The user prompt: the lines joined with ``"\\n"`` and stripped. Line
        order is: ego status, history, mission goal, [extra lines...],
        planning trajectory.
    """

    # Accumulates the prompt lines in their final order.
    lines: List[str] = []

    # NOTE: Scene information line and <EGO> token are OMITTED (tokens are in
    # the system prompt). Only the textual ego-status message is kept.
    lines.append(f"Ego status: {str(ego_msg or '').strip()}".strip())

    # Historical trajectory line; an empty history is rendered as "[]".
    hm = str(his_msg or "").strip()
    if hm == "":
        hm = "[]"
    lines.append(f"Historical trajectory (last 2 seconds): {hm}")
    # The mission-goal line is always present (possibly with an empty value).
    lines.append(f"Mission goal: {str(cmd_msg or '').strip()}")

    # Caller-provided extra lines (e.g. "Previous action: ..."); skip None and
    # whitespace-only entries.
    if extra_lines is not None:
        for l in extra_lines:
            if l is None:
                continue
            s = str(l).strip()
            if s:
                lines.append(s)

    # "Planning trajectory: <trajectory>" line from the OpenDriveVLA format.
    # "<trajectory>" is plain prompt text here (it is not in
    # DRIVEVLA_SPECIAL_TOKENS and nothing substitutes it); the actual
    # trajectory is decoded by the separate IK head.
    lines.append(f"Planning trajectory: {DEFAULT_TRAJ_TOKEN}")

    # Newline-joined user prompt with outer whitespace removed.
    return "\n".join(lines).strip()
