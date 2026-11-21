"""Conversation template used to build DriveIK prompts.

Only one template is kept: ``v3`` (registered as
``"qwen_bev_latent_action_world_model_v3"``). It renders a Qwen ChatML conversation
whose **system prompt** already contains the visual and ego placeholders::

    <|im_start|>system
    ... Current BEV tokens: <scene_start><SCENE><scene_end>
    ... Ego status: <ego_start><EGO><ego_end> ...<|im_end|>
    <|im_start|>user
    {question}<|im_end|>
    <|im_start|>assistant
    {answer}<|im_end|>

``drivevla/training/data.py`` builds two strings from it per sample: the full prompt
(user + assistant target) and the prefix prompt (user + an empty assistant turn).
Both are tokenized with ``llava.mm_utils.tokenizer_uniad_token``; the prefix length
decides which label positions are set to ``IGNORE_INDEX`` (only the assistant reply is
supervised). ``<SCENE>`` / ``<EGO>`` later become negative placeholder ids that
``llava/model/llava_arch.py`` replaces by projected DINOv3 camera tokens and the ego
embedding.
"""

import dataclasses
from enum import Enum, auto
from typing import Any, List, Union


class SeparatorStyle(Enum):
    """How turns are separated when rendering a prompt.

    Only ``CHATML`` exists in this release; ``Conversation.get_prompt`` implements the
    ChatML layout unconditionally and does not inspect this value.
    """

    CHATML = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history.

    Attributes:
        system: Full system block, including the leading ``"<|im_start|>system\\n"``.
            ``get_prompt`` appends ``sep`` and a newline after it; an empty string
            omits the system block.
        roles: ``(user_role, assistant_role)`` header strings, e.g.
            ``("<|im_start|>user", "<|im_start|>assistant")``.
        messages: List of ``[role, message]`` pairs. ``message`` is a string, ``None``
            / empty (renders only the role header, i.e. a generation prefix), or a
            tuple ``(text, images, _)`` (one ``"<image>"`` string is prepended per image).
        offset: Upstream LLaVA field (number of leading few-shot messages); not used
            by this module.
        sep_style: Separator style; see ``SeparatorStyle``.
        sep: String appended after the system block and after every non-empty message
            (``"<|im_end|>"`` for ChatML).
        sep2, version, tokenizer_id, tokenizer, stop_str, stop_token_ids, skip_next:
            Upstream LLaVA metadata. ``get_prompt`` does not read them.
    """

    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.CHATML
    sep: str = "###"
    sep2: str = None
    version: str = "Unknown"
    tokenizer_id: str = ""
    tokenizer: Any = None
    stop_str: Union[str, List[str]] = None
    stop_token_ids: List[int] = None
    skip_next: bool = False

    def get_prompt(self):
        """Render the conversation as a single ChatML string.

        Layout: ``system + sep + "\\n"`` (if ``system`` is non-empty), then for every
        message either ``role + "\\n" + message + sep + "\\n"`` or, when the message
        is falsy (``None`` / ``""``), just ``role + "\\n"``. The latter is used to end
        the prompt with an open assistant turn for the label prefix / generation.

        Returns:
            str: The rendered prompt. Placeholder strings such as ``<SCENE>`` are kept
            verbatim; they are converted to ids later by ``tokenizer_uniad_token``.
        """
        messages = self.messages
        # System block (skipped entirely when the system string is empty).
        ret = "" if self.system == "" else self.system + self.sep + "\n"
        for role, message in messages:
            if message:
                # Upstream multimodal message format (text, images, _): prepend one
                # "<image>" string per image. Note that "<image>" is lowercase and is
                # therefore not the "<IMAGE>" placeholder that tokenizer_uniad_token
                # recognizes. DriveIK passes plain strings, so this branch is not used.
                if type(message) is tuple:
                    message, images, _ = message
                    message = "<image>" * len(images) + message
                # Completed turn: header line, message body, end-of-turn separator.
                ret += role + "\n" + message + self.sep + "\n"
            else:
                # Empty turn: header only, leaving the turn open (generation prefix).
                ret += role + "\n"
        return ret

    def append_message(self, role, message):
        """Append one ``[role, message]`` pair to ``self.messages`` (in place).

        Args:
            role: Role header string, normally ``self.roles[0]`` or ``self.roles[1]``.
            message: Turn content (``str``), ``None`` for an open turn, or an upstream
                ``(text, images, _)`` tuple.
        """
        self.messages.append([role, message])

    def copy(self):
        """Return a new ``Conversation`` that can be filled without touching ``self``.

        The ``messages`` list and its ``[role, message]`` pairs are copied, so appending
        to the copy does not modify the template. ``roles`` and the string fields are
        shared. Only ``system``, ``roles``, ``messages``, ``offset``, ``sep_style``,
        ``sep``, ``sep2`` and ``version`` are carried over; ``tokenizer_id``,
        ``tokenizer``, ``stop_str``, ``stop_token_ids`` and ``skip_next`` fall back to
        their dataclass defaults.
        """
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for (x, y) in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            version=self.version,
        )


# The only template shipped with DriveIK. The system prompt:
#   * places the current camera tokens (<SCENE>) and the ego-status embedding (<EGO>)
#     between their wrapper tags; the user prompt (drivevla/training/prompts.py) therefore
#     does not repeat them;
#   * states the ego-centric BEV coordinate convention (x right, y forward, meters);
#   * asks for predicted visual-state blocks "<bev_start> <BEV_STATE> ... <bev_end>" and,
#     only when a QA question is given, an "<answer_start>...<answer_end>" block. The
#     trajectory itself is not generated as text; the IK diffusion head decodes it.
# The prompt text describes one next-state block; the assistant target that
# drivevla/training/data.py builds contains one such block per entry of its BEV horizon
# list (default 2,4,6 steps), joined by newlines.
# roles / sep implement Qwen ChatML ("<|im_start|>role\n ... <|im_end|>\n").
v3 = Conversation(
    system="<|im_start|>system\nYou are Open-DriveVLA, a vision-language driving world model with latent actions.\n\nYou are given:\n- Current BEV tokens: <scene_start><SCENE><scene_end>\n- Ego status: <ego_start><EGO><ego_end>\n\nYou will be given:\n- Previous action trajectory (the action executed just before the current state).\n- Optional nuScenes QA question.\n\nCoordinate convention (IMPORTANT):\n- Ego-centric 2D BEV coordinates in meters.\n- x is to the right, y is to the front (forward).\n\nYour tasks:\n1) Predict the NEXT BEV state (one step ahead).\n   IMPORTANT: To predict the next BEV state, you MUST internally infer a feasible\n   ego trajectory (current action) that will be executed next.\n   DO NOT output the trajectory. It is a latent variable handled by a separate\n   inverse-kinematics model.\n2) If the user asks a nuScenes QA question, output the QA answer.\n\nOutput format (STRICT). Output ONLY the blocks below; do NOT add extra words.\n\nBlock 1 (next BEV state placeholders):\n<bev_start> <BEV_STATE> <BEV_STATE> ... <BEV_STATE> <bev_end>\n\nBlock 2 (optional QA answer, only if a QA question is provided):\n<answer_start>SHORT_ANSWER<answer_end>\n\nIMPORTANT:\n- Between <bev_start> and <bev_end>, output ONLY the special token <BEV_STATE>\n  repeated exactly K times. (K is defined by the task/training setting.)\n- <BEV_STATE> is NOT natural language. Do NOT replace it with text.\n- If there is no QA question, do NOT output <answer_start>/<answer_end>.\n",
    roles=("<|im_start|>user", "<|im_start|>assistant"),
    version="qwen",
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.CHATML,
    sep="<|im_end|>",
)
# Registry looked up by name in drivevla/training/data.py. Callers must ``.copy()`` the
# template before appending messages, because append_message mutates in place.
conv_templates = {"qwen_bev_latent_action_world_model_v3": v3}
