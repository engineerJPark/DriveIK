"""Shared constants: placeholder token ids, placeholder strings and literal tag tokens.

There are three kinds of "tokens" in DriveIK prompts:

1. **Placeholder strings** such as ``<SCENE>`` and ``<EGO>`` (``DEFAULT_*_TOKEN`` below).
   They are never looked up in the vocabulary. ``llava.mm_utils.tokenizer_uniad_token``
   finds them in the prompt text and writes the matching **negative placeholder id**
   (``*_TOKEN_INDEX``) into ``input_ids`` instead.
2. **Negative placeholder ids** (``*_TOKEN_INDEX``). Because no real vocabulary id is
   negative, they can be located unambiguously inside ``input_ids``.
   ``LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal_uniad_vlm``
   (``llava/model/llava_arch.py``) replaces each of them with embedding rows: the
   projected DINOv3 camera tokens for ``<SCENE>``, the projected ego-status vector for
   ``<EGO>``, and a single all-zero row for the placeholders that DriveIK does not feed.
3. **Literal tag tokens** such as ``<scene_start>`` or ``<bev_start>`` (listed in
   ``DRIVEVLA_SPECIAL_TOKENS``). These are ordinary text in prompts / targets that are
   registered as additional special tokens so that each one maps to a single vocabulary
   id (see the comment block above ``DRIVEVLA_SPECIAL_TOKENS``).

The ``<BEV_STATE>`` token used for next-visual-state prediction is defined in
``drivevla/training`` (``DEFAULT_BEV_STATE_TOKEN``), not here; ``drivevla/training/train.py``
adds it to the tokenizer together with ``DRIVEVLA_SPECIAL_TOKENS``.
"""

# -----------------------------------------------------------------------------
# LLaVA serving constants (controller / worker heartbeat in seconds, log directory).
# Inherited from upstream LLaVA; nothing in this repository reads them.
# -----------------------------------------------------------------------------
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

LOGDIR = "."

# Model Constants
# IGNORE_INDEX: label value skipped by the cross-entropy loss (PyTorch's default
#   `ignore_index` of -100). Prompt tokens, padding and every embedding row spliced in
#   for a placeholder are labelled with it.
# *_TOKEN_INDEX: negative placeholder ids written into `input_ids` by
#   `tokenizer_uniad_token` and replaced by feature embeddings in `llava_arch.py`.
#   In DriveIK only <SCENE> (DINOv3 camera tokens) and <EGO> (ego-status / can_bus
#   vector) receive real features. <IMAGE> is dropped without inserting anything,
#   <TRACK>/<MAP>/<OBJECT> make `prepare_inputs_labels_for_multimodal_uniad_vlm` raise
#   whenever features are passed, and <GOAL_EGO> becomes one zero row.
IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = -200
SCENE_TOKEN_INDEX = -201
TRACK_TOKEN_INDEX = -202
MAP_TOKEN_INDEX = -203
OBJECT_TOKEN_INDEX = -204
EGO_TOKEN_INDEX = -205
GOAL_EGO_TOKEN_INDEX = -206

# Retired placeholder ids. Note that -206 is now used by GOAL_EGO_TOKEN_INDEX.
# `tokenizer_uniad_token` raises a ValueError if a prompt still contains "<BEVTOKEN>".
# POINT_TOKEN_INDEX = -206
# BEV_TOKEN_INDEX = -207

# Upstream LLaVA image-patch / image start-end strings. Not referenced elsewhere in this
# repository.
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

# Placeholder strings that `tokenizer_uniad_token` maps to the *_TOKEN_INDEX ids above.
# The v3 system prompt (llava/conversation.py) contains <SCENE> and <EGO>.
DEFAULT_IMAGE_TOKEN = "<IMAGE>"
DEFAULT_SCENE_TOKEN = "<SCENE>"
DEFAULT_TRACK_TOKEN = "<TRACK>"
DEFAULT_MAP_TOKEN = "<MAP>"
DEFAULT_OBJECT_TOKEN = "<OBJECT>"
DEFAULT_EGO_TOKEN = "<EGO>"
DEFAULT_GOAL_EGO_TOKEN = "<GOAL_EGO>"

# Retired placeholder strings (see the retired ids above).
# DEFAULT_POINT_TOKEN = "<POINT>"
# DEFAULT_BEV_TOKEN = "<BEVTOKEN>"

# Plain-text task markers. They are NOT in DRIVEVLA_SPECIAL_TOKENS, so the tokenizer
# splits them into ordinary BPE pieces. `drivevla/training/prompts.py` writes
# "Planning trajectory: <trajectory>" into the user prompt.
DEFAULT_TRAJ_TOKEN = "<trajectory>"
DEFAULT_OBJECT_PERCEPTION_TOKEN = "<object_perception>"
DEFAULT_DRIVABLE_ROAD_TOKEN = "<drivable_road>"

# Start/end wrappers. Every *_START / *_END pair below is registered as a special token
# through DRIVEVLA_SPECIAL_TOKENS (single vocabulary id each).
# Wrapper around the (unused in DriveIK) object-tracking placeholder.
DEFAULT_TRACK_START_TOKEN = "<track_start>"
DEFAULT_TRACK_END_TOKEN = "<track_end>"

# Wrapper around a textual trajectory "[(x, y), ...]" (e.g. the previous action).
DEFAULT_TRAJ_START_TOKEN = "<traj_start>"
DEFAULT_TRAJ_END_TOKEN = "<traj_end>"

# Wrappers for object-perception / drivable-road text blocks.
DEFAULT_OBJECT_PERCEPTION_START_TOKEN = "<object_perception_start>"
DEFAULT_OBJECT_PERCEPTION_END_TOKEN = "<object_perception_end>"

DEFAULT_DRIVABLE_ROAD_START_TOKEN = "<drivable_road_start>"
DEFAULT_DRIVABLE_ROAD_END_TOKEN = "<drivable_road_end>"

# DEFAULT_OBJ_START_TOKEN="<obj_start>"
# DEFAULT_OBJ_END_TOKEN="<obj_end>"

# Wrapper around the <SCENE> placeholder (current camera tokens) in the system prompt.
DEFAULT_SCENE_START_TOKEN = "<scene_start>"
DEFAULT_SCENE_END_TOKEN = "<scene_end>"

# Wrapper around one predicted visual-state block "<bev_start> <BEV_STATE>*K <bev_end>"
# in the assistant target (next-visual-state prediction).
DEFAULT_BEV_START_TOKEN = "<bev_start>"
DEFAULT_BEV_END_TOKEN = "<bev_end>"

# Wrapper around the (unused in DriveIK) map placeholder.
DEFAULT_MAP_START_TOKEN = "<map_start>"
DEFAULT_MAP_END_TOKEN = "<map_end>"

# Wrapper around the <EGO> placeholder (ego-status embedding) in the system prompt.
DEFAULT_EGO_START_TOKEN = "<ego_start>"
DEFAULT_EGO_END_TOKEN = "<ego_end>"

# Wrapper around the <GOAL_EGO> placeholder.
DEFAULT_GOAL_EGO_START_TOKEN = "<goal_ego_start>"
DEFAULT_GOAL_EGO_END_TOKEN = "<goal_ego_end>"

# Driving-command wrapper. Registered as special tokens but not written into any prompt by
# this repository.
DEFAULT_COMMAND_START_TOKEN = "<command_start>"
DEFAULT_COMMAND_END_TOKEN = "<command_end>"

# Question / answer wrappers. Registered as special tokens (kept so the vocabulary
# matches the released checkpoints); the NAVSIM prompts do not use them.
DEFAULT_QUESTION_START = "<question_start>"
DEFAULT_QUESTION_END = "<question_end>"

DEFAULT_ANSWER_START = "<answer_start>"
DEFAULT_ANSWER_END = "<answer_end>"

# -----------------------------------------------------------------------------
# Driving task literal tags
# -----------------------------------------------------------------------------
# These tokens appear *literally* in prompts/targets (trajectory/action tags, QA
# wrappers, BEV/map/ego wrappers, etc.).
#
# Making them "additional_special_tokens" ensures each tag is encoded as a single
# token (instead of being split into multiple BPE pieces like "<tr", "aj", ...),
# which stabilizes strict-format generation and simplifies debugging.
#
# The list is consumed by drivevla/training/train.py, which appends <BEV_STATE>,
# de-duplicates, and (only on a fresh run, not when resuming) adds every entry that is not
# already a single token as an additional special token and resizes the embedding matrix.
# The placeholder strings (<SCENE>, <EGO>, ...)
# are intentionally absent: they become negative ids, not vocabulary entries.

DRIVEVLA_SPECIAL_TOKENS = [
    # scene wrappers (the <SCENE> itself is handled via special negative indices)
    DEFAULT_SCENE_START_TOKEN,
    DEFAULT_SCENE_END_TOKEN,
    # track/map/ego wrappers
    DEFAULT_TRACK_START_TOKEN,
    DEFAULT_TRACK_END_TOKEN,
    DEFAULT_MAP_START_TOKEN,
    DEFAULT_MAP_END_TOKEN,
    DEFAULT_EGO_START_TOKEN,
    DEFAULT_EGO_END_TOKEN,
    DEFAULT_GOAL_EGO_START_TOKEN,
    DEFAULT_GOAL_EGO_END_TOKEN,
    # command / question / answer wrappers
    DEFAULT_COMMAND_START_TOKEN,
    DEFAULT_COMMAND_END_TOKEN,
    DEFAULT_QUESTION_START,
    DEFAULT_QUESTION_END,
    DEFAULT_ANSWER_START,
    DEFAULT_ANSWER_END,
    # action (perception/road/trajectory) wrappers
    DEFAULT_OBJECT_PERCEPTION_START_TOKEN,
    DEFAULT_OBJECT_PERCEPTION_END_TOKEN,
    DEFAULT_DRIVABLE_ROAD_START_TOKEN,
    DEFAULT_DRIVABLE_ROAD_END_TOKEN,
    DEFAULT_TRAJ_START_TOKEN,
    DEFAULT_TRAJ_END_TOKEN,
    # bev wrappers
    DEFAULT_BEV_START_TOKEN,
    DEFAULT_BEV_END_TOKEN,
]
