"""Prompt tokenization with multimodal placeholders.

``tokenizer_uniad_token`` turns a rendered prompt string (see ``llava/conversation.py``)
into ``input_ids`` in which each placeholder string (``<SCENE>``, ``<EGO>``) is
replaced by its negative placeholder id from ``llava.constants``. The text between
placeholders is tokenized normally. The negative ids are later swapped for feature
embeddings by ``LlavaMetaForCausalLM.prepare_inputs_labels_for_multimodal_uniad_vlm``
(``llava/model/llava_arch.py``).

Caller: ``drivevla/training/data.py`` (full prompt and label prefix of every sample).
"""

import torch


from llava.constants import (
    SCENE_TOKEN_INDEX,
    EGO_TOKEN_INDEX,
    DEFAULT_SCENE_TOKEN,
    DEFAULT_EGO_TOKEN,
)


def tokenizer_uniad_token(prompt, tokenizer, return_tensors=None):
    """process the prompt with the scene and ego placeholder tokens

    Every occurrence of a placeholder string (``<SCENE>``, ``<EGO>``) becomes exactly one
    negative id (``SCENE_TOKEN_INDEX`` / ``EGO_TOKEN_INDEX``). The text between placeholders
    is tokenized chunk by chunk with ``tokenizer(chunk).input_ids``, so BPE merges never
    cross a placeholder boundary. Matching is case-sensitive, so e.g. ``<scene_start>``
    is not mistaken for ``<SCENE>``.

    BOS handling: if tokenizing the text before the first placeholder starts with
    ``tokenizer.bos_token_id``, one BOS is emitted at the start and stripped from that
    first chunk. Later chunks are used unmodified (see the note in the loop below).
    The Qwen2 tokenizer used by DriveIK does not insert a BOS, so nothing is stripped.

    Args:
        prompt (str): the input prompt
        tokenizer: the tokenizer (a Hugging Face tokenizer whose ``__call__`` returns an
            object with ``.input_ids``; must expose ``bos_token_id``)
        return_tensors (str, optional): the type of the returned tensor. Defaults to None
            (return a Python list). Only ``"pt"`` is supported otherwise.

    Returns:
        Union[List[int], torch.Tensor]: the processed token ids. A 1-D
        ``torch.long`` tensor of shape ``(L,)`` when ``return_tensors="pt"``.

    Raises:
        ValueError: if ``return_tensors`` is neither ``None`` nor ``"pt"``.
    """
    # process all special tokens in order
    # Placeholder string -> negative placeholder id written into input_ids.
    special_tokens = {
        DEFAULT_SCENE_TOKEN: SCENE_TOKEN_INDEX,
        DEFAULT_EGO_TOKEN: EGO_TOKEN_INDEX,
    }

    # Find all special tokens and their positions in the prompt
    # Collected as (char_position, placeholder_string, placeholder_id). Each placeholder
    # type is scanned independently with str.find; the search restarts right after the
    # previous hit, so consecutive repeats are all found.
    token_positions = []
    for token, token_idx in special_tokens.items():
        start = 0
        while True:
            pos = prompt.find(token, start)
            if pos == -1:
                break
            token_positions.append((pos, token, token_idx))
            start = pos + len(token)

    # Sort positions to process tokens in order
    # (tuples sort by character position first, i.e. left-to-right order in the prompt).
    token_positions.sort()

    # Split text and build input_ids
    # last_end: character index just past the previously consumed placeholder.
    input_ids = []
    last_end = 0

    # Process BOS token if present
    # Tokenize the text before the first placeholder (or the whole prompt if there is
    # none) only to detect whether the tokenizer prepends BOS. If it does, emit BOS once
    # here and set offset=1 so the copy of BOS in that first chunk is skipped below.
    first_chunk = tokenizer(
        prompt[: token_positions[0][0]] if token_positions else prompt
    ).input_ids
    if first_chunk and first_chunk[0] == tokenizer.bos_token_id:
        input_ids.append(first_chunk[0])
        offset = 1
    else:
        offset = 0

    # Process each chunk and special token
    for pos, token, token_idx in token_positions:
        # Add text before the special token
        # (skipped when two placeholders are adjacent or the prompt starts with one).
        if pos > last_end:
            chunk_ids = tokenizer(prompt[last_end:pos]).input_ids
            input_ids.extend(chunk_ids[offset:])

        # Add the special token
        input_ids.append(token_idx)

        # Advance past the placeholder. offset is reset after the first placeholder, so
        # a BOS that a tokenizer might prepend to later chunks would be kept (upstream
        # LLaVA strips it from every chunk). This has no effect for Qwen2 (no BOS).
        last_end = pos + len(token)
        offset = 0  # Only apply offset to first chunk

    # Add remaining text after last special token
    # (when there are no placeholders at all, this tokenizes the whole prompt, with the
    # BOS offset still applied).
    if last_end < len(prompt):
        chunk_ids = tokenizer(prompt[last_end:]).input_ids
        input_ids.extend(chunk_ids[offset:])

    # Convert to tensor if needed
    # Only PyTorch tensors are supported; any other non-None value is an error.
    if return_tensors is not None:
        if return_tensors == "pt":
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f"Unsupported tensor type: {return_tensors}")

    return input_ids
