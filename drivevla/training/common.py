"""Temporal infos, prompt metadata, distributed setup and training diagnostics.

Shared helpers used across the training package; most callers import this
module as ``wm`` (``from . import common as wm``). Contents, in file order:

* Distributed / launch helpers: ``_env_int``, ``_get_dist_info``, a
  ``rank0_print`` that also works before ``torch.distributed`` is
  initialised, ``CompactProgressCallback`` (narrow tqdm bar),
  ``_resolve_resume_from_checkpoint`` ("auto"/"latest"/"last" -> newest
  ``checkpoint-*``), ``_maybe_init_deepspeed``.
* Tokenizer checks: ``_read_ckpt_vocab_size``, ``_is_single_token``,
  ``_add_special_tokens_and_resize`` (adds the DriveVLA / ``<BEV_STATE>``
  special tokens when they are not already single tokens).
* Step-0 diagnostics: ``debug_print_one_batch`` and its helpers print the
  tokenizer's special ids, label-mask statistics and shift-aligned
  teacher-forcing accuracy for one batch.
* Checkpoint tensor reader: ``_load_state_dict_keys_from_checkpoint`` reads a
  filtered subset of tensors from safetensors / ``.bin`` (single or sharded)
  checkpoints; used by ``checkpoint.py`` to restore dynamically attached
  modules.
* NAVSIM info-pkl reader used by the dataset in ``data.py``:
  ``load_token_drivevla_prompt_meta_from_infos_pkl`` builds the textual
  ``(ego status, history, mission goal)`` prompt lines per sample token from
  ``cached_navsim_info.pkl``.
* ``DebugTrainer``: a ``transformers.Trainer`` that prints periodic decoded
  prompt / GT / teacher-forced prediction previews and the step-0 batch
  diagnostic (base of ``trainer.LatentActionDebugTrainer``).
* ``BEVWorldModelMixin._mask_spans_between_tokens_inplace``: masks the
  ``<bev_start> ... <bev_end>`` span out of LM labels.

Throughout, "BEV" names (bev_tokens, ``<BEV_STATE>``, ``<bev_start>``) refer to
the DINOv3 camera-feature tokens that act as the visual state; the
names are inherited from the BEV-feature code this was derived from.
"""

from __future__ import annotations
import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch
from transformers import Trainer
from transformers.trainer_callback import ProgressCallback
from llava.constants import IGNORE_INDEX, DEFAULT_BEV_START_TOKEN, DEFAULT_BEV_END_TOKEN
# llava's rank0_print is wrapped by the rank0_print defined below.
from llava.utils import rank0_print as _llava_rank0_print
from tqdm.auto import tqdm
from transformers.trainer_utils import get_last_checkpoint


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable.

    Args:
        name: Environment variable name (e.g. ``"LOCAL_RANK"``).
        default: Value returned when the variable is unset, empty or not an
            integer.

    Returns:
        The parsed integer, or ``int(default)``.
    """
    v = os.environ.get(name, None)
    # Unset or empty -> default.
    if v is None or v == "":
        return int(default)
    # Non-integer strings also fall back to the default.
    try:
        return int(v)
    except Exception:
        return int(default)


def _get_dist_info(local_rank_arg: int = -1) -> Dict[str, int]:
    """Best-effort distributed info before torch.distributed is initialized.

    Works for both:
      - torchrun / torch.distributed.run (env: RANK/WORLD_SIZE/LOCAL_RANK)
      - torch.distributed.launch legacy (arg: --local_rank/--local-rank + env often set)

    Args:
        local_rank_arg: Value of ``--local_rank`` (-1 when not given).

    Returns:
        ``{"local_rank", "rank", "world_size"}`` as ints. ``local_rank`` is the
        CLI value if >= 0, else ``LOCAL_RANK``, else -1. ``rank`` is ``RANK``
        if set, else the local rank if >= 0, else 0. ``world_size`` is
        ``WORLD_SIZE`` (default 1, clamped to >= 1).
    """
    # Raw env values (-1 / 1 when unset).
    env_local_rank = _env_int("LOCAL_RANK", -1)
    env_rank = _env_int("RANK", -1)
    env_world = _env_int("WORLD_SIZE", 1)
    # The CLI value wins; fall back to the env var when it is unset.
    local_rank = int(local_rank_arg) if local_rank_arg is not None else -1
    if local_rank < 0 and env_local_rank >= 0:
        local_rank = env_local_rank
    # Global rank: env RANK, else the local rank (single node), else 0.
    rank = env_rank if env_rank >= 0 else local_rank if local_rank >= 0 else 0
    world_size = env_world if env_world >= 1 else 1
    return {"local_rank": int(local_rank), "rank": int(rank), "world_size": int(world_size)}


def rank0_print(*args, **kwargs):
    """Rank-0 print that works even before torch.distributed is initialized.

    llava.utils.rank0_print only checks dist.is_initialized(), so with torchrun it would
    print from *all* ranks during startup (before Trainer/Accelerate init).
    This wrapper uses env vars as a fallback.

    Args:
        *args, **kwargs: Forwarded to ``print`` (or to
            ``llava.utils.rank0_print`` once the process group exists).
    """
    # Process group up: let llava's helper decide via dist.get_rank().
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return _llava_rank0_print(*args, **kwargs)
    # Before init: rank from RANK, else LOCAL_RANK, else 0.
    try:
        r = _env_int("RANK", _env_int("LOCAL_RANK", 0))
    except Exception:
        r = 0
    if int(r) == 0:
        print(*args, **kwargs)


class CompactProgressCallback(ProgressCallback):
    """``ProgressCallback`` with a narrow (50-column) tqdm training bar.

    Only ``on_train_begin`` is overridden: it creates the bar with
    ``ncols=50`` and, when resuming, starts it at the restored
    ``global_step``. All other progress handling is inherited.
    train.py subclasses it (``RunInfoProgressCallback``).
    """

    def on_train_begin(self, args, state, control, **kwargs):
        """Create the compact progress bar and log the starting step.

        Args:
            args: ``TrainingArguments`` (unused).
            state: ``TrainerState``; ``max_steps`` sets the bar length and
                ``global_step`` > 0 (resume) sets its starting position.
            control: ``TrainerControl`` (unused).

        Side effects:
            Sets ``self.training_bar`` (local rank 0 only) and
            ``self.current_step``; prints a ``[TQDM_DEBUG]`` line on rank 0.
        """
        # Only the local main process owns a bar.
        if state.is_local_process_zero:
            self.training_bar = tqdm(total=state.max_steps, ncols=50)
            # On resume, advance the bar to the restored step.
            if getattr(state, "global_step", 0) > 0:
                self.training_bar.n = int(state.global_step)
                self.training_bar.refresh()
        # ProgressCallback.on_step_end updates the bar by
        # (global_step - current_step), so start from the restored step.
        self.current_step = int(getattr(state, "global_step", 0) or 0)
        rank0_print(
            f"[TQDM_DEBUG] train_begin global_step={state.global_step} max_steps={state.max_steps}"
        )


def _resolve_resume_from_checkpoint(
    resume_from_checkpoint: Optional[str], output_dir: str
) -> Optional[str]:
    """
    resume_from_checkpoint:
      - None: no resume
      - "auto"/"latest": pick latest checkpoint under output_dir
      - else: use as-is path

    ("last" is accepted as a synonym of "auto"/"latest", matching is
    case-insensitive, and an empty / whitespace-only string means no resume.)

    Args:
        resume_from_checkpoint: ``--resume-from-checkpoint`` value.
        output_dir: ``--output-dir``; searched for ``checkpoint-*`` by
            ``transformers.trainer_utils.get_last_checkpoint``.

    Returns:
        The checkpoint directory to resume from, or ``None``.

    Raises:
        FileNotFoundError: "auto" was requested but ``output_dir`` has no
            ``checkpoint-*``, or an explicit path does not exist.
    """
    # No resume requested.
    if resume_from_checkpoint is None:
        return None
    s = str(resume_from_checkpoint).strip()
    if s == "":
        return None
    # Keyword form: newest checkpoint-<step> under output_dir.
    if s.lower() in ("auto", "latest", "last"):
        ckpt = get_last_checkpoint(str(output_dir))
        if ckpt is None:
            raise FileNotFoundError(f"[RESUME] No checkpoint-* found under output_dir={output_dir}")
        return ckpt
    # Explicit path: must exist.
    if not os.path.exists(s):
        raise FileNotFoundError(f"[RESUME] resume_from_checkpoint not found: {s}")
    return s


def _maybe_init_deepspeed(deepspeed_config: Optional[str]) -> None:
    """Initialize HF DeepSpeed config (must be called BEFORE model.from_pretrained).

    For ZeRO-3 sharding, HF requires `HfDeepSpeedConfig(...)` to exist before
    model instantiation; otherwise the model may be materialized fully on each
    rank and OOM.

    Args:
        deepspeed_config: Path to the DeepSpeed JSON (``--deepspeed``);
            falsy = DeepSpeed not used, nothing happens.

    Side effects:
        Sets ``ENV_TYPE=deepspeed`` if unset and creates an
        ``HfDeepSpeedConfig``, which registers itself with Transformers.
        Note that Transformers only keeps a *weak* reference to that object
        and this function does not keep it alive (it is bound to the local
        ``_``), so the registration can lapse once the object is
        garbage-collected after return.

    Raises:
        FileNotFoundError: the config file does not exist.
        RuntimeError: ``HfDeepSpeedConfig`` cannot be imported.
    """
    if not deepspeed_config:
        return
    cfg_path = str(deepspeed_config)
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"--deepspeed config not found: {cfg_path}")
    os.environ.setdefault("ENV_TYPE", "deepspeed")
    # Import lazily so non-DeepSpeed runs never need deepspeed installed.
    try:
        from transformers.integrations import HfDeepSpeedConfig
    except Exception as e:
        raise RuntimeError(
            "DeepSpeed is requested but transformers.integrations.HfDeepSpeedConfig is unavailable. Please ensure transformers + deepspeed are installed."
        ) from e
    # Constructing it registers the config (weakly) with Transformers' ZeRO-3
    # support; see the note in the docstring.
    _ = HfDeepSpeedConfig(cfg_path)
    rank0_print(f"[DeepSpeed] enabled: {cfg_path}")


def _read_ckpt_vocab_size(ckpt_dir: str) -> Optional[int]:
    """Return ``vocab_size`` from ``<ckpt_dir>/config.json``.

    Used by train.py only for the ``[VOCAB]`` consistency log on resume.

    Returns:
        The integer vocab size, or ``None`` if the file / key is missing, not
        an int, or unreadable.
    """
    try:
        p = Path(ckpt_dir) / "config.json"
        if p.exists():
            cfg = json.loads(p.read_text())
            v = cfg.get("vocab_size", None)
            return int(v) if isinstance(v, int) else None
    except Exception:
        pass
    return None


def _is_single_token(tokenizer, tok: str) -> bool:
    """Whether ``tok`` encodes to exactly one token id that decodes back to ``tok``.

    The BEV span masking and ``<BEV_STATE>`` position lookup compare token
    ids, so these markers must be single tokens.

    Args:
        tokenizer: Hugging Face tokenizer.
        tok: Token string, e.g. ``"<BEV_STATE>"``.

    Returns:
        True if ``encode(tok)`` (no special tokens added) is one id and that
        id maps back to exactly ``tok``.
    """
    ids = tokenizer.encode(tok, add_special_tokens=False)
    return len(ids) == 1 and tokenizer.convert_ids_to_tokens(ids[0]) == tok


# --- Step-0 batch diagnostics ------------------------------------------------
# (Counter is used by debug_print_one_batch for the most frequent labels.)
from collections import Counter

# Marker tokens whose single-token encoding is checked (and whose ids are
# looked up) by debug_print_one_batch.
CHECK_TOKENS = [
    "<traj_start>",
    "<traj_end>",
    "<bev_start>",
    "<bev_end>",
    "<BEV_STATE>",
    "<answer_start>",
    "<answer_end>",
]


def _safe_id_to_tok(tokenizer, tid: int) -> str:
    """
    Convert token id -> token string safely.
    Handles negative multimodal placeholder ids (e.g., -200) without crashing.

    Returns:
        The token string; ``"<MM:{id}>"`` for negative ids (multimodal
        placeholders, or -1 used for masked label positions in the debug
        input proxy) and ``"<id:{id}>"`` if the tokenizer cannot map the id.
    """
    tid = int(tid)
    # Negative ids are not vocabulary entries.
    if tid < 0:
        return f"<MM:{tid}>"
    try:
        tok = tokenizer.convert_ids_to_tokens(tid)
    except Exception:
        tok = None
    if tok is None:
        return f"<id:{tid}>"
    return tok


def _rank0_only() -> bool:
    """True on global rank 0, or always when no process group is initialised."""
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_rank() == 0
    return True


def _encode_one(tokenizer, s: str):
    """Check whether ``s`` encodes to exactly one token.

    (Original note: s가 '정확히 1토큰'으로 인코딩되는지 체크.)

    Returns:
        ``(ids, toks, is_single_exact, single_id)``: the encoded ids (no
        special tokens added), their token strings, whether there is exactly
        one id whose token string equals ``s``, and that id (or ``None`` when
        there is not exactly one id).
    """
    ids = tokenizer.encode(s, add_special_tokens=False)
    toks = tokenizer.convert_ids_to_tokens(ids)
    is_single_exact = len(ids) == 1 and toks[0] == s
    single_id = ids[0] if len(ids) == 1 else None
    return (ids, toks, is_single_exact, single_id)


def _fmt_ids(tokenizer, ids_1d, max_len=120):
    """Format token ids as ``"id:token"`` pairs, at most ``max_len`` of them.

    (Original note: id -> token 나열 (길이 제한).)

    Args:
        tokenizer: HF tokenizer.
        ids_1d: 1-D tensor of token ids (negative ids allowed).
        max_len: Maximum number of ids to show.

    Returns:
        A space-separated string.
    """
    ids = [int(x) for x in ids_1d[:max_len].tolist()]
    toks = [_safe_id_to_tok(tokenizer, i) for i in ids]
    return " ".join([f"{i}:{t}" for (i, t) in zip(ids, toks)])


def _fmt_labels(tokenizer, labels_1d, max_len=120):
    """Format a label row readably, showing ignored (-100) positions as ``IGN``.

    (Original note: labels(-100 포함) 보기 좋게 출력.)

    Args:
        tokenizer: HF tokenizer.
        labels_1d: 1-D tensor of label ids (-100 = ignored).
        max_len: Maximum number of positions to show.

    Returns:
        A space-separated string of ``"IGN"`` / ``"id:token"`` items.
    """
    out = []
    for x in labels_1d[:max_len].tolist():
        if x == -100:
            out.append("IGN")
        else:
            out.append(f"{int(x)}:{_safe_id_to_tok(tokenizer, int(x))}")
    return " ".join(out)


@torch.no_grad()
def debug_print_one_batch(tokenizer, batch, logits=None, sample_idx: int = 0, max_show: int = 160):
    """
    tokenizer: HF tokenizer
    batch: dict with input_ids, attention_mask, labels (typical)
    logits: (B, T, V) optional. 있으면 shift 정렬/정확도까지 같이 확인.
        (If given, shift-aligned predictions / accuracy are checked as well.)

    One-time step-0 diagnostic called by ``DebugTrainer.compute_loss`` when
    ``--debug-one-batch`` is set. Prints, for ``batch[sample_idx]``:

    * batch shape / dtype and attention-mask sum;
    * the tokenizer's pad / eos / bos / unk ids, with a warning if pad == eos;
    * whether each of ``CHECK_TOKENS`` is a single token;
    * label-mask statistics (supervised vs. ignored, unmasked pad positions,
      eos count in supervised labels);
    * a window of input ids / labels around the first supervised position,
      the most frequent supervised label ids and the ``<BEV_STATE>`` count;
    * with ``logits``: shift-aligned argmax accuracy on all supervised labels
      and on the LM labels after masking the ``<bev_start>..<bev_end>`` span
      and ``<BEV_STATE>``, a no-shift accuracy, and how often the shifted
      predictions are one of the marker tokens (collapse check).

    Args:
        tokenizer: HF tokenizer.
        batch: Dict with ``input_ids`` (B, T) and ``labels`` (B, T) int
            tensors (-100 = ignored) and optionally ``attention_mask``. The
            caller passes the multimodal-expanded sequences when available.
        logits: Optional (B, T, V) model logits aligned with ``labels``.
        sample_idx: Row of the batch to inspect.
        max_show: Number of tokens shown in the windows.

    Returns:
        None; output goes to stdout on rank 0 only. Runs under
        ``torch.no_grad()``.

    Raises:
        AssertionError: ``batch`` lacks ``input_ids`` or ``labels``.
    """
    if not _rank0_only():
        return
    # Messages: "batch has no input_ids" / "batch has no labels".
    assert "input_ids" in batch, "batch에 input_ids가 없습니다."
    assert "labels" in batch, "batch에 labels가 없습니다."
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attn = batch.get("attention_mask", None)
    # If input_ids and labels have different lengths (e.g. text-level ids vs.
    # multimodal-expanded labels), build a proxy input from the labels, with
    # ignored positions shown as -1, so that both rows line up.
    if torch.is_tensor(input_ids) and torch.is_tensor(labels):
        if input_ids.shape != labels.shape:
            print(
                f"[WARN] input_ids.shape {tuple(input_ids.shape)} != labels.shape {tuple(labels.shape)}; using labels-proxy input_ids for debug."
            )
            input_ids = labels.clone()
            input_ids[input_ids == IGNORE_INDEX] = -1
    # Header: batch geometry and attention mask.
    B, T = input_ids.shape
    print("\n" + "=" * 120)
    print(f"[DEBUG ONE BATCH] B={B}, T={T}, device={input_ids.device}, dtype={input_ids.dtype}")
    if attn is not None:
        print(
            f"attention_mask: shape={tuple(attn.shape)}, dtype={attn.dtype}, sum={attn.sum().item()}"
        )
    # Tokenizer core ids.
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    bos_id = getattr(tokenizer, "bos_token_id", None)
    unk_id = tokenizer.unk_token_id
    print("\n[Tokenizer core ids]")
    print(f"  pad_token={tokenizer.pad_token!r} pad_token_id={pad_id}")
    print(f"  eos_token={tokenizer.eos_token!r} eos_token_id={eos_id}")
    print(f"  bos_token_id={bos_id}  unk_token={tokenizer.unk_token!r} unk_token_id={unk_id}")
    # pad == eos is risky: if padding is not masked correctly the model learns
    # to emit EOS (the Korean note says this collapse is very common).
    if pad_id is not None and eos_id is not None and (pad_id == eos_id):
        print(
            "  !!! WARNING: pad_token_id == eos_token_id (패딩 마스킹 실수 시 EOS/종료 토큰 붕괴 매우 흔함)"
        )
    # Check every marker token and remember the ids of those that are single
    # tokens (tok2id); multi-token markers usually mean the special tokens
    # were not registered with the tokenizer.
    print("\n[Special token single-id check]")
    tok2id = {}
    for s in CHECK_TOKENS:
        ids, toks, is_single_exact, single_id = _encode_one(tokenizer, s)
        print(f"  {s:14s} -> ids={ids} toks={toks}  single_exact={is_single_exact}")
        if is_single_exact:
            tok2id[s] = single_id
        else:
            print(
                f"    !!! WARNING: {s} 가 '정확히 1토큰'으로 인코딩되지 않습니다. (토크나이저 special token 등록 문제 가능)"
            )
    # --- Label-mask statistics for the chosen sample -------------------------
    # x: (T,) input ids, y: (T,) labels; nonmask marks supervised positions.
    x = input_ids[sample_idx]
    y = labels[sample_idx]
    nonmask = y != -100
    n_nonmask = int(nonmask.sum().item())
    n_mask = int((~nonmask).sum().item())
    print("\n[Labels mask stats]")
    print(f"  sample={sample_idx} nonmask(labels!=-100)={n_nonmask}  masked={n_mask}")
    # No supervised token at all -> no training signal.
    if n_nonmask == 0:
        print(
            "  !!! CRITICAL: labels가 전부 -100 입니다. => 학습 신호가 거의/완전히 없음 (마스킹 로직 문제)"
        )
    # Pad positions must be masked in the labels; otherwise the model can
    # collapse to predicting pad / eos / <bev_end>.
    if pad_id is not None:
        pad_in_input = int((x == pad_id).sum().item())
        unmasked_pad = int(((x == pad_id) & (y != -100)).sum().item())
        print(f"  pad_in_input={pad_in_input}  unmasked_pad_positions={unmasked_pad}")
        if unmasked_pad > 0:
            print(
                "  !!! CRITICAL: PAD 위치가 labels에서 -100으로 마스킹되지 않았습니다. => pad/eos/<bev_end>로 붕괴 가능"
            )
    # How many supervised labels are EOS (and their share).
    if eos_id is not None:
        eos_in_labels = int(((y == eos_id) & nonmask).sum().item())
        if eos_in_labels > 0:
            print(
                f"  eos tokens in non-masked labels = {eos_in_labels}  (비율={eos_in_labels / max(1, n_nonmask):.3f})"
            )
    # --- Token window around the first supervised position -------------------
    # Shows 16 tokens of context before it and up to max_show tokens from it.
    idx = torch.where(nonmask)[0]
    if len(idx) > 0:
        first = int(idx[0].item())
        last = int(idx[-1].item())
        a = max(0, first - 16)
        b = min(T, first + max_show)
        print("\n[Window around FIRST supervised label position]")
        print(f"  supervised span: first={first}, last={last}, window=[{a}:{b})")
        print("  input_ids window:")
        print("   ", _fmt_ids(tokenizer, x[a:b], max_len=max_show))
        print("  labels window (-100 shown as IGN):")
        print("   ", _fmt_labels(tokenizer, y[a:b], max_len=max_show))
        # The 8 most frequent supervised label ids.
        vals = y[nonmask].tolist()
        top = Counter(vals).most_common(8)
        top_str = ", ".join(
            [f"{int(tid)}:{_safe_id_to_tok(tokenizer, int(tid))}({cnt})" for (tid, cnt) in top]
        )
        print("\n  top label tokens (non-masked, top8):")
        print("   ", top_str)
        # <BEV_STATE> count; expected to be K (--bev-pred-len) times the
        # number of predicted horizons.
        if "<BEV_STATE>" in tok2id:
            bev_state_id = tok2id["<BEV_STATE>"]
            n_bev = int(((y == bev_state_id) & nonmask).sum().item())
            print(
                f"\n  <BEV_STATE> count in supervised labels = {n_bev}  (expected K e.g. 625 근처인지 확인)"
            )
    else:
        print("\n[No supervised tokens found in this sample]")
    # --- Checks that need logits ---------------------------------------------
    if logits is not None:
        # Expect (B, L, V).
        if not torch.is_tensor(logits) or logits.ndim != 3:
            print("\n[WARN] logits is not a (B,L,V) tensor; skip shift-aligned checks.")
            print("=" * 120 + "\n")
            return
        # Logits are over the multimodal-expanded sequence; text-level labels
        # cannot be aligned with them.
        log_L = int(logits.shape[1])
        lbl_L = int(labels.shape[1])
        if log_L != lbl_L:
            extra = log_L - lbl_L
            print(
                f"\n[WARN] logits seq_len != labels seq_len (logits_L={log_L}, labels_L={lbl_L}, extra={extra}). This is expected when multimodal tokens are inserted. Pass multimodal-expanded labels/input_ids (wm_mm_labels/wm_mm_input_ids) to debug_print_one_batch."
            )
            print("=" * 120 + "\n")
            return
        # Causal-LM alignment: the prediction for label position t comes from
        # the logits at position t-1, so shift the argmax right by one.
        pred = logits.argmax(-1)
        px = pred[sample_idx]
        pred_for_label = torch.full_like(y, fill_value=-100)
        pred_for_label[1:] = px[:-1]
        # Accuracy over all supervised positions except t=0 (no prediction).
        valid = (y != -100) & (torch.arange(T, device=y.device) > 0)
        n_valid = int(valid.sum().item())
        if n_valid > 0:
            acc = (pred_for_label[valid] == y[valid]).float().mean().item()
            print("\n[Shift-aligned token accuracy on supervised positions]")
            print(f"  valid_positions={n_valid}  acc={acc:.4f}")
        else:
            print(
                "\n[Shift-aligned token accuracy] valid_positions=0 (supervised가 0이거나, 전부 첫 토큰만 supervised인 이상 케이스)"
            )
        # Accuracy on the labels the LM loss actually uses: mask each
        # <bev_start>..<bev_end> span (inclusive; to the end if unclosed) and
        # every <BEV_STATE>, mirroring the model's lm_labels.
        y_lm = y.clone()
        try:
            bs = tok2id.get("<bev_start>", None)
            be = tok2id.get("<bev_end>", None)
            if bs is not None and be is not None:
                starts = (y_lm == int(bs)).nonzero(as_tuple=False).flatten()
                for s in starts.tolist():
                    rel = (y_lm[s:] == int(be)).nonzero(as_tuple=False)
                    if rel.numel() == 0:
                        e = T - 1
                    else:
                        e = s + int(rel[0].item())
                    y_lm[s : e + 1] = IGNORE_INDEX
            bev_state_id = tok2id.get("<BEV_STATE>", None)
            if bev_state_id is not None:
                y_lm[y_lm == int(bev_state_id)] = IGNORE_INDEX
            valid_lm = (y_lm != IGNORE_INDEX) & (torch.arange(T, device=y_lm.device) > 0)
            n_valid_lm = int(valid_lm.sum().item())
            if n_valid_lm > 0:
                acc_lm = (pred_for_label[valid_lm] == y_lm[valid_lm]).float().mean().item()
                print("\n[Shift-aligned token accuracy on lm_labels (after BEV span masking)]")
                print(
                    f"  valid_positions={n_valid_lm}  acc={acc_lm:.4f}  (should ~= ce_tokens_after_mask)"
                )
            else:
                print(
                    "\n[Shift-aligned token accuracy on lm_labels] valid_positions=0 (all masked?)"
                )
        except Exception as e:
            print(f"\n[WARN] Failed to compute lm_labels-masked accuracy: {type(e).__name__}: {e}")
        # Unshifted accuracy, only as a sanity reference (the Korean note:
        # even in normal training the shift-aligned accuracy is the meaningful
        # one).
        valid_noshift = y != -100
        acc_noshift = (
            (px[valid_noshift] == y[valid_noshift]).float().mean().item()
            if int(valid_noshift.sum().item()) > 0
            else float("nan")
        )
        print("\n[No-shift accuracy (for sanity check only)]")
        print(f"  acc_noshift={acc_noshift:.4f}  (정상 학습이라도 보통 shift-acc가 의미 있음)")
        # Collapse check: fraction of supervised positions predicted as each
        # marker token.
        for name in ["<bev_end>", "<bev_start>", "<traj_start>", "<traj_end>", "<BEV_STATE>"]:
            if name in tok2id:
                tid = tok2id[name]
                frac = (
                    (pred_for_label[valid] == tid).float().mean().item()
                    if n_valid > 0
                    else float("nan")
                )
                print(
                    f"  pred_for_label collapse check: {name} id={tid} frac_on_supervised={frac:.3f}"
                )
        # Shifted predictions over the same window as above.
        if len(idx) > 0:
            print("\n  pred_for_label window (shift-aligned, -100 is IGN):")
            print("   ", _fmt_labels(tokenizer, pred_for_label[a:b], max_len=max_show))
    print("=" * 120 + "\n")


# Placeholder token whose hidden states are regressed onto the future visual
# tokens (same string as data.DEFAULT_BEV_STATE_TOKEN).
DEFAULT_BEV_STATE_TOKEN = "<BEV_STATE>"


def _load_state_dict_keys_from_checkpoint(ckpt_dir: str, *, keep_fn) -> Dict[str, torch.Tensor]:
    """Read a filtered subset of tensors from a Hugging Face checkpoint directory.

    Supported layouts, tried in this order (the first one present wins):
    ``model.safetensors``, ``model.safetensors.index.json`` (sharded),
    ``pytorch_model.bin``, ``pytorch_model.bin.index.json`` (sharded). For
    sharded checkpoints only the shards that contain at least one kept key
    are opened (missing shard files are skipped). Tensors are loaded on CPU.

    Args:
        ckpt_dir: Checkpoint directory.
        keep_fn: Keyword-only predicate ``keep_fn(key) -> bool`` selecting
            the state-dict keys to return.

    Returns:
        ``{key: tensor}`` for the kept keys; ``{}`` if none of the four layouts
        exists.

    Raises:
        FileNotFoundError: ``ckpt_dir`` does not exist.
        TypeError: ``pytorch_model.bin`` does not contain a dict.
    """
    ckpt = Path(ckpt_dir)
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt}")
    # Single safetensors file: read only the selected keys lazily.
    st = ckpt / "model.safetensors"
    if st.exists():
        from safetensors import safe_open

        out = {}
        with safe_open(str(st), framework="pt", device="cpu") as f:
            for k in f.keys():
                if keep_fn(k):
                    out[k] = f.get_tensor(k)
        return out
    # Sharded safetensors: use the index's weight_map to pick shards.
    st_index = ckpt / "model.safetensors.index.json"
    if st_index.exists():
        from safetensors import safe_open

        idx = json.loads(st_index.read_text())
        weight_map = idx.get("weight_map", {})
        shards = sorted({weight_map[k] for k in weight_map.keys() if keep_fn(k)})
        out = {}
        for shard in shards:
            shard_path = ckpt / shard
            if not shard_path.exists():
                continue
            with safe_open(str(shard_path), framework="pt", device="cpu") as f:
                for k in f.keys():
                    if keep_fn(k):
                        out[k] = f.get_tensor(k)
        return out
    # Single pickle checkpoint: loads the whole state dict, then filters.
    bin1 = ckpt / "pytorch_model.bin"
    if bin1.exists():
        sd = torch.load(str(bin1), map_location="cpu")
        if not isinstance(sd, dict):
            raise TypeError(f"Unexpected state_dict type from {bin1}: {type(sd)}")
        return {k: v for (k, v) in sd.items() if keep_fn(k)}
    # Sharded pickle checkpoint: load each needed shard fully, then filter.
    bin_index = ckpt / "pytorch_model.bin.index.json"
    if bin_index.exists():
        idx = json.loads(bin_index.read_text())
        weight_map = idx.get("weight_map", {})
        shards = sorted({weight_map[k] for k in weight_map.keys() if keep_fn(k)})
        out = {}
        for shard in shards:
            shard_path = ckpt / shard
            if not shard_path.exists():
                continue
            part = torch.load(str(shard_path), map_location="cpu")
            if not isinstance(part, dict):
                continue
            for k, v in part.items():
                if keep_fn(k):
                    out[k] = v
        return out
    # No recognised weight files.
    return {}


def _format_drivevla_prompt_meta_from_info(info: Dict[str, Any]) -> Tuple[str, str, str]:
    """Build (ego_status, his_traj, mission_goal) strings from an info dict.

    This mirrors the OpenDriveVLA prompt builder. It is best-effort and returns
    safe fallbacks when any field is missing.

    Fields read from ``info``:

    * ``gt_ego_lcf_feat`` -- per the upstream builder's docstring
      ``[vx, vy, ?, ?, v_yaw (rad/s), ego_length, ego_width, v0, Kappa]``.
      vx, vy and v0 ("Heading Speed") are multiplied by 0.5 and indices 2-3
      are printed as "Can Bus", exactly as upstream does.
    * ``gt_ego_his_diff`` -- (N, 2) history displacements; the difference of
      its last two rows is printed as the acceleration (ax, ay).
    * ``gt_ego_his_trajs`` -- (>=4, >=2) past positions; the first four
      (x, y) points form the "Historical trajectory" text.
    * ``gt_ego_fut_cmd`` -- one-hot-like ``[right, left, forward]`` command.

    Args:
        info: Per-sample info dict (an entry of ``cached_navsim_info.pkl``).

    Returns:
        ``(ego_msg, his_msg, cmd_msg)``. ``ego_msg`` is ``""`` if the ego
        fields are missing or malformed; ``his_msg`` defaults to ``"[]"``;
        ``cmd_msg`` is ``"turn right"``, ``"turn left"``, ``"keep forward"``
        or ``""``.
    """
    # --- Ego status text -----------------------------------------------------
    ego_msg = ""
    try:
        lcf = info.get("gt_ego_lcf_feat", None)
        his_diff = info.get("gt_ego_his_diff", None)
        # Both fields are needed; otherwise ego_msg stays empty.
        if lcf is not None and his_diff is not None:
            lcf_t = torch.as_tensor(lcf, dtype=torch.float32).flatten()
            his_diff_t = torch.as_tensor(his_diff, dtype=torch.float32)
            # Velocity components (x0.5, as upstream) and yaw rate.
            vx = float(lcf_t[0]) * 0.5
            vy = float(lcf_t[1]) * 0.5
            v_yaw = float(lcf_t[4]) if lcf_t.numel() > 4 else 0.0
            # Acceleration = change between the last two history
            # displacements (0 if fewer than two rows).
            ax = (
                float(his_diff_t[-1, 0] - his_diff_t[-2, 0])
                if his_diff_t.ndim == 2 and his_diff_t.shape[0] >= 2
                else 0.0
            )
            ay = (
                float(his_diff_t[-1, 1] - his_diff_t[-2, 1])
                if his_diff_t.ndim == 2 and his_diff_t.shape[0] >= 2
                else 0.0
            )
            # Remaining lcf entries, each 0.0 if the vector is too short.
            cx = float(lcf_t[2]) if lcf_t.numel() > 2 else 0.0
            cy = float(lcf_t[3]) if lcf_t.numel() > 3 else 0.0
            vhead = float(lcf_t[7]) * 0.5 if lcf_t.numel() > 7 else 0.0
            steering = float(lcf_t[8]) if lcf_t.numel() > 8 else 0.0
            ego_msg = f"- Velocity (vx,vy): ({vx:.2f},{vy:.2f}) - Heading Angular Velocity (v_yaw): ({v_yaw:.2f}) - Acceleration (ax,ay): ({ax:.2f},{ay:.2f}) - Can Bus: ({cx:.2f},{cy:.2f}) - Heading Speed: ({vhead:.2f}) - Steering: ({steering:.2f})"
    except Exception:
        ego_msg = ""
    # --- Historical trajectory text: first four (x, y) points ----------------
    his_msg = "[]"
    try:
        his_trajs = info.get("gt_ego_his_trajs", None)
        if his_trajs is not None:
            ht = torch.as_tensor(his_trajs, dtype=torch.float32)
            if ht.ndim == 2 and ht.shape[0] >= 4 and (ht.shape[1] >= 2):
                pts = [(float(ht[i, 0]), float(ht[i, 1])) for i in range(4)]
                inner = ",".join([f"({x:.2f},{y:.2f})" for (x, y) in pts])
                his_msg = f"[{inner}]"
    except Exception:
        his_msg = "[]"
    # --- Mission goal from the driving command -------------------------------
    # Order of precedence: right, then left, then forward.
    cmd_msg = ""
    try:
        cmd_vec = info.get("gt_ego_fut_cmd", None)
        if cmd_vec is not None:
            cv = torch.as_tensor(cmd_vec, dtype=torch.float32).flatten()
            if cv.numel() >= 3:
                right, left, forward = (float(cv[0]), float(cv[1]), float(cv[2]))
                if right > 0:
                    cmd_msg = "turn right"
                elif left > 0:
                    cmd_msg = "turn left"
                else:
                    cmd_msg = "keep forward" if forward > 0 else ""
    except Exception:
        cmd_msg = ""
    return (ego_msg, his_msg, cmd_msg)


def _load_drivevla_prompt_meta_from_cached_info_pkl(
    cached_info_pkl: str,
) -> Dict[str, Tuple[str, str, str]]:
    """Load {sample_token -> (ego_status, his_traj, mission_goal)} from a cached info pkl.

    The cache (``cached_navsim_info.pkl``) has the schema:
        Dict[sample_token, Dict[str, Any]]
    where the inner dict contains keys such as:
        gt_ego_lcf_feat, gt_ego_his_diff, gt_ego_his_trajs, gt_ego_fut_cmd, gt_ego_fut_trajs

    (The NAVSIM recipes pass ``cached_navsim_info.pkl`` via ``--cached-info-pkl``.)

    Args:
        cached_info_pkl: Path to the cached info pkl.

    Returns:
        ``{token: (ego_msg, his_msg, cmd_msg)}``.

    Raises:
        FileNotFoundError: the file does not exist.
        ValueError: the pickle is not a ``{token: dict}`` mapping.
    """
    p = Path(cached_info_pkl)
    if not p.exists():
        raise FileNotFoundError(f"cached_info_pkl not found: {p}")
    with p.open("rb") as f:
        obj = pickle.load(f)
    # Schema: {sample_token (str): info dict}.
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported cached_info_pkl format (expected dict): {p}")
    out: Dict[str, Tuple[str, str, str]] = {}
    for tok, info in obj.items():
        # Skip non-string / empty keys and non-dict values.
        if not isinstance(tok, str) or not tok:
            continue
        if not isinstance(info, dict):
            continue
        out[tok] = _format_drivevla_prompt_meta_from_info(info)
    return out


def load_token_drivevla_prompt_meta_from_infos_pkl(
    cached_info_pkl: Optional[str],
) -> Dict[str, Tuple[str, str, str]]:
    """Load {sample_token -> (ego_status, his_traj, mission_goal)} for DriveVLA-style prompts.

    The texts come from the cached info pkl (``cached_navsim_info.pkl``,
    sample_token -> dict), which covers every NAVSIM train / val token.

    Args:
        cached_info_pkl: Cached info pkl (``--cached-info-pkl``).

    Returns:
        ``{token: (ego_msg, his_msg, cmd_msg)}``. A missing path or a load
        failure is logged as a warning, not raised, so the map may be empty.

    Side effects:
        Logs a ``[DATA]`` summary with how many entries have non-empty ego /
        history / command text.
    """
    out: Dict[str, Tuple[str, str, str]] = {}
    cached_path = None
    try:
        if cached_info_pkl and Path(str(cached_info_pkl)).exists():
            out = _load_drivevla_prompt_meta_from_cached_info_pkl(str(cached_info_pkl))
            cached_path = str(cached_info_pkl)
    except Exception as e:
        rank0_print(
            f"[WARN] Failed to load cached prompt meta from {cached_info_pkl}: {type(e).__name__}: {e}"
        )
    # Coverage statistics for the log line: entries with non-empty ego text,
    # non-empty history (not "" / "[]") and non-empty command.
    ego_nonempty = sum(
        (
            1
            for v in out.values()
            if isinstance(v, (tuple, list)) and len(v) >= 1 and (str(v[0] or "").strip() != "")
        )
    )
    his_nonempty = sum(
        (
            1
            for v in out.values()
            if isinstance(v, (tuple, list))
            and len(v) >= 2
            and (str(v[1] or "").strip() not in ("", "[]"))
        )
    )
    cmd_nonempty = sum(
        (
            1
            for v in out.values()
            if isinstance(v, (tuple, list)) and len(v) >= 3 and (str(v[2] or "").strip() != "")
        )
    )
    rank0_print(
        f"[DATA] Loaded DriveVLA prompt meta: tokens={len(out)} | ego={ego_nonempty} his={his_nonempty} cmd={cmd_nonempty} | cached={len(out)}{('@' + cached_path if cached_path else '')}"
    )
    return out


def _preview_text(text: str, max_chars: int) -> str:
    """Shorten ``text`` for logging.

    Args:
        text: Text to preview.
        max_chars: Character budget; <= 0 returns ``""``.

    Returns:
        ``text`` if it fits, else its first and last ``max_chars // 2``
        characters joined by ``"\\n...\\n"``.
    """
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    # Keep head and tail halves.
    half = max_chars // 2
    return text[:half] + "\n...\n" + text[-half:]


def _safe_decode_token_ids(tokenizer, ids: List[int]) -> Tuple[str, List[int], List[int]]:
    """
    Decode token ids safely even if ids contain negative multimodal placeholders.
    Returns (text, negative_ids, unmapped_ids).

    Negative ids are rendered as ``<MM:{id}>`` and ids the tokenizer cannot
    map as ``<id:{id}>``. If ``convert_tokens_to_string`` fails, the tokens
    are joined with spaces instead.
    """
    neg_ids = []
    bad_ids = []
    toks = []
    # Map each id to a token string, recording the problematic ones.
    for tid in ids:
        tid = int(tid)
        if tid < 0:
            neg_ids.append(tid)
            toks.append(f"<MM:{tid}>")
            continue
        tok = tokenizer.convert_ids_to_tokens(tid)
        if tok is None:
            bad_ids.append(tid)
            toks.append(f"<id:{tid}>")
        else:
            toks.append(tok)
    # Detokenise (falls back to a plain join).
    try:
        txt = tokenizer.convert_tokens_to_string(toks)
    except Exception:
        txt = " ".join(toks)
    return (txt, neg_ids, bad_ids)


class DebugTrainer(Trainer):
    """``transformers.Trainer`` with training-time text diagnostics.

    Adds, on top of the stock Trainer (all printing happens on the world
    process zero only):

    * truncation warnings when the collator reports
      ``inputs["wm_meta"]["was_truncated"]`` (first 3, then every
      ``warn_truncation_steps`` steps);
    * every ``debug_print_steps`` steps, a block with the conversation
      template, system prompt, question, GT answer preview, the
      teacher-forced argmax prediction (head and tail of the supervised
      tokens, BEV span masked) and world-model loss statistics from the
      model outputs;
    * once at step 0 with ``debug_one_batch``, the token / label diagnostic
      of :func:`debug_print_one_batch` on the multimodal-expanded sequence.

    To obtain multimodal-expanded labels / input ids (the sequence after the
    visual tokens are inserted, which is what the logits are aligned with)
    it sets ``inputs["return_mm_labels_for_debug"] = True`` on the steps that
    print, and reads ``wm_mm_labels`` / ``wm_mm_input_ids`` /
    ``wm_mm_attention_mask`` from the model outputs.

    ``trainer.LatentActionDebugTrainer`` subclasses this.
    """

    def __init__(
        self,
        *args,
        debug_print_steps: int = 0,
        debug_print_only_truncated: bool = False,
        debug_print_max_chars: int = 4000,
        warn_truncation: bool = True,
        warn_truncation_steps: int = 50,
        debug_one_batch: bool = False,
        debug_one_batch_max_show: int = 160,
        pred_head_tokens: int = 128,
        pred_tail_tokens: int = 64,
        bev_sample_tokens: int = 32,
        **kwargs,
    ):
        """Store the debug options; all other arguments go to ``Trainer``.

        Args:
            debug_print_steps: Print the preview block every N global steps
                (0 disables; ``--debug-print-steps``).
            debug_print_only_truncated: Only print on steps whose batch was
                truncated.
            debug_print_max_chars: Character budget per preview block.
            warn_truncation: Emit ``[WARN][TRUNC]`` messages.
            warn_truncation_steps: After the first 3 warnings, warn only on
                steps divisible by this.
            debug_one_batch: Run :func:`debug_print_one_batch` once at step 0.
            debug_one_batch_max_show: Token window size for that diagnostic.
            pred_head_tokens: Number of supervised tokens decoded for the
                HEAD preview.
            pred_tail_tokens: Number of supervised tokens decoded for the
                TAIL preview.
            bev_sample_tokens: Stored as ``_bev_sample_tokens``; not used by
                this class.
            *args, **kwargs: Forwarded to ``transformers.Trainer``.
        """
        super().__init__(*args, **kwargs)
        # Periodic preview options.
        self._dbg_steps = int(debug_print_steps or 0)
        self._dbg_only_trunc = bool(debug_print_only_truncated)
        self._dbg_max_chars = int(debug_print_max_chars or 4000)
        # Truncation warnings (_seen_trunc counts truncated batches seen).
        self._warn_trunc = bool(warn_truncation)
        self._warn_trunc_steps = int(warn_truncation_steps or 50)
        self._seen_trunc = 0
        # One-time step-0 diagnostic state.
        self._dbg_one_batch = bool(debug_one_batch)
        self._dbg_one_batch_max_show = int(debug_one_batch_max_show or 160)
        self._dbg_one_batch_done = False
        self._dbg_request_mm_labels = True
        # Preview sizes.
        self._pred_head_tokens = int(pred_head_tokens)
        self._pred_tail_tokens = int(pred_tail_tokens)
        self._bev_sample_tokens = int(bev_sample_tokens)

    @staticmethod
    def _to_float(x):
        """Python float from a tensor / number, or ``None`` if not convertible."""
        if x is None:
            return None
        if torch.is_tensor(x):
            return float(x.detach().float().cpu().item())
        try:
            return float(x)
        except Exception:
            return None

    @staticmethod
    def _to_int(x):
        """Python int from a tensor / number, or ``None`` if not convertible."""
        if x is None:
            return None
        if torch.is_tensor(x):
            try:
                return int(x.detach().cpu().item())
            except Exception:
                return None
        try:
            return int(x)
        except Exception:
            return None

    def _should_print(self, gs: int, was_trunc: bool) -> bool:
        """Whether global step ``gs`` is a preview step.

        True iff previews are enabled, ``gs`` is a multiple of
        ``debug_print_steps`` and (with ``debug_print_only_truncated``) the
        batch was truncated.
        """
        if self._dbg_steps <= 0:
            return False
        if gs % self._dbg_steps != 0:
            return False
        if self._dbg_only_trunc and (not was_trunc):
            return False
        return True

    def _decode_ids(self, ids: List[int]) -> str:
        """Decode ids with :func:`_safe_decode_token_ids`, warning (rank 0)
        when negative or unmapped ids were present."""
        txt, neg_ids, bad_ids = _safe_decode_token_ids(self.tokenizer, ids)
        if (len(neg_ids) > 0 or len(bad_ids) > 0) and self.is_world_process_zero():
            rank0_print(
                f"[WARN][DECODE] contains neg_ids={sorted(set(neg_ids))[:8]} bad_ids={sorted(set(bad_ids))[:8]}"
            )
        return txt

    def _maybe_print_debug(
        self, inputs: Dict[str, Any], outputs: Any, loss: Optional[torch.Tensor]
    ) -> None:
        """Emit the truncation warning and, on preview steps, the preview block.

        Args:
            inputs: The batch dict; ``inputs["wm_meta"]`` (from the collator)
                provides sample ids, task type, prompt text and truncation
                info, ``inputs["labels"]`` the (B, L) text-level labels.
            outputs: Model outputs (dict-like); optionally ``logits``
                (B, L_mm, V), ``wm_mm_labels`` (B, L_mm) and world-model
                scalars (``lm_loss``, ``bev_loss``, ``wm_bev_*``).
            loss: The training loss of this step.

        The model-output preview decodes the shift-aligned argmax predictions
        of sample 0 at its first ``pred_head_tokens`` and last
        ``pred_tail_tokens`` supervised positions (after masking the BEV span
        and ``<BEV_STATE>``), next to the corresponding GT tokens.
        """
        if not self.is_world_process_zero():
            return
        # Step / batch metadata from the collator.
        gs = int(getattr(self.state, "global_step", 0))
        meta = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
        was_trunc = bool(meta.get("was_truncated", False))
        orig_len = meta.get("orig_len", None)
        max_len = meta.get("max_len", None)
        # Throttled truncation warning: always for the first 3 truncated
        # batches, then only on multiples of warn_truncation_steps.
        if self._warn_trunc and was_trunc:
            should_print = self._seen_trunc < 3 or (
                self._warn_trunc_steps > 0 and gs % self._warn_trunc_steps == 0
            )
            if should_print:
                rank0_print(
                    f"[WARN][TRUNC] step={gs} input_ids truncated: orig_len={orig_len}, max_len={max_len} (tokenizer/model max length may be too small for bev_pred_len)"
                )
            self._seen_trunc += 1
        # Everything below only on preview steps.
        if not self._should_print(gs, was_trunc):
            return
        # Prompt / sample fields for the header and text blocks.
        sid = meta.get("sample_id", None)
        nid = meta.get("next_id", None)
        tt = meta.get("task_type", None)
        oo = meta.get("output_order", None)
        tmpl = meta.get("conv_template", None)
        sys_prompt = meta.get("system", "")
        q = meta.get("question", "")
        gt_preview = meta.get("answer_preview", "")
        # World-model statistics reported by the model forward (if any).
        lm_loss = None
        bev_loss = None
        wm_bev_k = None
        wm_tgt_len = None
        wm_bev_pred_norm = None
        wm_bev_tgt_norm = None
        wm_bev_cos = None
        try:
            if hasattr(outputs, "get"):
                lm_loss = outputs.get("lm_loss", None)
                bev_loss = outputs.get("bev_loss", None)
                wm_bev_k = outputs.get("wm_bev_k", None)
                wm_tgt_len = outputs.get("wm_tgt_len", None)
                wm_bev_pred_norm = outputs.get("wm_bev_pred_norm", None)
                wm_bev_tgt_norm = outputs.get("wm_bev_tgt_norm", None)
                wm_bev_cos = outputs.get("wm_bev_cos", None)
        except Exception:
            pass
        # Convert to plain Python numbers for printing.
        loss_f = self._to_float(loss)
        lm_loss_f = self._to_float(lm_loss)
        bev_loss_f = self._to_float(bev_loss)
        wm_bev_k_i = self._to_int(wm_bev_k)
        wm_tgt_len_i = self._to_int(wm_tgt_len)
        wm_bev_cos_f = self._to_float(wm_bev_cos)
        wm_bev_pred_norm_f = self._to_float(wm_bev_pred_norm)
        wm_bev_tgt_norm_f = self._to_float(wm_bev_tgt_norm)
        pred_head_txt = ""
        pred_tail_txt = ""
        gt_head_txt = ""
        gt_tail_txt = ""
        # Logits and the labels to compare them with. Prefer the
        # multimodal-expanded labels when their length matches the logits.
        logits = (
            outputs.get("logits", None)
            if hasattr(outputs, "get")
            else getattr(outputs, "logits", None)
        )
        labels = inputs.get("labels", None)
        mm_labels = None
        if hasattr(outputs, "get"):
            mm_labels = outputs.get("wm_mm_labels", None)
        if (
            torch.is_tensor(logits)
            and torch.is_tensor(mm_labels)
            and (mm_labels.ndim == 2)
            and (mm_labels.shape[1] == logits.shape[1])
        ):
            labels = mm_labels
        # LM-view labels: mask the <bev_start>..<bev_end> span and every
        # <BEV_STATE> (each step best-effort).
        lm_labels_preview = None
        if torch.is_tensor(labels):
            lm_labels_preview = labels.clone()
            try:
                bs_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_BEV_START_TOKEN))
                be_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_BEV_END_TOKEN))
                BEVWorldModelMixin._mask_spans_between_tokens_inplace(
                    lm_labels_preview, start_id=bs_id, end_id=be_id, ignore_index=IGNORE_INDEX
                )
            except Exception:
                pass
            try:
                bev_state_id = int(self.tokenizer.convert_tokens_to_ids(DEFAULT_BEV_STATE_TOKEN))
                lm_labels_preview[lm_labels_preview == bev_state_id] = IGNORE_INDEX
            except Exception:
                pass
        if torch.is_tensor(lm_labels_preview) and torch.is_tensor(logits):
            # Causal shift: logits[:, t] predict labels[:, t + 1].
            aligned_logits = logits[:, :-1, :]
            aligned_labels = lm_labels_preview[:, 1:]
            if aligned_logits.shape[1] != aligned_labels.shape[1]:
                pred_head_txt = "(skip preview: logits/labels length mismatch; enable return_mm_labels_for_debug)"
                pred_tail_txt = ""
                gt_head_txt = ""
                gt_tail_txt = ""
            else:
                sup_mask = aligned_labels.ne(IGNORE_INDEX)
            # NOTE: the lines below are outside the else-branch, so in the
            # length-mismatch case ``sup_mask`` is unbound and the resulting
            # error is caught (and reported as a preview failure) by
            # compute_loss.
            # Supervised positions of sample 0.
            pos = torch.nonzero(sup_mask[0], as_tuple=False).squeeze(1)
            if pos.numel() > 0:
                # HEAD: the first pred_head_tokens supervised positions.
                h = min(self._pred_head_tokens, int(pos.numel()))
                head_pos = pos[:h]
                gt_head = aligned_labels[0, head_pos].detach().cpu().tolist()
                pred_head = aligned_logits[0, head_pos].argmax(dim=-1).detach().cpu().tolist()
                gt_head_txt = _preview_text(
                    self._decode_ids(gt_head), max_chars=min(self._dbg_max_chars, 2500)
                )
                pred_head_txt = _preview_text(
                    self._decode_ids(pred_head), max_chars=min(self._dbg_max_chars, 2500)
                )
                # TAIL: the last pred_tail_tokens supervised positions; left
                # empty when it would overlap the HEAD window.
                t = min(self._pred_tail_tokens, int(pos.numel()))
                tail_pos = pos[-t:]
                if (
                    head_pos.numel() > 0
                    and tail_pos.numel() > 0
                    and (int(tail_pos[0]) <= int(head_pos[-1]))
                ):
                    gt_tail_txt = ""
                    pred_tail_txt = ""
                else:
                    gt_tail = aligned_labels[0, tail_pos].detach().cpu().tolist()
                    pred_tail = aligned_logits[0, tail_pos].argmax(dim=-1).detach().cpu().tolist()
                    gt_tail_txt = _preview_text(
                        self._decode_ids(gt_tail), max_chars=min(self._dbg_max_chars, 1800)
                    )
                    pred_tail_txt = _preview_text(
                        self._decode_ids(pred_tail), max_chars=min(self._dbg_max_chars, 1800)
                    )
        else:
            pred_head_txt = "(no logits/labels available to compute model output preview)"
        # Shortened prompt / GT text blocks.
        sys_prev = _preview_text(str(sys_prompt or ""), max_chars=min(self._dbg_max_chars, 2000))
        q_prev = _preview_text(str(q or ""), max_chars=min(self._dbg_max_chars, 2000))
        gt_prev = _preview_text(str(gt_preview or ""), max_chars=self._dbg_max_chars)
        # Assemble and print the block (empty TAIL / GT sections are omitted).
        header = f"[DEBUG] step={gs} (tokens={(inputs.get('input_ids', torch.empty(1)).shape[-1] if torch.is_tensor(inputs.get('input_ids', None)) else 'NA')}, truncated={was_trunc}, orig_len={orig_len}, max_len={max_len})\n        sample_id={sid} next_id={nid} task_type={tt} output_order={oo}\n        loss={loss_f} lm_loss={lm_loss_f} bev_loss={bev_loss_f}\n        wm_bev_k={wm_bev_k_i} wm_tgt_len={wm_tgt_len_i} wm_bev_cos={wm_bev_cos_f} wm_pred_norm={wm_bev_pred_norm_f} wm_tgt_norm={wm_bev_tgt_norm_f}"
        rank0_print(
            "\n"
            + "=" * 90
            + "\n"
            + header
            + "\n"
            + "-" * 90
            + "\n[CONV_TEMPLATE]\n"
            + str(tmpl or "")
            + "\n\n[SYSTEM]\n"
            + sys_prev
            + "\n[QUESTION]\n"
            + q_prev
            + "\n\n[GT_ANSWER_PREVIEW]\n"
            + gt_prev
            + "\n\n[MODEL_OUTPUT (teacher-forcing argmax) | HEAD]\n"
            + pred_head_txt
            + (
                "\n\n[MODEL_OUTPUT (teacher-forcing argmax) | TAIL]\n" + pred_tail_txt
                if pred_tail_txt
                else ""
            )
            + ("\n\n[GT_TOKENS | HEAD]\n" + gt_head_txt if gt_head_txt else "")
            + ("\n\n[GT_TOKENS | TAIL]\n" + gt_tail_txt if gt_tail_txt else "")
            + "\n"
            + "=" * 90
            + "\n"
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        transformers 최신 버전은 training_step -> compute_loss(model, inputs, num_items_in_batch=...) 로 호출함.
        따라서 num_items_in_batch/kwargs를 둘 다 받아 호환되게 처리한다.

        (Recent Transformers versions call ``compute_loss(model, inputs,
        num_items_in_batch=...)`` from ``training_step``, so both
        ``num_items_in_batch`` and ``**kwargs`` are accepted and forwarded
        when supported.)

        Flow: (1) on steps that will print, request multimodal-expanded
        labels from the model via ``inputs["return_mm_labels_for_debug"]``;
        (2) compute loss and outputs with the stock ``Trainer.compute_loss``;
        (3) run the one-time step-0 diagnostic and the periodic preview.
        Diagnostic failures are logged and never interrupt training.

        Args:
            model: The (possibly wrapped) model.
            inputs: Batch dict from the collator (modified in place in step 1).
            return_outputs: Return ``(loss, outputs)`` instead of ``loss``.
            num_items_in_batch: Forwarded to ``Trainer.compute_loss`` if given.
            **kwargs: Forwarded to ``Trainer.compute_loss``.

        Returns:
            ``loss`` or ``(loss, outputs)``.
        """
        # Current global step (0 if the state is not available).
        try:
            gs0 = int(getattr(self.state, "global_step", 0))
        except Exception:
            gs0 = 0
        # (1) Decide whether this step needs multimodal-expanded labels: the
        # pending one-batch diagnostic at step 0, or a periodic preview step.
        if isinstance(inputs, dict):
            request_mm = False
            if (
                self._dbg_one_batch
                and (not self._dbg_one_batch_done)
                and self._dbg_request_mm_labels
                and (gs0 == 0)
            ):
                request_mm = True
            else:
                try:
                    meta = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
                    was_trunc = bool(meta.get("was_truncated", False))
                    gs = int(getattr(self.state, "global_step", 0))
                    if self._dbg_steps > 0 and self._should_print(gs, was_trunc):
                        request_mm = True
                except Exception:
                    pass
            if request_mm:
                inputs["return_mm_labels_for_debug"] = True
        # (2) Stock loss computation; always ask for outputs (needed below).
        # Older Transformers do not accept num_items_in_batch -> TypeError ->
        # retry without the extra keywords.
        try:
            kw = dict(kwargs)
            if num_items_in_batch is not None:
                kw["num_items_in_batch"] = num_items_in_batch
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True, **kw)
        except TypeError:
            loss, outputs = super().compute_loss(model, inputs, return_outputs=True)
        # (3) Diagnostics (rank 0 only, best-effort).
        try:
            if (
                self._dbg_one_batch
                and (not self._dbg_one_batch_done)
                and self.is_world_process_zero()
            ):
                gs0 = int(getattr(self.state, "global_step", 0))
                if gs0 == 0:
                    # One-line summary of the sample being inspected.
                    meta = inputs.get("wm_meta", {}) if isinstance(inputs, dict) else {}
                    try:
                        rank0_print(
                            f"[ONEBATCH_DIAG] step={gs0} sample_id={meta.get('sample_id', None)} next_id={meta.get('next_id', None)} task_type={meta.get('task_type', None)} orig_len={meta.get('orig_len', None)} max_len={meta.get('max_len', None)} truncated={meta.get('was_truncated', None)}"
                        )
                    except Exception:
                        pass
                    # Multimodal-expanded tensors returned by the model.
                    mm_input_ids = None
                    mm_labels = None
                    mm_attn = None
                    if hasattr(outputs, "get"):
                        mm_input_ids = outputs.get("wm_mm_input_ids", None)
                        mm_labels = outputs.get("wm_mm_labels", None)
                        mm_attn = outputs.get("wm_mm_attention_mask", None)
                    # Build the batch for debug_print_one_batch: labels =
                    # multimodal labels if available, else the text labels.
                    batch_dbg = dict(inputs) if isinstance(inputs, dict) else {}
                    dbg_labels = (
                        mm_labels if torch.is_tensor(mm_labels) else inputs.get("labels", None)
                    )
                    if torch.is_tensor(dbg_labels):
                        batch_dbg["labels"] = dbg_labels
                    # input_ids aligned with those labels: the multimodal ids
                    # if shapes match, else a proxy built from the labels
                    # (-1 at ignored positions), else the original ids.
                    dbg_input_ids = None
                    if (
                        torch.is_tensor(mm_input_ids)
                        and torch.is_tensor(dbg_labels)
                        and (mm_input_ids.shape == dbg_labels.shape)
                    ):
                        dbg_input_ids = mm_input_ids
                    elif torch.is_tensor(dbg_labels):
                        dbg_input_ids = dbg_labels.clone()
                        dbg_input_ids[dbg_input_ids == IGNORE_INDEX] = -1
                    else:
                        dbg_input_ids = inputs.get("input_ids", None)
                    if torch.is_tensor(dbg_input_ids):
                        batch_dbg["input_ids"] = dbg_input_ids
                    # Matching attention mask, or all-ones as a fallback.
                    if (
                        torch.is_tensor(mm_attn)
                        and torch.is_tensor(dbg_input_ids)
                        and (mm_attn.shape == dbg_input_ids.shape)
                    ):
                        batch_dbg["attention_mask"] = mm_attn
                    elif torch.is_tensor(dbg_input_ids):
                        batch_dbg["attention_mask"] = torch.ones_like(
                            dbg_input_ids, dtype=torch.bool
                        )
                    # CE token count reported by the model (before BEV-span
                    # masking) and text / multimodal lengths.
                    if hasattr(outputs, "get"):
                        ce_raw = outputs.get("wm_ce_tokens_raw", None)
                        bev_span_ok = outputs.get("wm_bev_span_mask_applied", None)
                        mmL = outputs.get("wm_mm_seq_len", None)
                        txtL = outputs.get("wm_text_seq_len", None)
                        try:
                            rank0_print(
                                f"[ONEBATCH_CE] raw_ce_tokens={ce_raw} bev_span_mask_applied={bev_span_ok} | text_L={txtL} mm_L={mmL}"
                            )
                        except Exception:
                            pass
                    debug_print_one_batch(
                        self.tokenizer,
                        batch_dbg,
                        logits=getattr(outputs, "logits", None),
                        sample_idx=0,
                        max_show=self._dbg_one_batch_max_show,
                    )
                    self._dbg_one_batch_done = True
            # Truncation warning + periodic preview.
            self._maybe_print_debug(inputs, outputs, loss)
        except Exception as e:
            if self.is_world_process_zero():
                rank0_print(
                    f"[WARN][DEBUG] Failed to print model output preview: {type(e).__name__}: {e}"
                )
        return (loss, outputs) if return_outputs else loss


class BEVWorldModelMixin:
    """Namespace for label-masking helpers of the world model.

    In this module only the static method below is used (by
    ``DebugTrainer._maybe_print_debug``).
    """

    @staticmethod
    def _mask_spans_between_tokens_inplace(
        labels: torch.Tensor, start_id: int, end_id: int, ignore_index: int = IGNORE_INDEX
    ) -> None:
        """In-place: mask labels from each <start_id> to the next <end_id> (inclusive).

        This is used to exclude the entire BEV block (<bev_start> ... <bev_end>) from LM CE loss.
        NOTE: labels is expected to be (B, L).

        An unclosed span (no ``end_id`` after a ``start_id``) is masked to the
        end of the row.

        Args:
            labels: (B, L) integer label tensor, modified in place; ``None``
                is a no-op.
            start_id: Token id opening a span (e.g. ``<bev_start>``).
            end_id: Token id closing a span (e.g. ``<bev_end>``).
            ignore_index: Value written into masked positions (-100).

        Raises:
            ValueError: ``labels`` is not 2-D.
        """
        if labels is None:
            return
        if labels.ndim != 2:
            raise ValueError(f"labels must be (B,L), got shape={tuple(labels.shape)}")
        start_id = int(start_id)
        end_id = int(end_id)
        B, L = labels.shape
        for b in range(B):
            # Every occurrence of the start token in this row.
            starts = (labels[b] == start_id).nonzero(as_tuple=False).flatten()
            if starts.numel() == 0:
                continue
            for s in starts.tolist():
                # First end token at or after s (else the row end).
                rel = (labels[b, s:] == end_id).nonzero(as_tuple=False)
                if rel.numel() == 0:
                    e = L - 1
                else:
                    e = s + int(rel[0].item())
                labels[b, s : e + 1] = ignore_index


def _add_special_tokens_and_resize(
    tokenizer, model, extra_tokens: List[str], *, allow_add: bool = True
) -> List[str]:
    """Register missing special tokens and resize the embedding matrix.

    train.py calls this for fresh (non-resumed) runs with
    ``DRIVEVLA_SPECIAL_TOKENS + ["<BEV_STATE>"]``.

    Args:
        tokenizer: HF tokenizer (modified in place).
        model: Model whose token embeddings are resized to ``len(tokenizer)``.
        extra_tokens: Tokens that must be single tokens; duplicates are
            removed preserving order.
        allow_add: If False, only report which tokens would be added.

    Returns:
        The tokens that were not single tokens (added when ``allow_add``);
        ``[]`` if every token already was one (then nothing is changed).

    Side effects:
        ``tokenizer.add_special_tokens({"additional_special_tokens": ...})``
        and ``model.resize_token_embeddings(len(tokenizer))``.
    """
    # Order-preserving de-duplication.
    extra_tokens = list(dict.fromkeys(list(extra_tokens)))
    # Only tokens that are not already single tokens need adding.
    to_add = [t for t in extra_tokens if not _is_single_token(tokenizer, t)]
    if not to_add:
        return []
    if not allow_add:
        return to_add
    tokenizer.add_special_tokens({"additional_special_tokens": to_add})
    model.resize_token_embeddings(len(tokenizer))
    return to_add
