"""Compatibility with the installed PyTorch, Transformers and Accelerate versions.

This module has no public API; it is imported only for its side effects
(``from . import compat`` in ``drivevla/training/train.py``, before any
Trainer / Accelerator is constructed). Each block below monkey-patches one
library symbol, and only when the installed version actually needs it, so the
same code runs on both older and newer library combinations. Every shim except
the last one is wrapped in ``try/except`` so a failed patch never prevents
training from starting.

Shims, in order:

1. ``torch.distributed.elastic.agent.server.api._get_socket_with_port`` --
   re-added for DeepSpeed 0.14.x on PyTorch >= 2.5 (e.g. DeepSpeed 0.14.2 with
   torch 2.7 in the reference environment).
2. ``accelerate.Accelerator.unwrap_model`` -- accept (and ignore) the
   ``keep_torch_compile`` keyword that newer Transformers pass to older
   Accelerate releases.
3. ``transformers.Trainer.__init__`` -- accept the old ``tokenizer=`` keyword on
   Transformers versions that only know ``processing_class=``.
4. ``transformers.trainer_callback.TrainerState.load_from_json`` -- drop
   ``trainer_state.json`` fields that the installed ``TrainerState`` does not
   know (checkpoints written by a newer Transformers).
5. ``torch.load`` -- default ``weights_only`` back to ``False`` (PyTorch >= 2.6
   defaults it to ``True``), process-wide.
"""

from __future__ import annotations
import json
import os
import socket

# DeepSpeed < 0.15 imports a helper that PyTorch >= 2.5 removed. Accelerate imports
# DeepSpeed whenever it is installed, so restore the helper before that happens.
#
# Concretely, DeepSpeed 0.14.2's ``deepspeed/elasticity/elastic_agent.py`` does
# ``from torch.distributed.elastic.agent.server.api import _get_socket_with_port``
# at import time, which raises ImportError on torch >= 2.5. If the attribute is
# missing, a re-implementation is attached to that torch module so the import
# succeeds whenever DeepSpeed is imported later. Nothing is patched when the
# installed torch still provides the helper.
try:
    import torch.distributed.elastic.agent.server.api as _elastic_api

    if not hasattr(_elastic_api, "_get_socket_with_port"):

        def _get_socket_with_port() -> socket.socket:
            """Return a listening TCP socket bound to a free port on localhost.

            Re-implementation of the helper removed from PyTorch: binding to
            port 0 lets the OS choose an unused port; the caller reads the
            port from ``sock.getsockname()`` and is responsible for closing the
            socket.

            Returns:
                A ``socket.socket`` that is bound and listening.

            Raises:
                RuntimeError: if no address family returned by
                    ``getaddrinfo("localhost")`` could be bound.
            """
            # Try each address family/socket type that "localhost" resolves to
            # (typically IPv6 and IPv4) until one can be bound.
            for family, type_, proto, _, _ in socket.getaddrinfo(
                host="localhost",
                port=None,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                flags=socket.AI_PASSIVE,
            ):
                s = socket.socket(family, type_, proto)
                # Bind to port 0 (OS picks a free port) and start listening;
                # on failure close this socket and try the next family.
                try:
                    s.bind(("localhost", 0))
                    s.listen(0)
                    return s
                except OSError:
                    s.close()
            # No address family could be bound.
            raise RuntimeError("Failed to create a socket")

        # Install the helper on the torch module DeepSpeed imports it from.
        _elastic_api._get_socket_with_port = _get_socket_with_port
except Exception:
    # torch built without distributed/elastic support (or any other failure):
    # leave everything as is.
    pass

# Transformers is imported here (after the DeepSpeed shim) because the shims
# below patch ``Trainer`` and read ``transformers.__version__``.
import transformers
from transformers import Trainer

# Accelerate shim: newer Transformers call
# ``accelerator.unwrap_model(model, keep_torch_compile=...)``, but older
# Accelerate releases (e.g. 0.29.x) define ``unwrap_model(self, model,
# keep_fp32_wrapper=True)`` and would raise TypeError on the unknown keyword.
# If the installed signature lacks ``keep_torch_compile``, wrap the original
# method so it accepts and silently drops any extra keyword arguments.
try:
    from accelerate import Accelerator as _Accelerator

    if "keep_torch_compile" not in str(__import__("inspect").signature(_Accelerator.unwrap_model)):
        _orig_unwrap_model = _Accelerator.unwrap_model

        def _patched_unwrap_model(
            self, model, keep_fp32_wrapper: bool = True, _orig=_orig_unwrap_model, **_kwargs
        ):
            """``Accelerator.unwrap_model`` that ignores unsupported keywords.

            Args:
                model: The (possibly DDP/DeepSpeed/compiled-wrapped) model.
                keep_fp32_wrapper: Forwarded unchanged to the original method.
                _orig: The original unbound ``unwrap_model`` (bound as a
                    default argument so the closure does not depend on the
                    module-level name).
                **_kwargs: Extra keywords such as ``keep_torch_compile``;
                    discarded.

            Returns:
                Whatever the original ``unwrap_model`` returns (the inner model).
            """
            return _orig(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

        _Accelerator.unwrap_model = _patched_unwrap_model
except Exception:
    # Accelerate missing or signature not inspectable: no patch.
    pass
# Trainer shim: Transformers renamed the ``Trainer(tokenizer=...)`` argument to
# ``processing_class`` (and later versions no longer accept ``tokenizer``). The
# training code passes ``tokenizer=``; when the installed ``Trainer.__init__``
# has ``processing_class`` but no ``tokenizer`` parameter, wrap it so that
# ``tokenizer=`` is forwarded as ``processing_class=``. Older Transformers
# (e.g. 4.44.x, which still has ``tokenizer``) are left untouched.
try:
    import inspect as _inspect_trainer_kw

    _trainer_init_params = _inspect_trainer_kw.signature(Trainer.__init__).parameters
    if "tokenizer" not in _trainer_init_params and "processing_class" in _trainer_init_params:
        _orig_trainer_init = Trainer.__init__

        def _patched_trainer_init(
            self, *_args, tokenizer=None, _orig=_orig_trainer_init, **_kwargs
        ):
            """``Trainer.__init__`` that maps ``tokenizer=`` to ``processing_class=``.

            Args:
                *_args: Positional arguments forwarded unchanged.
                tokenizer: Legacy keyword. If given and ``processing_class`` is
                    not passed explicitly, it is used as ``processing_class``;
                    otherwise it is dropped.
                _orig: The original ``Trainer.__init__`` (bound as a default).
                **_kwargs: Remaining keyword arguments, forwarded unchanged.
            """
            # An explicit processing_class always wins over tokenizer.
            if tokenizer is not None and "processing_class" not in _kwargs:
                _kwargs["processing_class"] = tokenizer
            return _orig(self, *_args, **_kwargs)

        Trainer.__init__ = _patched_trainer_init
except Exception:
    pass
# TrainerState shim: ``Trainer`` restores ``trainer_state.json`` with
# ``TrainerState.load_from_json``, which calls ``cls(**json_data)``.
# ``TrainerState`` is a dataclass whose ``__init__`` accepts only its declared
# fields, so a checkpoint written by a newer Transformers (with extra fields)
# would raise TypeError on resume. When ``__init__`` has no ``**kwargs``,
# replace ``load_from_json`` with a version that drops unknown keys.
try:
    import inspect as _inspect_ts
    from transformers.trainer_callback import TrainerState as _TrainerState

    # Field names accepted by the installed TrainerState.__init__.
    _ts_allowed = set(_inspect_ts.signature(_TrainerState.__init__).parameters.keys()) - {"self"}
    # Whether __init__ already takes **kwargs (then nothing needs filtering).
    _ts_has_var_kw = any(
        (
            p.kind == _inspect_ts.Parameter.VAR_KEYWORD
            for p in _inspect_ts.signature(_TrainerState.__init__).parameters.values()
        )
    )
    if not _ts_has_var_kw:

        @classmethod
        def _ts_load_from_json_compat(cls, json_path):
            """Build a ``TrainerState`` from JSON, ignoring unknown fields.

            Args:
                json_path: Path to a ``trainer_state.json`` file.

            Returns:
                A ``TrainerState`` built from the keys the installed class
                accepts.

            Side effects:
                If any fields are dropped, prints their names on local rank 0
                (rank read from the ``LOCAL_RANK`` env var, default 0).
            """
            # Read the saved state.
            with open(json_path, "r", encoding="utf-8") as _f:
                _data = json.loads(_f.read())
            # Report the fields that will be dropped (rank 0 only).
            _unknown = [k for k in _data.keys() if k not in _ts_allowed]
            if _unknown and str(os.environ.get("LOCAL_RANK", "0")) == "0":
                print(
                    f"[RESUME] trainer_state.json: dropping unknown fields {_unknown} (saved by newer transformers; current={transformers.__version__})",
                    flush=True,
                )
            # Construct the state from the known fields only.
            return cls(**{k: v for (k, v) in _data.items() if k in _ts_allowed})

        _TrainerState.load_from_json = _ts_load_from_json_compat
except Exception as _e:
    # Unlike the other shims, a failure here is reported, because a resume
    # from a newer-Transformers checkpoint would then fail later.
    print(
        f"[RESUME][WARN] failed to install TrainerState.load_from_json compat shim: {_e}",
        flush=True,
    )

# Trainer RNG snapshots contain NumPy state as well as tensor weights.
# Preserve the original training scripts' explicit trusted-checkpoint default.
#
# PyTorch >= 2.6 changed the default of ``torch.load(weights_only=...)`` to
# True, which refuses to unpickle the NumPy RNG state in ``rng_state*.pth``
# (loaded by ``Trainer`` on resume) and other non-tensor objects in
# optimizer/scheduler/checkpoint files. This replaces ``torch.load`` for the
# whole process with a wrapper whose default is ``weights_only=False``; an
# explicit ``weights_only=`` from the caller is still honoured. Because it is
# global, it also applies to every other ``torch.load`` call made through the
# ``torch`` module after this import. Only load checkpoints from trusted sources.
# (train.py additionally disables Transformers' ``check_torch_load_is_safe``.)
import torch

# Keep a handle to the real implementation for the wrapper to call.
_original_torch_load = torch.load


def _load_training_checkpoint(
    f, map_location=None, pickle_module=None, *, weights_only=False, **kwargs
):
    """``torch.load`` with ``weights_only`` defaulting to ``False``.

    Args:
        f: File path or file-like object, as for ``torch.load``.
        map_location: Forwarded to ``torch.load`` (e.g. ``"cpu"``).
        pickle_module: Forwarded to ``torch.load`` (``None`` = PyTorch default).
        weights_only: Keyword-only; ``False`` unless the caller overrides it.
        **kwargs: Any further ``torch.load`` keyword arguments (e.g. ``mmap``).

    Returns:
        The deserialized object, exactly as ``torch.load`` returns it.
    """
    return _original_torch_load(
        f,
        map_location=map_location,
        pickle_module=pickle_module,
        weights_only=weights_only,
        **kwargs,
    )


# Install the wrapper process-wide.
torch.load = _load_training_checkpoint
