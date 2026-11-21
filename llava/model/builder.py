"""Model-loading helpers.

Contents:

* An import-time patch that replaces ``transformers``' ``check_torch_load_is_safe`` with
  a no-op (see the comment above the ``try`` block).
* ``_materialize_meta_tensors``: after ``from_pretrained``, replace parameters / buffers
  that are still on the ``meta`` device by real, freshly initialized tensors.

``drivevla/training/train.py`` imports ``_materialize_meta_tensors`` lazily, right after
it has loaded the base model with ``LlavaQwenForCausalLM.from_pretrained``. Importing
this module is also what applies the patch, so the patch only affects model loads that
happen after that import.
"""

import torch

# Newer `transformers` releases define `check_torch_load_is_safe()`, which raises before
# `torch.load` is used on pickle checkpoints (e.g. `.bin` files) when the installed torch
# is older than the version they require. This block replaces that check with a no-op in
# both modules that reference it, so such checkpoints load regardless of the torch
# version. `hasattr` guards make it a no-op on releases without the function, and any
# import error is ignored. Note that this disables a safety check of `transformers`.
try:
    import transformers.utils.import_utils as _tiu

    if hasattr(_tiu, "check_torch_load_is_safe"):
        _tiu.check_torch_load_is_safe = lambda: None
    import transformers.modeling_utils as _tmu

    if hasattr(_tmu, "check_torch_load_is_safe"):
        _tmu.check_torch_load_is_safe = lambda: None
except Exception:
    pass


def _materialize_meta_tensors(
    model: torch.nn.Module, device: str = "cpu", dtype: torch.dtype | None = None
) -> list[str]:
    """Replace any parameters/buffers still on the `meta` device with real tensors.

    This is a safety net for checkpoints that introduce new modules/weights while loading with
    `low_cpu_mem_usage` enabled, which can leave some tensors on `meta` and later crash on `.to()`.
    Returns the list of tensor names that were materialized.

    ``drivevla/training/train.py`` loads with ``low_cpu_mem_usage=False`` and calls this
    function only if it actually finds meta tensors after loading.

    Args:
        model: Module to fix in place (normally the whole ``LlavaQwenForCausalLM``).
        device: Device on which the replacement tensors are allocated (default CPU).
        dtype: Target dtype for floating-point / complex replacements. ``None`` keeps
            the dtype recorded on the meta tensor. Integer / bool tensors always keep
            their own dtype.

    Side effects:
        * Each meta **parameter** is replaced by a new ``nn.Parameter`` of the same shape,
          initialized with ``N(0, 0.02)`` (floating / complex) or zeros (other dtypes),
          keeping the original ``requires_grad`` flag.
        * Each meta **buffer** is replaced by a zero tensor of the same shape.
        * ``model.tie_weights()`` is called afterwards (errors ignored), so tied weights
          such as Qwen's ``lm_head`` / ``embed_tokens`` share storage again.

    Returns:
        list[str]: Dotted names of the parameters and buffers that were replaced
        (empty if nothing was on ``meta``).
    """
    import torch.nn as nn

    def _resolve_parent(root: torch.nn.Module, name: str):
        """Walk a dotted tensor name and return ``(owning_module, attribute_name)``.

        E.g. ``"model.layers.3.mlp.up_proj.weight"`` -> ``(<up_proj Linear>, "weight")``.
        Numeric path components index into ``nn.ModuleList`` / ``nn.Sequential``; keys of
        an ``nn.ModuleDict`` are looked up by item; anything else uses ``getattr``.
        """
        parts = name.split(".")
        obj = root
        for p in parts[:-1]:
            if isinstance(obj, (nn.ModuleList, nn.Sequential)) and p.isdigit():
                obj = obj[int(p)]
            elif isinstance(obj, nn.ModuleDict):
                obj = obj[p]
            else:
                obj = getattr(obj, p)
        return obj, parts[-1]

    # Names of the tensors replaced below (the return value).
    materialized: list[str] = []

    def _is_meta_tensor(t: torch.Tensor) -> bool:
        """Robust meta check across torch/accelerate variants.

        Checks ``t.is_meta`` first and then ``t.device.type == "meta"``; any exception
        during either check counts as "not meta".
        """
        try:
            if getattr(t, "is_meta", False):
                return True
        except Exception:
            pass
        try:
            return getattr(t, "device", None) is not None and t.device.type == "meta"
        except Exception:
            return False

    # Parameters
    # Iterate over a snapshot (list) because setattr below mutates the module tree.
    for name, param in list(model.named_parameters(recurse=True)):
        if _is_meta_tensor(param):
            parent, attr = _resolve_parent(model, name)
            # Use the requested dtype only for floating / complex parameters; integer
            # parameters keep their own dtype.
            tgt_dtype = (
                dtype
                if (dtype is not None and (param.is_floating_point() or param.is_complex()))
                else param.dtype
            )
            new_t = torch.empty(param.shape, device=device, dtype=tgt_dtype)
            # Initialize reasonably; this affects only params that were never loaded from the checkpoint.
            if new_t.is_floating_point() or new_t.is_complex():
                nn.init.normal_(new_t, mean=0.0, std=0.02)
            else:
                new_t.zero_()
            # Re-register on the owning module so the old meta parameter is dropped.
            setattr(parent, attr, nn.Parameter(new_t, requires_grad=param.requires_grad))
            materialized.append(name)

    # Buffers
    # Same dtype rule as for parameters; buffers are always zero-filled. Assigning a
    # tensor to a registered buffer name through setattr keeps it registered as a buffer
    # (nn.Module.__setattr__ routes it into `_buffers`).
    for name, buf in list(model.named_buffers(recurse=True)):
        if _is_meta_tensor(buf):
            parent, attr = _resolve_parent(model, name)
            tgt_dtype = (
                dtype
                if (dtype is not None and (buf.is_floating_point() or buf.is_complex()))
                else buf.dtype
            )
            new_b = torch.empty(buf.shape, device=device, dtype=tgt_dtype)
            new_b.zero_()
            setattr(parent, attr, new_b)
            materialized.append(name)

    # Re-tie weights if supported (safe no-op for most models)
    try:
        model.tie_weights()
    except Exception:
        pass

    return materialized
