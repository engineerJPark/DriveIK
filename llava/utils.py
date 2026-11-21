"""Rank-aware printing helpers for (optionally) distributed runs.

Used throughout ``drivevla/training`` (``rank0_print``) so that log lines are not
duplicated once per GPU process under ``torch.distributed`` / DeepSpeed.
"""

import torch.distributed as dist


def rank0_print(*args):
    """Print only from the global rank-0 process.

    Args:
        *args: Objects forwarded unchanged to ``print``.

    Behavior:
        * If ``torch.distributed`` is initialized, only rank 0 prints, and the output
          is prefixed with ``"Rank 0: "``; all other ranks print nothing.
        * If it is not initialized (single-process run), every call prints ``args``
          without a prefix.

    Returns:
        None. Side effect: writes to stdout.
    """
    # Distributed run: filter to rank 0 and tag the line with the rank.
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    # Non-distributed run: plain print.
    else:
        print(*args)


def rank_print(*args):
    """Print from every process, prefixed with the caller's rank when distributed.

    Args:
        *args: Objects forwarded unchanged to ``print``.

    Returns:
        None. Side effect: writes to stdout (once per process when distributed).
    """
    # Distributed run: every rank prints, tagged with its own rank id.
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    # Non-distributed run: plain print.
    else:
        print(*args)
