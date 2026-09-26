"""Exact competition metric: per-S1 F0.5, macro-averaged over all S1 entities."""
from __future__ import annotations

from typing import Dict, Iterable, Set

BETA2 = 0.25  # beta = 0.5


def f05(truth: Set[str], pred: Set[str]) -> float:
    """F0.5 for one S1 entity with the empty-set conventions of the challenge."""
    if not truth:
        return 1.0 if not pred else 0.0
    if not pred:
        return 0.0
    tp = len(truth & pred)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(truth)
    return (1 + BETA2) * p * r / (BETA2 * p + r)


def macro_f05(truth: Dict[str, Set[str]], pred: Dict[str, Set[str]], ids: Iterable[str] | None = None) -> float:
    """Mean F0.5 over `ids` (default: every S1 in `truth`)."""
    ids = list(truth.keys()) if ids is None else list(ids)
    if not ids:
        return 0.0
    return sum(f05(truth.get(i, set()), pred.get(i, set())) for i in ids) / len(ids)
