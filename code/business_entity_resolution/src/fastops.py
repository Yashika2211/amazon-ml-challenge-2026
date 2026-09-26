"""Compiled pairwise set-overlap kernels (numba, BSD). Memory-light replacements for the
(n, w, w) numpy broadcasts used in blocking and pair features."""
from __future__ import annotations

import numpy as np
from numba import njit, prange


def sort_ids(ids: np.ndarray) -> np.ndarray:
    """Sort each padded row ascending with -1 padding moved to the end."""
    x = np.where(ids < 0, np.iinfo(np.int32).max, ids).astype(np.int32)
    x.sort(axis=1)
    return np.where(x == np.iinfo(np.int32).max, -1, x).astype(np.int32)


@njit(cache=True, inline="always")
def _merge(ra, rb, w):
    wa_ = ra.shape[0]
    wb_ = rb.shape[0]
    i = 0
    j = 0
    inter = 0.0
    sa = 0.0
    sb = 0.0
    nm = 0.0
    ua = 0.0
    ub = 0.0
    done = False
    while not done:
        xa = ra[i] if i < wa_ else -1
        xb = rb[j] if j < wb_ else -1
        if xa < 0 and xb < 0:
            done = True
        elif xb < 0 or (xa >= 0 and xa < xb):
            v = w[xa]
            sa += v
            ua = max(ua, v)
            i += 1
        elif xa < 0 or xb < xa:
            v = w[xb]
            sb += v
            ub = max(ub, v)
            j += 1
        else:
            v = w[xa]
            sa += v
            sb += v
            inter += v
            nm += 1.0
            i += 1
            j += 1
    return inter, sa, sb, nm, ua, ub


@njit(parallel=True, cache=True)
def weighted_overlap(A, B, ai, bi, w):
    """For sorted padded id rows A[ai[k]], B[bi[k]]: (shared weight, weight A, weight B,
    n shared, max weight of unmatched A token, max weight of unmatched B token)."""
    n = ai.shape[0]
    out = np.zeros((6, n), np.float32)
    for k in prange(n):
        r = _merge(A[ai[k]], B[bi[k]], w)
        for q in range(6):
            out[q, k] = r[q]
    return out[0], out[1], out[2], out[3], out[4], out[5]


def ov_jac(inter, sa, sb):
    mn = np.minimum(sa, sb)
    un = sa + sb - inter
    ov = np.divide(inter, mn, out=np.zeros_like(inter), where=mn > 0)
    jac = np.divide(inter, un, out=np.zeros_like(inter), where=un > 0)
    return ov, jac
