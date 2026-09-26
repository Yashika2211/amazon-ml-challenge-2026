"""Pair features (numeric, country-agnostic).

Record-level arrays (token ids padded to fixed width, idf, skeleton strings) are built once
per split; pair features are then computed chunk-wise with rapidfuzz.cpdist and numpy.
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from fastops import ov_jac, sort_ids, weighted_overlap
from io_utils import WORK

NT, NA, NN, NC = 8, 16, 6, 8  # widths: name tokens, address tokens, numbers, components
_SK = [("ph", "f"), ("x", "ks"), ("ck", "k"), ("c", "k"), ("q", "k"), ("w", "v"), ("z", "j"),
       ("y", "i"), ("sh", "s"), ("th", "t"), ("dh", "d"), ("bh", "b"), ("kh", "k"), ("gh", "g"), ("m", "n")]


def skeleton_expr(col: str) -> pl.Expr:
    """Vectorized consonant skeleton of every token (see mine.skeleton)."""
    e = pl.col(col).list.eval(pl.element().str.replace_many([a for a, _ in _SK], [b for _, b in _SK]))
    e = e.list.eval(pl.element().str.slice(0, 1) + pl.element().str.slice(1).str.replace_all("[aeiouh]", ""))
    e = e.list.join(" ")
    for ch in "bdfgjklnprstvk":
        e = e.str.replace_all(ch + "+", ch)
    return e


def _ids(df: pl.DataFrame, col: str, vocab: pl.DataFrame, width: int) -> np.ndarray:
    x = df.select(pl.int_range(pl.len()).alias("r"), pl.col(col).list.head(width).alias("t")).explode("t")
    x = x.join(vocab, on="t", how="left").with_columns(pl.col("id").fill_null(-1))
    x = x.with_columns(pl.int_range(pl.len()).over("r").alias("j"))
    out = np.full((df.height, width), -1, dtype=np.int32)
    r, j, v = x["r"].to_numpy(), x["j"].to_numpy(), x["id"].to_numpy()
    ok = v >= 0
    out[r[ok], j[ok]] = v[ok]
    return out


def build_records(split: str, A: pl.DataFrame, B: pl.DataFrame, generic: dict) -> dict:
    """Arrays indexed by global idx for both sides. Cached as npz."""
    path = os.path.join(WORK, f"rec_{split}.npz")
    if os.path.exists(path):
        z = np.load(path, allow_pickle=True)
        return {k: z[k] for k in z.files}
    # name tokens for matching = core tokens plus the DBA / AKA / FKA alternate name
    A = A.with_columns(pl.concat_list("core_tok", "alt_tok").list.unique(maintain_order=True).alias("core_tok"))
    B = B.with_columns(pl.concat_list("core_tok", "alt_tok").list.unique(maintain_order=True).alias("core_tok"))
    both = pl.concat([A.with_columns(pl.lit(0).alias("side")), B.with_columns(pl.lit(1).alias("side"))], how="diagonal_relaxed")
    n_rec = both.height
    rec = {}
    for col, width, key in (("core_tok", NT, "nt"), ("addr_tok", NA, "at"), ("addr_comp", NC, "ac"), ("addr_num", NN, "nm")):
        v = both.select(pl.col(col).list.unique().alias("t")).explode("t").drop_nulls().group_by("t").len()
        v = v.with_row_index("id").with_columns(pl.col("id").cast(pl.Int32))
        idf = np.log(n_rec / v["len"].to_numpy().astype(np.float64)).astype(np.float32)
        rec[key + "_idf"] = idf
        rec[key + "_df"] = v["len"].to_numpy().astype(np.int32)
        if key == "nt":
            g = np.zeros(len(v), np.float32)
            gm = v.select("id", "t").join(pl.DataFrame({"t": list(generic), "g": list(generic.values())}), on="t")
            g[gm["id"].to_numpy()] = gm["g"].to_numpy()
            rec["nt_gen"] = g
        vocab = v.select("t", "id")
        for side, df in (("A", A), ("B", B)):
            ids = _ids(df, col, vocab, width)
            rec[f"{side}_{key}0"] = ids[:, 0].copy()  # first token (order matters for first_eq)
            rec[f"{side}_{key}"] = sort_ids(ids)       # sorted rows for merge-intersection kernels
    for side, df in (("A", A), ("B", B)):
        rec[side + "_num"] = (df.select(pl.col("addr_num").list.head(NN).list.eval(pl.element().hash())
                                        .list.concat(pl.lit([0] * NN, dtype=pl.List(pl.UInt64))).list.head(NN)
                                        .list.to_array(NN))["addr_num"].to_numpy().astype(np.uint64))
        # count of S1 records sharing the exact core string -> "common name"
    for side, df in (("A", A), ("B", B)):
        # first 3 address numbers as integers (<= 9 digits) for digit-edit features
        v = (df.select(pl.col("addr_num").list.eval(pl.element().filter(pl.element().str.len_chars() <= 9))
                       .list.head(3).list.eval(pl.element().cast(pl.Int64))
                       .list.concat(pl.lit([-1, -1, -1], dtype=pl.List(pl.Int64))).list.head(3).list.to_array(3))
             ["addr_num"].to_numpy())
        rec[side + "_numv"] = v.astype(np.int64)
        rec[side + "_ch"] = df["compact"].hash().to_numpy()
        rec[side + "_aempty"] = df["addr_empty"].to_numpy()
        rec[side + "_src"] = df["src"].to_numpy().astype(np.int8)
    cf = both.group_by("core").len()
    for side, df in (("A", A), ("B", B)):
        rec[side + "_corefreq"] = df.select("core").join(cf, on="core", how="left")["len"].to_numpy().astype(np.int32)
    np.savez(path, **rec)
    return rec


def _set_feats(rec: dict, key: str, ai: np.ndarray, bi: np.ndarray, idf, df=None, gen=None, prefix=""):
    """IDF-weighted set overlap between sorted padded id rows (numba merge kernel)."""
    A, B = rec["A_" + key], rec["B_" + key]
    inter, sa, sb, nm, ua, ub = weighted_overlap(A, B, ai, bi, idf)
    ones = np.ones_like(idf)
    _, na, nb, _, _, _ = weighted_overlap(A, B, ai, bi, ones)
    ib = inter  # symmetric: matched weight is the same on both sides
    z = np.zeros_like(inter)
    f = {
        prefix + "wjac": np.divide(inter, sa + sb - inter, out=z.copy(), where=(sa + sb - inter) > 0),
        prefix + "wov": np.divide(inter, np.minimum(sa, sb), out=z.copy(), where=np.minimum(sa, sb) > 0),
        prefix + "wcov_a": np.divide(inter, sa, out=z.copy(), where=sa > 0),
        prefix + "wcov_b": np.divide(ib, sb, out=z.copy(), where=sb > 0),
        prefix + "nmatch": nm,
        prefix + "na": na,
        prefix + "nb": nb,
        prefix + "maxidf_un_a": ua,
        prefix + "maxidf_un_b": ub,
    }
    if df is not None:
        dfw = df.astype(np.float32)
        _, _, _, _, ma_, mb_ = weighted_overlap(A, B, ai, bi, dfw)
        f[prefix + "maxdf_un_a"] = np.log1p(ma_)
        f[prefix + "maxdf_un_b"] = np.log1p(mb_)
    if gen is not None:
        dist = ((df >= 3) & (gen < 0.3)).astype(np.float32)
        i_d, s_d_a, s_d_b, _, _, _ = weighted_overlap(A, B, ai, bi, dist)
        f[prefix + "ndist_un_b"] = s_d_b - i_d
        f[prefix + "ndist_un_a"] = s_d_a - i_d
        i_g, s_g_a, s_g_b, _, _, _ = weighted_overlap(A, B, ai, bi, gen.astype(np.float32))
        f[prefix + "gen_un_b"] = s_g_b - i_g
        f[prefix + "gen_frac_b"] = np.divide(s_g_b, nb, out=z.copy(), where=nb > 0)
        wng = (idf * (1 - gen)).astype(np.float32)
        i_n, s_n_a, _, _, _, _ = weighted_overlap(A, B, ai, bi, wng)
        f[prefix + "wov_nogen"] = np.divide(i_n, s_n_a, out=z.copy(), where=s_n_a > 0)
        a0, b0 = rec["A_" + key + "0"][ai], rec["B_" + key + "0"][bi]
        f[prefix + "first_eq"] = ((a0 == b0) & (a0 >= 0)).astype(np.float32)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


def _ndig(x: np.ndarray) -> np.ndarray:
    n = np.ones_like(x)
    for k in range(1, 10):
        n += (x >= 10 ** k)
    return n


def digit_edit1(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """True where non-negative ints a != b are within one digit edit (sub / insert / delete)."""
    ok = (a >= 0) & (b >= 0) & (a != b)
    la, lb = _ndig(np.maximum(a, 0)), _ndig(np.maximum(b, 0))
    # substitution: same length, exactly one differing digit
    diff = np.zeros_like(a)
    x, y = a.copy(), b.copy()
    for _ in range(10):
        diff += (x % 10) != (y % 10)
        x //= 10
        y //= 10
    sub = (la == lb) & (diff == 1)
    # deletion: remove one digit from the longer number
    lo, sh = np.where(la > lb, a, b), np.where(la > lb, b, a)
    dele = np.zeros(a.shape, bool)
    for k in range(10):
        p = 10 ** k
        dele |= ((lo // (p * 10)) * p + lo % p) == sh
    dele &= np.abs(la - lb) == 1
    return ok & (sub | dele)


def _cp(fn, x, y):
    return process.cpdist(x, y, scorer=fn, workers=-1, dtype=np.float32) / 100.0


STR_COLS_A = ["core", "compact", "name_norm", "addr", "skel", "legal", "alt"]


def string_table(df: pl.DataFrame) -> pl.DataFrame:
    return df.select(
        "idx", "core", "compact", "name_norm", "addr", "legal", "native", "addr_empty",
        skeleton_expr("core_tok").alias("skel"),
        pl.col("alt_tok").list.join(" ").alias("alt"))


def pair_features(pairs: pl.DataFrame, SA: pl.DataFrame, SB: pl.DataFrame, rec: dict,
                  posA: np.ndarray, posB: np.ndarray) -> pl.DataFrame:
    """Compute features for a chunk of pairs. SA/SB are string tables sorted by idx."""
    ai = posA[pairs["a_idx"].to_numpy()]
    bi = posB[pairs["b_idx"].to_numpy()]
    ga = {c: SA[c].gather(ai).to_list() for c in ("core", "compact", "name_norm", "addr", "skel", "alt")}
    gb = {c: SB[c].gather(bi).to_list() for c in ("core", "compact", "name_norm", "addr", "skel", "alt")}
    f = {}
    f["n_tset"] = _cp(fuzz.token_set_ratio, ga["core"], gb["core"])
    f["n_tsort"] = _cp(fuzz.token_sort_ratio, ga["core"], gb["core"])
    f["n_ratio"] = _cp(fuzz.ratio, ga["core"], gb["core"])
    f["n_pratio"] = _cp(fuzz.partial_ratio, ga["core"], gb["core"])
    f["c_ratio"] = _cp(fuzz.ratio, ga["compact"], gb["compact"])
    f["c_pratio"] = _cp(fuzz.partial_ratio, ga["compact"], gb["compact"])
    f["c_jw"] = process.cpdist(ga["compact"], gb["compact"], scorer=JaroWinkler.normalized_similarity,
                               workers=-1, dtype=np.float32)
    f["full_tset"] = _cp(fuzz.token_set_ratio, ga["name_norm"], gb["name_norm"])
    f["sk_ratio"] = _cp(fuzz.ratio, ga["skel"], gb["skel"])
    f["sk_tsort"] = _cp(fuzz.token_sort_ratio, ga["skel"], gb["skel"])
    f["alt_b"] = _cp(fuzz.token_set_ratio, ga["core"], gb["alt"])
    f["alt_a"] = _cp(fuzz.token_set_ratio, ga["alt"], gb["core"])
    f["a_tset"] = _cp(fuzz.token_set_ratio, ga["addr"], gb["addr"])
    f["a_tsort"] = _cp(fuzz.token_sort_ratio, ga["addr"], gb["addr"])
    f["a_pratio"] = _cp(fuzz.partial_ratio, ga["addr"], gb["addr"])
    lena = np.array([len(s) for s in ga["compact"]], np.float32)
    lenb = np.array([len(s) for s in gb["compact"]], np.float32)
    f["len_a"], f["len_b"] = lena, lenb
    f["c_contain"] = np.array([(x in y or y in x) and min(len(x), len(y)) >= 4 for x, y in zip(ga["compact"], gb["compact"])], np.float32)
    la, lb = SA["legal"].gather(ai).to_numpy(), SB["legal"].gather(bi).to_numpy()
    f["legal_eq"] = ((la == lb) & (la != "")).astype(np.float32)
    f["legal_conf"] = ((la != lb) & (la != "") & (lb != "")).astype(np.float32)
    f["legal_miss"] = ((la == "") != (lb == "")).astype(np.float32)
    f["native_b"] = SB["native"].gather(bi).to_numpy().astype(np.float32)
    f["aempty_b"] = SB["addr_empty"].gather(bi).to_numpy().astype(np.float32)
    ra, rb = pairs["a_idx"].to_numpy(), pairs["b_idx"].to_numpy()
    f.update(_set_feats(rec, "nt", ai, bi, rec["nt_idf"], rec["nt_df"], rec["nt_gen"], "nt_"))
    f.update(_set_feats(rec, "at", ai, bi, rec["at_idf"], rec["at_df"], None, "at_"))
    f.update(_set_feats(rec, "ac", ai, bi, rec["ac_idf"], None, None, "ac_"))
    na, nb = rec["A_num"][ai], rec["B_num"][bi]
    va, vb = na != 0, nb != 0
    eqn = (na[:, :, None] == nb[:, None, :]) & va[:, :, None]
    inter = eqn.any(2).sum(1)
    cnt_a, cnt_b = va.sum(1), vb.sum(1)
    un = cnt_a + cnt_b - inter
    f["num_jac"] = np.divide(inter, un, out=np.zeros(len(ai), np.float64), where=un > 0).astype(np.float32)
    f["num_inter"] = inter.astype(np.float32)
    f["num_first_eq"] = ((na[:, 0] == nb[:, 0]) & va[:, 0]).astype(np.float32)
    f["num_first_in"] = (eqn[:, 0, :].any(1)).astype(np.float32)
    f["num_conf"] = ((cnt_a > 0) & (cnt_b > 0) & (inter == 0)).astype(np.float32)
    f["num_na"], f["num_nb"] = cnt_a.astype(np.float32), cnt_b.astype(np.float32)
    va_, vb_ = rec["A_numv"][ai], rec["B_numv"][bi]
    e1 = np.zeros(len(ai), bool)
    for i in range(3):
        for j in range(3):
            e1 |= digit_edit1(va_[:, i], vb_[:, j])
    f["num_edit1"] = e1.astype(np.float32)
    f["num_first_edit1"] = digit_edit1(va_[:, 0], vb_[:, 0]).astype(np.float32)
    # size of house-number shift: distractors are copies at a nearby but different number
    both0 = (va_[:, 0] >= 0) & (vb_[:, 0] >= 0)
    d0 = np.abs(va_[:, 0] - vb_[:, 0]).astype(np.float64)
    f["num_first_logdiff"] = np.where(both0, np.log1p(d0), -1).astype(np.float32)
    f["num_first_reldiff"] = np.where(both0, d0 / np.maximum(np.maximum(va_[:, 0], vb_[:, 0]), 1), -1).astype(np.float32)
    mind = np.full(len(ai), np.inf)
    for i in range(3):
        for j in range(3):
            ok = (va_[:, i] >= 0) & (vb_[:, j] >= 0)
            mind = np.where(ok, np.minimum(mind, np.abs(va_[:, i] - vb_[:, j])), mind)
    f["num_min_logdiff"] = np.where(np.isfinite(mind), np.log1p(np.where(np.isfinite(mind), mind, 0)), -1).astype(np.float32)
    f["num_near_shift"] = (np.isfinite(mind) & (mind > 0) & (mind <= 50) & ~e1).astype(np.float32)
    f["corefreq_a"] = np.log1p(rec["A_corefreq"][ai]).astype(np.float32)
    f["corefreq_b"] = np.log1p(rec["B_corefreq"][bi]).astype(np.float32)
    out = pairs.select("a_idx", "b_idx").with_columns([pl.Series(k, np.asarray(v, np.float32)) for k, v in f.items()])
    return out
