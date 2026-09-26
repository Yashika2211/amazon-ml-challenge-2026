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
    both = pl.concat([A.with_columns(pl.lit(0).alias("side")), B.with_columns(pl.lit(1).alias("side"))], how="diagonal_relaxed")
    n_rec = both.height
    rec = {}
    for col, width, key in (("core_tok", NT, "nt"), ("addr_tok", NA, "at"), ("addr_comp", NC, "ac")):
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
        rec["A_" + key] = _ids(A, col, vocab, width)
        rec["B_" + key] = _ids(B, col, vocab, width)
    for side, df in (("A", A), ("B", B)):
        rec[side + "_num"] = (df.select(pl.col("addr_num").list.head(NN).list.eval(pl.element().hash())
                                        .list.concat(pl.lit([0] * NN, dtype=pl.List(pl.UInt64))).list.head(NN)
                                        .list.to_array(NN))["addr_num"].to_numpy().astype(np.uint64))
        # count of S1 records sharing the exact core string -> "common name"
    for side, df in (("A", A), ("B", B)):
        rec[side + "_ch"] = df["compact"].hash().to_numpy()
        rec[side + "_aempty"] = df["addr_empty"].to_numpy()
    cf = both.group_by("core").len()
    for side, df in (("A", A), ("B", B)):
        rec[side + "_corefreq"] = df.select("core").join(cf, on="core", how="left")["len"].to_numpy().astype(np.int32)
    np.savez(path, **rec)
    return rec


def _set_feats(a_ids, b_ids, idf, df=None, gen=None, prefix=""):
    """IDF-weighted set overlap between padded id arrays (n, w)."""
    va, vb = a_ids >= 0, b_ids >= 0
    wa = np.where(va, idf[np.maximum(a_ids, 0)], 0.0)
    wb = np.where(vb, idf[np.maximum(b_ids, 0)], 0.0)
    eq = (a_ids[:, :, None] == b_ids[:, None, :]) & va[:, :, None]
    ma = eq.any(2)  # a-token matched
    mb = eq.any(1)
    sa, sb = wa.sum(1), wb.sum(1)
    inter = (wa * ma).sum(1)
    union = sa + sb - inter
    f = {
        prefix + "wjac": np.divide(inter, union, out=np.zeros_like(inter), where=union > 0),
        prefix + "wov": np.divide(inter, np.minimum(sa, sb), out=np.zeros_like(inter), where=np.minimum(sa, sb) > 0),
        prefix + "wcov_a": np.divide(inter, sa, out=np.zeros_like(inter), where=sa > 0),
        prefix + "wcov_b": np.divide((wb * mb).sum(1), sb, out=np.zeros_like(inter), where=sb > 0),
        prefix + "nmatch": ma.sum(1).astype(np.float32),
        prefix + "na": va.sum(1).astype(np.float32),
        prefix + "nb": vb.sum(1).astype(np.float32),
        prefix + "maxidf_un_a": np.where(va & ~ma, wa, 0).max(1),
        prefix + "maxidf_un_b": np.where(vb & ~mb, wb, 0).max(1),
    }
    if df is not None:
        dfa = np.where(va & ~ma, df[np.maximum(a_ids, 0)], 0)
        dfb = np.where(vb & ~mb, df[np.maximum(b_ids, 0)], 0)
        f[prefix + "maxdf_un_a"] = np.log1p(dfa.max(1)).astype(np.float32)
        f[prefix + "maxdf_un_b"] = np.log1p(dfb.max(1)).astype(np.float32)
    if gen is not None:
        g_b = np.where(vb, gen[np.maximum(b_ids, 0)], 0)
        g_a = np.where(va, gen[np.maximum(a_ids, 0)], 0)
        # distinctive unmatched tokens: seen elsewhere (df>=3) and not a known noise token
        dist_b = vb & ~mb & (df[np.maximum(b_ids, 0)] >= 3) & (g_b < 0.3)
        dist_a = va & ~ma & (df[np.maximum(a_ids, 0)] >= 3) & (g_a < 0.3)
        f[prefix + "ndist_un_b"] = dist_b.sum(1).astype(np.float32)
        f[prefix + "ndist_un_a"] = dist_a.sum(1).astype(np.float32)
        f[prefix + "gen_un_b"] = np.where(vb & ~mb, g_b, 0).sum(1).astype(np.float32)
        f[prefix + "gen_frac_b"] = np.divide(np.where(vb, g_b, 0).sum(1), vb.sum(1), out=np.zeros(len(vb), np.float32), where=vb.sum(1) > 0)
        f[prefix + "wov_nogen"] = np.divide(((wa * ma) * (1 - g_a)).sum(1), (wa * (1 - g_a)).sum(1),
                                            out=np.zeros(len(va), np.float32), where=(wa * (1 - g_a)).sum(1) > 0)
        f[prefix + "first_eq"] = (a_ids[:, 0] == b_ids[:, 0]) & (a_ids[:, 0] >= 0)
    return {k: np.asarray(v, dtype=np.float32) for k, v in f.items()}


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
    f.update(_set_feats(rec["A_nt"][ai], rec["B_nt"][bi], rec["nt_idf"], rec["nt_df"], rec["nt_gen"], "nt_"))
    f.update(_set_feats(rec["A_at"][ai], rec["B_at"][bi], rec["at_idf"], rec["at_df"], None, "at_"))
    f.update(_set_feats(rec["A_ac"][ai], rec["B_ac"][bi], rec["ac_idf"], None, None, "ac_"))
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
    f["corefreq_a"] = np.log1p(rec["A_corefreq"][ai]).astype(np.float32)
    f["corefreq_b"] = np.log1p(rec["B_corefreq"][bi]).astype(np.float32)
    out = pairs.select("a_idx", "b_idx").with_columns([pl.Series(k, np.asarray(v, np.float32)) for k, v in f.items()])
    return out
