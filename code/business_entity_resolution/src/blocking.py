"""Candidate generation (blocking), partitioned by country.

Keys (hashed to u64), union over:
  t  the 3 rarest core-name tokens (document-frequency capped)
  b  bigram of the 2 rarest core-name tokens
  c/p/s  compact name: full, 6-char prefix, 6-char suffix
  a  (address number, rare address token)
  d  bigrams of the 3 rarest non-numeric address tokens (street / locality)
  x  (rarest name token, rarest address token)
  y  (rare name token, address number)
  e  1-deletion neighbourhood of the 2 rarest name tokens (typos)
Keys whose |A|*|B| block product exceeds a cap are skipped.

Candidates are then ranked with a cheap rapidfuzz similarity; we keep the top K_A per
S1 and additionally any pair ranked in the top K_B S1s of its S2/S3 record.
"""
from __future__ import annotations

import time

import numpy as np
import polars as pl

TOK_DF_CAP = 5000
PAIR_CAP = {"t": 40_000, "b": 200_000, "c": 200_000, "p": 20_000, "s": 20_000, "a": 20_000,
            "d": 20_000, "x": 20_000, "y": 20_000, "e": 40_000}
KEY_TYPES = list(PAIR_CAP)


def _df_table(a: pl.DataFrame, b: pl.DataFrame, col: str) -> pl.DataFrame:
    toks = pl.concat([a.select(pl.col(col).list.unique().alias("t")), b.select(pl.col(col).list.unique().alias("t"))])
    return toks.explode("t").drop_nulls().group_by("t").len().rename({"len": "df"})


def _rare(df: pl.DataFrame, col: str, dft: pl.DataFrame, k: int, cap: int | None, filt=None) -> pl.DataFrame:
    x = df.select("idx", pl.col(col).alias("t")).explode("t").drop_nulls()
    if filt is not None:
        x = x.filter(filt)
    x = x.join(dft, on="t")
    if cap is not None:
        x = x.filter(pl.col("df") <= cap)
    return x.sort("idx", "df", "t").group_by("idx", maintain_order=True).head(k)


def record_keys(df: pl.DataFrame, name_df: pl.DataFrame, addr_df: pl.DataFrame) -> pl.DataFrame:
    """Long table (idx, kt, key) of blocking keys for one side."""
    name = df.select("idx", pl.concat_list("core_tok", "alt_tok").list.unique().alias("nt"))
    r = _rare(name, "nt", name_df, 3, TOK_DF_CAP)
    parts = [r.select("idx", pl.lit("t").alias("kt"), pl.col("t").alias("key"))]
    r2 = _rare(name, "nt", name_df, 2, None).group_by("idx").agg(pl.col("t").sort().str.join(" ").alias("key"),
                                                               pl.len().alias("n"))
    parts.append(r2.filter(pl.col("n") == 2).select("idx", pl.lit("b").alias("kt"), "key"))
    comp = df.select("idx", "compact").filter(pl.col("compact").str.len_chars() >= 4)
    parts.append(comp.select("idx", pl.lit("c").alias("kt"), pl.col("compact").alias("key")))
    long = comp.filter(pl.col("compact").str.len_chars() >= 8)
    parts.append(long.select("idx", pl.lit("p").alias("kt"), pl.col("compact").str.slice(0, 6).alias("key")))
    parts.append(long.select("idx", pl.lit("s").alias("kt"), pl.col("compact").str.slice(-6).alias("key")))
    at = _rare(df, "addr_tok", addr_df, 3, None, ~pl.col("t").str.contains(r"^[0-9]+$"))
    nums = df.select("idx", pl.col("addr_num").list.head(2).alias("n")).explode("n").drop_nulls()
    parts.append(nums.join(at.select("idx", "t"), on="idx").select(
        "idx", pl.lit("a").alias("kt"), (pl.col("n") + "|" + pl.col("t")).alias("key")))
    # bigrams of the 3 rarest street/locality tokens (3 keys)
    at_l = at.group_by("idx").agg(pl.col("t"))
    for i, j in ((0, 1), (0, 2), (1, 2)):
        bg = at_l.filter(pl.col("t").list.len() > j).select(
            "idx", pl.concat_list(pl.col("t").list.get(i), pl.col("t").list.get(j)).list.sort().list.join(" ").alias("key"))
        parts.append(bg.select("idx", pl.lit("d").alias("kt"), "key"))
    r1 = _rare(name, "nt", name_df, 1, None).select("idx", pl.col("t").alias("nt1"))
    a1 = at.group_by("idx", maintain_order=True).first().select("idx", pl.col("t").alias("at1"))
    parts.append(r1.join(a1, on="idx").select("idx", pl.lit("x").alias("kt"),
                                              (pl.col("nt1") + "|" + pl.col("at1")).alias("key")))
    # (rare name token, address number): same business number, name partially intact
    r2n = _rare(name, "nt", name_df, 2, None).select("idx", "t")
    parts.append(r2n.join(nums, on="idx").select("idx", pl.lit("y").alias("kt"),
                                                (pl.col("t") + "|" + pl.col("n")).alias("key")))
    # typo tolerance: 1-deletion neighbourhood of the 2 rarest name tokens (len >= 5)
    lng = r2n.rename({"t": "nt1"}).filter(pl.col("nt1").str.len_chars() >= 5)
    dels = [lng.select("idx", pl.col("nt1").alias("key"))]
    for k in range(12):
        dels.append(lng.filter(pl.col("nt1").str.len_chars() > k).select(
            "idx", (pl.col("nt1").str.slice(0, k) + pl.col("nt1").str.slice(k + 1)).alias("key")))
    parts.append(pl.concat(dels).select("idx", pl.lit("e").alias("kt"), "key"))
    out = pl.concat(parts).unique()
    return out.select("idx", "kt", (pl.col("kt") + ":" + pl.col("key")).hash().alias("h"))


def block_country(a: pl.DataFrame, b: pl.DataFrame, rec: dict, chunk: int = 100_000,
                  scorer=None, sample_mod: int | None = None, keep_raw: bool = False) -> pl.DataFrame:
    """Return (a_idx, b_idx, hit counts per key type, cheap score) for one country.

    `rec` holds record arrays indexed by global idx (features.build_records).
    `scorer(pairs, rec)` adds the ranking column "cheap" (default: hand-weighted cheap_score).
    `sample_mod`/`keep_raw` are used to collect un-pruned pairs for training the learned ranker.
    """
    scorer = scorer or cheap_score
    t0 = time.time()
    name_df = _df_table(a.select(pl.concat_list("core_tok", "alt_tok").alias("x")),
                        b.select(pl.concat_list("core_tok", "alt_tok").alias("x")), "x")
    addr_df = _df_table(a, b, "addr_tok")
    ka = record_keys(a, name_df, addr_df)
    kb = record_keys(b, name_df, addr_df)
    na = ka.group_by("h").len().rename({"len": "na"})
    nb = kb.group_by("h").len().rename({"len": "nb"})
    sizes = na.join(nb, on="h").with_columns((pl.col("na").cast(pl.Int64) * pl.col("nb")).alias("prod"))
    caps = ka.select("h", "kt").unique("h").join(sizes, on="h")
    cap_expr = pl.lit(0)
    for kt, cap in PAIR_CAP.items():
        cap_expr = pl.when(pl.col("kt") == kt).then(cap).otherwise(cap_expr)
    ok = caps.filter(pl.col("prod") <= cap_expr).select("h")
    kbit = pl.lit(0, pl.UInt64)
    for i, kt in enumerate(KEY_TYPES):
        kbit = pl.when(pl.col("kt") == kt).then(pl.lit(1 << (6 * i), pl.UInt64)).otherwise(kbit)
    ka = ka.join(ok, on="h", how="semi").with_columns(kbit.alias("kbit")).select("idx", "h", "kbit")
    kb = kb.join(ok, on="h", how="semi").select("h", pl.col("idx").alias("b_idx"))
    print(f"  keys built {time.time() - t0:.0f}s  A-keys {ka.height:,}  B-keys {kb.height:,}", flush=True)
    outs = []
    a_ids = a["idx"].to_numpy()
    for s in range(0, len(a_ids), chunk):
        lo, hi = a_ids[s], a_ids[min(s + chunk, len(a_ids)) - 1]
        sub = ka.filter(pl.col("idx").is_between(lo, hi))
        if sample_mod:
            sub = sub.filter(pl.col("idx") % sample_mod == 0)
        sub = sub.rename({"idx": "a_idx"})
        # per-key-type hit counts packed into 6-bit fields of one u64, summed in a single aggregation
        j = sub.join(kb, on="h")
        pairs = j.group_by("a_idx", "b_idx").agg(pl.col("kbit").sum())
        pairs = pairs.with_columns(
            *[((pl.col("kbit") // (1 << (6 * i))) % 64).cast(pl.UInt8).alias(f"k_{kt}") for i, kt in enumerate(KEY_TYPES)]
        ).drop("kbit")
        if keep_raw:
            outs.append(pairs)
            continue
        pairs = scorer(pairs, rec)
        # per-S1 pre-cut (generous) so the global per-record ranking fits in memory
        pairs = pairs.filter(pl.col("cheap").rank("ordinal", descending=True).over("a_idx") <= PRE_CUT)
        outs.append(pairs)
    res = pl.concat(outs)
    print(f"  pairs {res.height:,} in {time.time() - t0:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# cheap ranking + pruning (numeric only: padded token-id arrays, no string conversion)
# ---------------------------------------------------------------------------
K_A = 25
K_B = 5
PRE_CUT = 80


def _wov(rec: dict, key: str, ai: np.ndarray, bi: np.ndarray, idf: np.ndarray):
    """IDF-weighted overlap coefficient and Jaccard between sorted padded token-id rows."""
    from fastops import ov_jac, weighted_overlap
    inter, sa, sb, _, _, _ = weighted_overlap(rec["A_" + key], rec["B_" + key], ai, bi, idf)
    return ov_jac(inter, sa, sb)


def cheap_score(pairs: pl.DataFrame, rec: dict) -> pl.DataFrame:
    """Numeric name + address similarity used only to rank candidates within blocks."""
    ai, bi = pairs["a_idx"].to_numpy(), pairs["b_idx"].to_numpy()
    n_ov, n_jac = _wov(rec, "nt", ai, bi, rec["nt_idf"])
    name = np.maximum(0.5 * n_ov + 0.5 * n_jac, (rec["A_ch"][ai] == rec["B_ch"][bi]).astype(np.float64))
    _, a_jac = _wov(rec, "at", ai, bi, rec["at_idf"])
    from fastops import weighted_overlap
    num = weighted_overlap(rec["A_nm"], rec["B_nm"], ai, bi, np.ones(len(rec["nm_idf"]), np.float32))[3] > 0
    addr = np.where(rec["B_aempty"][bi], 0.6, a_jac + 0.2 * num)  # empty address: neutral, let the model decide
    hits = pairs.select(pl.sum_horizontal(pl.col("^k_.*$").cast(pl.Float32))).to_series().to_numpy()
    score = name + 0.8 * addr + 0.02 * np.minimum(hits, 5)
    return pairs.with_columns(pl.Series("cheap", score.astype(np.float32)))


def rank_and_prune(pairs: pl.DataFrame, k_a: int = K_A, k_b: int = K_B) -> pl.DataFrame:
    pairs = pairs.with_columns(
        pl.col("cheap").rank("ordinal", descending=True).over("a_idx").cast(pl.UInt16).alias("rank_a"),
        pl.col("cheap").rank("ordinal", descending=True).over("b_idx").cast(pl.UInt16).alias("rank_b"))
    return pairs.filter((pl.col("rank_a") <= k_a) | (pl.col("rank_b") <= k_b))
