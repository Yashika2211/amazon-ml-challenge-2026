"""Candidate generation (blocking), partitioned by country.

Keys (hashed to u64), union over:
  t  the 3 rarest core-name tokens (document-frequency capped)
  b  bigram of the 2 rarest core-name tokens
  c/p/s  compact name: full, 6-char prefix, 6-char suffix
  a  (address number, rare address token)
  d  bigram of the 2 rarest non-numeric address tokens (street / locality)
  x  (rarest name token, rarest address token)
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
            "d": 20_000, "x": 20_000}
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
    at = _rare(df, "addr_tok", addr_df, 2, None, ~pl.col("t").str.contains(r"^[0-9]+$"))
    nums = df.select("idx", pl.col("addr_num").list.head(2).alias("n")).explode("n").drop_nulls()
    an = nums.join(at.select("idx", "t"), on="idx").select(
        "idx", pl.lit("a").alias("kt"), (pl.col("n") + "|" + pl.col("t")).alias("key"))
    parts.append(an)
    at2 = at.group_by("idx").agg(pl.col("t").sort().str.join(" ").alias("key"), pl.len().alias("n"))
    parts.append(at2.filter(pl.col("n") == 2).select("idx", pl.lit("d").alias("kt"), "key"))
    r1 = _rare(name, "nt", name_df, 1, None).select("idx", pl.col("t").alias("nt1"))
    a1 = at.group_by("idx", maintain_order=True).first().select("idx", pl.col("t").alias("at1"))
    parts.append(r1.join(a1, on="idx").select("idx", pl.lit("x").alias("kt"),
                                              (pl.col("nt1") + "|" + pl.col("at1")).alias("key")))
    out = pl.concat(parts).unique()
    return out.select("idx", "kt", (pl.col("kt") + ":" + pl.col("key")).hash().alias("h"))


def block_country(a: pl.DataFrame, b: pl.DataFrame, rec: dict, chunk: int = 100_000) -> pl.DataFrame:
    """Return (a_idx, b_idx, hit counts per key type, cheap score) for one country.

    `rec` holds record arrays indexed by global idx (features.build_records).
    """
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
    ka = ka.join(ok, on="h", how="semi")
    kb = kb.join(ok, on="h", how="semi").select("h", pl.col("idx").alias("b_idx"))
    print(f"  keys built {time.time() - t0:.0f}s  A-keys {ka.height:,}  B-keys {kb.height:,}", flush=True)
    outs = []
    a_ids = a["idx"].to_numpy()
    for s in range(0, len(a_ids), chunk):
        lo, hi = a_ids[s], a_ids[min(s + chunk, len(a_ids)) - 1]
        sub = ka.filter(pl.col("idx").is_between(lo, hi)).rename({"idx": "a_idx"})
        pairs = sub.join(kb, on="h").group_by("a_idx", "b_idx").agg(
            *[(pl.col("kt") == kt).sum().cast(pl.UInt8).alias(f"k_{kt}") for kt in KEY_TYPES])
        pairs = cheap_score(pairs, rec)
        # per-S1 pre-cut (generous) so the global per-record ranking fits in memory
        pairs = pairs.filter(pl.col("cheap").rank("ordinal", descending=True).over("a_idx") <= PRE_CUT)
        outs.append(pairs)
    res = pl.concat(outs)
    print(f"  pairs {res.height:,} in {time.time() - t0:.0f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# cheap ranking + pruning (numeric only: padded token-id arrays, no string conversion)
# ---------------------------------------------------------------------------
K_A = 15
K_B = 3
PRE_CUT = 150


def _wov(a_ids: np.ndarray, b_ids: np.ndarray, idf: np.ndarray):
    va, vb = a_ids >= 0, b_ids >= 0
    wa = np.where(va, idf[np.maximum(a_ids, 0)], 0.0)
    wb = np.where(vb, idf[np.maximum(b_ids, 0)], 0.0)
    ma = ((a_ids[:, :, None] == b_ids[:, None, :]) & va[:, :, None]).any(2)
    inter = (wa * ma).sum(1)
    sa, sb = wa.sum(1), wb.sum(1)
    ov = np.divide(inter, np.minimum(sa, sb), out=np.zeros_like(inter), where=np.minimum(sa, sb) > 0)
    jac = np.divide(inter, sa + sb - inter, out=np.zeros_like(inter), where=(sa + sb - inter) > 0)
    return ov, jac


def cheap_score(pairs: pl.DataFrame, rec: dict) -> pl.DataFrame:
    """Numeric name + address similarity used only to rank candidates within blocks."""
    ai, bi = pairs["a_idx"].to_numpy(), pairs["b_idx"].to_numpy()
    n_ov, n_jac = _wov(rec["A_nt"][ai], rec["B_nt"][bi], rec["nt_idf"])
    name = np.maximum(0.5 * n_ov + 0.5 * n_jac, (rec["A_ch"][ai] == rec["B_ch"][bi]).astype(np.float64))
    _, a_jac = _wov(rec["A_at"][ai], rec["B_at"][bi], rec["at_idf"])
    na, nb = rec["A_num"][ai], rec["B_num"][bi]
    num = ((na[:, :, None] == nb[:, None, :]) & (na != 0)[:, :, None]).any(2).any(1)
    addr = np.where(rec["B_aempty"][bi], 0.3, a_jac + 0.2 * num)
    hits = pairs.select(pl.sum_horizontal(pl.col("^k_.*$").cast(pl.Float32))).to_series().to_numpy()
    score = name + 0.8 * addr + 0.02 * np.minimum(hits, 5)
    return pairs.with_columns(pl.Series("cheap", score.astype(np.float32)))


def rank_and_prune(pairs: pl.DataFrame, k_a: int = K_A, k_b: int = K_B) -> pl.DataFrame:
    pairs = pairs.with_columns(
        pl.col("cheap").rank("ordinal", descending=True).over("a_idx").cast(pl.UInt16).alias("rank_a"),
        pl.col("cheap").rank("ordinal", descending=True).over("b_idx").cast(pl.UInt16).alias("rank_b"))
    return pairs.filter((pl.col("rank_a") <= k_a) | (pl.col("rank_b") <= k_b))
