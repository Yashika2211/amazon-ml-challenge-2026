"""Label-free prior-shift correction for test density.

Test contains many more near-duplicate decoys (same name, slightly different house number)
per S1 than train, while exact copies grow only slightly. For each pair bucket b we compare
its frequency per S1 on test (per country) with train: r_b = (test_b / S1) / (train_b / S1).
Genuine copies scale like the exact-match bucket (r_ref); the excess is decoys, so the match
odds in bucket b are multiplied by f_b = clip(r_ref / r_b, F_MIN, 1). Never increases odds.
(A leaderboard probe that raised these odds lost 0.003, confirming the direction.)
"""
from __future__ import annotations

import glob
import os

import numpy as np
import polars as pl

from io_utils import WORK
from prepare import load

F_MIN = 0.2
COLS = ["a_idx", "b_idx", "n_tset", "nt_wov", "num_na", "num_nb", "num_first_logdiff", "num_first_edit1", "aempty_b"]


def bucket_expr() -> pl.Expr:
    name = (pl.when(pl.col("n_tset") >= 0.95).then(pl.lit("same")).when(pl.col("nt_wov") >= 0.5)
              .then(pl.lit("partial")).otherwise(pl.lit("diff")))
    addr = (pl.when(pl.col("aempty_b") == 1).then(pl.lit("empty"))
              .when((pl.col("num_na") == 0) | (pl.col("num_nb") == 0)).then(pl.lit("nonum"))
              .when(pl.col("num_first_logdiff") == 0).then(pl.lit("eq"))
              .when(pl.col("num_first_edit1") == 1).then(pl.lit("edit1"))
              .when(pl.col("num_first_logdiff") <= float(np.log1p(20))).then(pl.lit("shift20"))
              .when(pl.col("num_first_logdiff") <= float(np.log1p(200))).then(pl.lit("shift200"))
              .otherwise(pl.lit("far")))
    return pl.concat_str([name, addr], separator="|").alias("bk")


def _counts(split: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    f = pl.concat([pl.read_parquet(x, columns=COLS) for x in sorted(glob.glob(os.path.join(WORK, f"feat_{split}_*.parquet")))])
    A = load(split, "A").select(pl.col("idx").alias("a_idx"), "country")
    f = f.join(A, on="a_idx").with_columns(bucket_expr())
    n = A.group_by("country").len().rename({"len": "n_s1"})
    c = f.group_by("country", "bk").len().join(n, on="country").with_columns((pl.col("len") / pl.col("n_s1")).alias("rate"))
    return f.select("a_idx", "b_idx", "country", "bk"), c


def factors() -> pl.DataFrame:
    _, ctr = _counts("train")
    te_pairs, cte = _counts("test")
    pooled = ctr.group_by("bk").agg((pl.col("len").sum() / ctr.select("country", "n_s1").unique()["n_s1"].sum()).alias("rate_tr"))
    tr_c = ctr.select("country", "bk", pl.col("rate").alias("rate_tr_c"))
    d = cte.select("country", "bk", pl.col("rate").alias("rate_te")).join(tr_c, on=["country", "bk"], how="left").join(pooled, on="bk", how="left")
    d = d.with_columns(pl.coalesce("rate_tr_c", "rate_tr").alias("rate_train")).with_columns((pl.col("rate_te") / pl.col("rate_train")).alias("r"))
    ref = d.filter(pl.col("bk") == "same|eq").select("country", pl.col("r").alias("r_ref"))
    d = d.join(ref, on="country").with_columns((pl.col("r_ref") / pl.col("r")).clip(F_MIN, 1.0).alias("f"))
    return d.select("country", "bk", "rate_train", "rate_te", "r", "r_ref", "f"), te_pairs


def apply(pred: pl.DataFrame) -> pl.DataFrame:
    """pred: test (a_idx, b_idx, p) -> same with odds-corrected p."""
    fac, te_pairs = factors()
    print("[shift] factors:\n", fac.filter(pl.col("f") < 0.999).sort("country", "f"), flush=True)
    d = pred.join(te_pairs, on=["a_idx", "b_idx"], how="left").join(fac.select("country", "bk", "f"), on=["country", "bk"], how="left")
    d = d.with_columns(pl.col("f").fill_null(1.0))
    d = d.with_columns((pl.col("p") * pl.col("f") / (pl.col("p") * pl.col("f") + 1 - pl.col("p"))).cast(pl.Float32).alias("p"))
    return d.select(pred.columns)
