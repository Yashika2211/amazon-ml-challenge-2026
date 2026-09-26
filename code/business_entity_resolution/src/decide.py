"""Decision layer: one-to-one assignment + per-S1 thresholds, and the vectorized scorer."""
from __future__ import annotations

import itertools

import numpy as np
import polars as pl


def one_to_one(pairs: pl.DataFrame) -> pl.DataFrame:
    """Each S2/S3 record keeps only its highest-probability S1 (ties: first)."""
    best = pairs.with_columns(pl.col("p").rank("ordinal", descending=True).over("b_idx").alias("_r"))
    return best.filter(pl.col("_r") == 1).drop("_r")


def apply_thresholds(pairs: pl.DataFrame, t_abs: float, r: float, t_empty: float) -> pl.DataFrame:
    p = pairs.with_columns(pl.col("p").max().over("a_idx").alias("_pmax"))
    return p.filter((pl.col("p") >= t_abs) & (pl.col("p") >= r * pl.col("_pmax")) & (pl.col("_pmax") >= t_empty)).drop("_pmax")


def expected_f05_select(pairs: pl.DataFrame, miss_rate: float = 0.0, t_empty: float = 0.0) -> pl.DataFrame:
    """Per S1 choose the probability-sorted prefix maximizing expected F0.5.

    E[F(prefix k)] ~ 1.25 * sum_top_k(p) / (0.25 * E[n_true] + k); the empty set scores
    P(no match) = prod(1 - p_i). E[n_true] includes the blocking miss rate.
    """
    d = pairs.sort(["a_idx", "p"], descending=[False, True]).with_columns(
        pl.col("p").cum_sum().over("a_idx").alias("_cs"),
        pl.int_range(1, pl.len() + 1).over("a_idx").alias("_k"),
        pl.col("p").sum().over("a_idx").alias("_sum"),
        (1 - pl.col("p")).log().sum().over("a_idx").exp().alias("_p0"),
    )
    d = d.with_columns((pl.col("_sum") / (1 - miss_rate)).alias("_nt"))
    d = d.with_columns((1.25 * pl.col("_cs") / (0.25 * pl.col("_nt") + pl.col("_k"))).alias("_ef"))
    best = d.group_by("a_idx").agg(pl.col("_ef").max().alias("_best"), pl.col("_p0").first(),
                                   pl.col("_k").sort_by("_ef", descending=True).first().alias("_kbest"))
    d = d.join(best, on="a_idx")
    d = d.with_columns(pl.col("p").max().over("a_idx").alias("_pmax"))
    return d.filter((pl.col("_best") > pl.col("_p0")) & (pl.col("_k") <= pl.col("_kbest"))
                    & (pl.col("_pmax") >= t_empty)).select(pairs.columns)


def macro_f05(pred: pl.DataFrame, truth_counts: pl.DataFrame, all_a: pl.DataFrame) -> float:
    """pred: (a_idx, y) of predicted pairs; truth_counts: (a_idx, ntrue); all_a: (a_idx) evaluated."""
    agg = pred.group_by("a_idx").agg(pl.len().alias("npred"), pl.col("y").sum().alias("tp"))
    d = all_a.join(truth_counts, on="a_idx", how="left").join(agg, on="a_idx", how="left").fill_null(0)
    f = (pl.when(pl.col("ntrue") == 0).then((pl.col("npred") == 0).cast(pl.Float64))
         .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("ntrue") + pl.col("npred"))))
    return d.select(f.mean()).item()


def per_a_scores(pred: pl.DataFrame, truth_counts: pl.DataFrame, all_a: pl.DataFrame) -> pl.DataFrame:
    agg = pred.group_by("a_idx").agg(pl.len().alias("npred"), pl.col("y").sum().alias("tp"))
    d = all_a.join(truth_counts, on="a_idx", how="left").join(agg, on="a_idx", how="left").fill_null(0)
    return d.with_columns(
        pl.when(pl.col("ntrue") == 0).then((pl.col("npred") == 0).cast(pl.Float64))
        .otherwise(1.25 * pl.col("tp") / (0.25 * pl.col("ntrue") + pl.col("npred"))).alias("f05"))


def grid_search(pairs: pl.DataFrame, truth_counts: pl.DataFrame, all_a: pl.DataFrame,
                t_abs_grid=None, r_grid=None, t_empty_grid=None) -> tuple:
    t_abs_grid = t_abs_grid or [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
    r_grid = r_grid or [0.0, 0.3, 0.5, 0.6, 0.7, 0.8]
    t_empty_grid = t_empty_grid or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    base = one_to_one(pairs).filter(pl.col("p") >= min(t_abs_grid))
    best = (-1, None)
    for t_abs, r, t_e in itertools.product(t_abs_grid, r_grid, t_empty_grid):
        if t_e < t_abs:
            continue
        s = macro_f05(apply_thresholds(base, t_abs, r, t_e), truth_counts, all_a)
        if s > best[0]:
            best = (s, (t_abs, r, t_e))
    return best
