"""Reproducible OOF error and feature audit for entity-resolution experiments.

Usage:
  python src/audit.py --tag m2 --extra work/cluster_train_m1.parquet

Writes compact CSV reports under ``work/audit_<tag>/``.  The analysis is based
only on out-of-fold predictions and the supplied training labels.  In
particular, it never uses test labels or a prediction made by a model trained
on the same S1 group.
"""
from __future__ import annotations

import argparse
import csv
import os
import pickle

import numpy as np
import polars as pl

from decide import expected_f05_select, one_to_one, per_a_scores
from evaluate import truth_frames
from io_utils import WORK
from model import feature_names
from prepare import load


def _decision(tag: str) -> tuple[float, float]:
    """Read the saved decision rule and return its expected-F parameters."""
    import json
    with open(os.path.join(WORK, f"{tag}_decision.json")) as f:
        cfg = json.load(f)
    return tuple(cfg.get("ef_params", (0.0, 0.5)))


def _feature_stats(tag: str, extra: str | None, out_dir: str) -> None:
    """Pair-label Pearson correlation and mean gain across fold models.

    Pearson is intentionally reported alongside gain, not instead of it: most
    useful pair signals are non-monotonic (for example near-but-not-equal house
    numbers), so correlation alone can make a valuable feature look inert.
    Statistics are exact over all candidate pairs in the OOF file.
    """
    p = os.path.join(WORK, f"{tag}_oof.parquet")
    feats = feature_names("train", extra)
    models_path = os.path.join(WORK, f"{tag}_models.pkl")
    with open(models_path, "rb") as f:
        models = pickle.load(f)
    gains = np.mean([m.feature_importance("gain") for m in models], axis=0)
    splits = np.mean([m.feature_importance("split") for m in models], axis=0)
    # OOF stores y, and feature files retain the remaining columns.  Build the
    # sufficient statistics chunkwise to keep the audit below the model RAM.
    files = sorted([os.path.join(WORK, x) for x in os.listdir(WORK)
                    if x.startswith("feat_train_") and x.endswith(".parquet")])
    n = 0
    sy = sy2 = 0.0
    sx = np.zeros(len(feats), dtype=np.float64)
    sx2 = np.zeros(len(feats), dtype=np.float64)
    sxy = np.zeros(len(feats), dtype=np.float64)
    # Extra features have one row per pair and are joined only when necessary.
    extra_lazy = pl.scan_parquet(extra) if extra else None
    truth = pl.read_parquet(p, columns=["a_idx", "b_idx", "y"])
    for i, path in enumerate(files, 1):
        d = pl.read_parquet(path, columns=["a_idx", "b_idx"] + [x for x in feats if x not in {"a_idx", "b_idx"} and x in pl.read_parquet_schema(path)])
        if extra_lazy is not None:
            ex_cols = [x for x in feats if x not in d.columns]
            if ex_cols:
                ex = (extra_lazy.join(d.select("a_idx", "b_idx").lazy(), on=["a_idx", "b_idx"], how="semi")
                      .select(["a_idx", "b_idx"] + ex_cols).collect())
                d = d.join(ex, on=["a_idx", "b_idx"], how="left")
        d = d.join(truth, on=["a_idx", "b_idx"], how="left")
        y = d["y"].to_numpy().astype(np.float64, copy=False)
        x = d.select(feats).fill_null(0).to_numpy().astype(np.float64, copy=False)
        n += len(d); sy += y.sum(); sy2 += np.dot(y, y)
        sx += x.sum(axis=0); sx2 += np.einsum("ij,ij->j", x, x); sxy += x.T @ y
        print(f"[audit] correlation chunks {i}/{len(files)}", flush=True)
    cov = n * sxy - sx * sy
    den = np.sqrt(np.maximum(n * sx2 - sx * sx, 0) * max(n * sy2 - sy * sy, 0))
    corr = np.divide(cov, den, out=np.zeros_like(cov), where=den > 0)
    rows = sorted(zip(feats, corr, gains, splits), key=lambda z: -z[2])
    with open(os.path.join(out_dir, "feature_audit.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "pearson_corr_y", "mean_gain", "mean_splits"])
        w.writerows(rows)


def run(tag: str, extra: str | None) -> str:
    out_dir = os.path.join(WORK, f"audit_{tag}")
    os.makedirs(out_dir, exist_ok=True)
    oof = pl.read_parquet(os.path.join(WORK, f"{tag}_oof.parquet"))
    tp, counts, all_a = truth_frames()
    pred = expected_f05_select(one_to_one(oof).filter(pl.col("p") >= 0.02), *_decision(tag))
    scores = per_a_scores(pred, counts, all_a)
    A = load("train", "A").select(pl.col("idx").alias("a_idx"), "entity_id", "country", "name_norm", "addr", "legal", "addr_empty")
    # A compact diagnostic segmentation: these fields describe either the S1
    # record or its candidate topology and can be reproduced for every fold.
    topo = (oof.group_by("a_idx").agg(
        pl.len().alias("n_cand"),
        (pl.col("p") >= 0.1).sum().alias("n_p_ge_10"),
        (pl.col("p") >= 0.5).sum().alias("n_p_ge_50"),
        pl.col("p").max().alias("p_max"),
        pl.col("p").sort(descending=True).slice(1, 1).first().fill_null(0).alias("p_2"),
    ).with_columns((pl.col("p_max") - pl.col("p_2")).alias("p_margin")))
    d = (scores.join(A, on="a_idx").join(topo, on="a_idx")
        .with_columns(
            pl.when(pl.col("ntrue") == 0).then(pl.lit("singleton"))
              .when(pl.col("ntrue") == 1).then(pl.lit("one_true"))
              .when(pl.col("ntrue") <= 3).then(pl.lit("2_3_true"))
              .otherwise(pl.lit("4plus_true")).alias("truth_size"),
            pl.when(pl.col("n_cand") <= 5).then(pl.lit("<=5"))
              .when(pl.col("n_cand") <= 15).then(pl.lit("6_15"))
              .when(pl.col("n_cand") <= 30).then(pl.lit("16_30"))
              .otherwise(pl.lit("31plus")).alias("candidate_band"),
            pl.when(pl.col("p_margin") < 0.05).then(pl.lit("<.05"))
              .when(pl.col("p_margin") < 0.20).then(pl.lit(".05_.20"))
              .otherwise(pl.lit(">=.20")).alias("margin_band"),
            pl.when(pl.col("addr_empty")).then(pl.lit("empty"))
              .otherwise(pl.lit("present")).alias("address_state"),
        ))
    for group in ("country", "truth_size", "candidate_band", "margin_band", "address_state",
                  ["country", "margin_band"], ["country", "candidate_band"]):
        label = "_x_".join(group) if isinstance(group, list) else group
        s = (d.group_by(group).agg(pl.len().alias("n_s1"), pl.mean("f05").alias("mean_f05"),
            (pl.col("f05") < 1).mean().alias("error_rate"), pl.mean("ntrue").alias("mean_ntrue"),
            pl.mean("npred").alias("mean_npred"), pl.mean("tp").alias("mean_tp"))
             .sort(["mean_f05", "n_s1"], descending=[False, True]))
        s.write_csv(os.path.join(out_dir, f"segments_{label}.csv"))
    # Inspectable worst S1 rows.  Include record text and counts but omit pairs
    # to keep it sharable and small; pair-level detail follows in error_pairs.
    worst = d.sort(["f05", "ntrue", "npred"], descending=[False, True, True]).head(500)
    worst.write_csv(os.path.join(out_dir, "worst_s1.csv"))
    bad_a = worst.select("a_idx")
    pair_detail = (oof.join(bad_a, on="a_idx", how="semi")
                   .join(load("train", "B").select(pl.col("idx").alias("b_idx"), pl.col("entity_id").alias("b_entity_id"),
                                                   pl.col("name_norm").alias("b_name"), pl.col("addr").alias("b_addr")), on="b_idx")
                   .join(tp.select("a_idx", "b_idx").with_columns(pl.lit(1).alias("is_true")), on=["a_idx", "b_idx"], how="left")
                   .with_columns(pl.col("is_true").fill_null(0))
                   .sort(["a_idx", "p"], descending=[False, True]))
    pair_detail.write_csv(os.path.join(out_dir, "worst_s1_pairs.csv"))
    _feature_stats(tag, extra, out_dir)
    print(f"[audit] wrote {out_dir}", flush=True)
    return out_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="m2")
    ap.add_argument("--extra", default=None, help="cluster feature parquet used by this model")
    args = ap.parse_args()
    run(args.tag, args.extra)
