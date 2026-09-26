"""Learned blocking ranker.

A small LightGBM scores every raw key-matched pair from cheap numeric evidence (key hits
per key type, IDF-weighted name / address / component overlap, number agreement, empty
address). It replaces the hand-weighted cheap score for pruning to top-K per S1 and
top-k per S2/S3 record, and its score/ranks feed the pair model as context features.
Trained on un-pruned pairs of a 10% sample of TRAIN S1 records.
"""
from __future__ import annotations

import os
import time

import lightgbm as lgb
import numpy as np
import polars as pl

from blocking import KEY_TYPES, _wov, block_country, cheap_score
from features import digit_edit1
from io_utils import WORK

MODEL = os.path.join(WORK, "blocker.txt")
FEATS = [f"k_{k}" for k in KEY_TYPES] + [
    "n_ov", "n_jac", "c_eq", "a_ov", "a_jac", "ac_ov", "ac_jac", "num_any", "num_first_eq",
    "num_first_edit1", "b_aempty", "a_aempty", "na_tok", "nb_tok", "src_b", "cf_a", "cf_b", "hand"]
_booster = None


def raw_features(pairs: pl.DataFrame, rec: dict) -> np.ndarray:
    ai, bi = pairs["a_idx"].to_numpy(), pairs["b_idx"].to_numpy()
    f = {}
    for k in KEY_TYPES:
        f[f"k_{k}"] = pairs[f"k_{k}"].to_numpy().astype(np.float32)
    f["n_ov"], f["n_jac"] = _wov(rec["A_nt"][ai], rec["B_nt"][bi], rec["nt_idf"])
    f["c_eq"] = (rec["A_ch"][ai] == rec["B_ch"][bi]).astype(np.float32)
    f["a_ov"], f["a_jac"] = _wov(rec["A_at"][ai], rec["B_at"][bi], rec["at_idf"])
    f["ac_ov"], f["ac_jac"] = _wov(rec["A_ac"][ai], rec["B_ac"][bi], rec["ac_idf"])
    na, nb = rec["A_num"][ai], rec["B_num"][bi]
    f["num_any"] = ((na[:, :, None] == nb[:, None, :]) & (na != 0)[:, :, None]).any(2).any(1).astype(np.float32)
    f["num_first_eq"] = ((na[:, 0] == nb[:, 0]) & (na[:, 0] != 0)).astype(np.float32)
    f["num_first_edit1"] = digit_edit1(rec["A_numv"][ai, 0], rec["B_numv"][bi, 0]).astype(np.float32)
    f["b_aempty"] = rec["B_aempty"][bi].astype(np.float32)
    f["a_aempty"] = rec["A_aempty"][ai].astype(np.float32)
    f["na_tok"] = (rec["A_nt"][ai] >= 0).sum(1).astype(np.float32)
    f["nb_tok"] = (rec["B_nt"][bi] >= 0).sum(1).astype(np.float32)
    f["src_b"] = rec["B_src"][bi].astype(np.float32)
    f["cf_a"] = np.log1p(rec["A_corefreq"][ai]).astype(np.float32)
    f["cf_b"] = np.log1p(rec["B_corefreq"][bi]).astype(np.float32)
    f["hand"] = cheap_score(pairs, rec)["cheap"].to_numpy()
    return np.column_stack([np.asarray(f[k], dtype=np.float32) for k in FEATS])


def learned_score(pairs: pl.DataFrame, rec: dict) -> pl.DataFrame:
    global _booster
    if _booster is None:
        _booster = lgb.Booster(model_file=MODEL)
    s = _booster.predict(raw_features(pairs, rec), raw_score=True).astype(np.float32)
    return pairs.with_columns(pl.Series("cheap", s))


def available() -> bool:
    return os.path.exists(MODEL)


def train(A: pl.DataFrame, B: pl.DataFrame, rec: dict, truth: pl.DataFrame, sample_mod: int = 10) -> None:
    """Collect un-pruned pairs for 1/sample_mod of S1, fit the ranker, report recall@K."""
    t0 = time.time()
    tp = truth.select("a_idx", "b_idx").with_columns(pl.lit(1, pl.Int8).alias("y"))
    Xs, ys, gs = [], [], []
    for c in A["country"].unique(maintain_order=True).to_list():
        raw = block_country(A.filter(pl.col("country") == c), B.filter(pl.col("country") == c), rec,
                            sample_mod=sample_mod, keep_raw=True)
        raw = raw.join(tp, on=["a_idx", "b_idx"], how="left").with_columns(pl.col("y").fill_null(0))
        n_true = truth.filter((pl.col("country") == c) & (pl.col("a_idx") % sample_mod == 0)).height
        print(f"[blocker] {c}: raw pairs {raw.height:,}, key-union recall {raw['y'].sum() / n_true:.5f}", flush=True)
        for s in range(0, raw.height, 5_000_000):
            ch = raw.slice(s, 5_000_000)
            Xs.append(raw_features(ch, rec)); ys.append(ch["y"].to_numpy()); gs.append(ch["a_idx"].to_numpy())
    X, y, g = np.vstack(Xs), np.concatenate(ys), np.concatenate(gs)
    del Xs, ys, gs
    va = (g // sample_mod) % 5 == 0
    params = dict(objective="binary", learning_rate=0.1, num_leaves=63, min_data_in_leaf=500,
                  feature_fraction=0.9, bagging_fraction=0.5, bagging_freq=1, max_bin=63, num_threads=10, verbose=-1)
    dtr = lgb.Dataset(X[~va], y[~va], feature_name=FEATS)
    dva = lgb.Dataset(X[va], y[va], reference=dtr)
    m = lgb.train(params, dtr, 1000, valid_sets=[dva], callbacks=[lgb.early_stopping(30, verbose=False)])
    m.save_model(MODEL, num_iteration=m.best_iteration)
    global _booster
    _booster = None
    # recall@K on the held-out S1 groups: hand-weighted vs learned
    hv = pl.DataFrame({"a": g[va], "y": y[va], "hand": X[va, FEATS.index("hand")],
                       "lrn": m.predict(X[va], num_iteration=m.best_iteration, raw_score=True)})
    for col in ("hand", "lrn"):
        r = hv.with_columns(pl.col(col).rank("ordinal", descending=True).over("a").alias("r"))
        tot = r["y"].sum()
        print(f"[blocker] {col}: " + " ".join(f"K={k}:{r.filter(pl.col('r') <= k)['y'].sum() / tot:.5f}"
                                              for k in (5, 10, 15, 20, 25, 30, 50)), flush=True)
    print(f"[blocker] trained in {time.time() - t0:.0f}s, best_iter {m.best_iteration}", flush=True)
