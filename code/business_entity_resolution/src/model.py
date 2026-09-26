"""LightGBM pair classifier with GroupKFold-by-S1 out-of-fold predictions."""
from __future__ import annotations

import glob
import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

from io_utils import WORK, load_truth
from prepare import load

N_FOLDS = 3
TRAIN_FRAC = 0.3  # fraction of S1 groups whose pairs are used for fitting (memory bound)
ID_COLS = ["a_idx", "b_idx"]
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=127, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=127, num_threads=10, verbose=-1)


def fold_of(a_idx: pl.Expr) -> pl.Expr:
    return (a_idx.cast(pl.UInt64).hash(seed=7) % N_FOLDS).cast(pl.Int8)


def in_train_sample(a_idx: pl.Expr) -> pl.Expr:
    return (a_idx.cast(pl.UInt64).hash(seed=11) % 1000) < int(TRAIN_FRAC * 1000)


def truth_pairs() -> pl.DataFrame:
    A, B = load("train", "A").select("entity_id", "idx", "country"), load("train", "B").select("entity_id", "idx")
    return (load_truth().join(A.rename({"entity_id": "s1_id", "idx": "a_idx"}), on="s1_id")
            .join(B.rename({"entity_id": "m_id", "idx": "b_idx"}), on="m_id")
            .select("a_idx", "b_idx", "country"))


def feat_files(split: str) -> list:
    return sorted(glob.glob(os.path.join(WORK, f"feat_{split}_*.parquet")))


def feature_names(split: str = "train", extra=None) -> list:
    cols = list(pl.read_parquet_schema(feat_files(split)[0]))
    if extra is not None:
        cols += list(pl.read_parquet_schema(extra)) if isinstance(extra, str) else list(extra.columns)
    return [c for c in dict.fromkeys(cols) if c not in ID_COLS]


def _read(f: str, extra, filt=None) -> pl.DataFrame:
    """Read one feature chunk; `extra` is a DataFrame or a parquet path joined on (a_idx, b_idx)."""
    d = pl.read_parquet(f)
    if filt is not None:
        d = d.filter(filt)
    if extra is not None:
        if isinstance(extra, str):
            keys = d.select(ID_COLS).lazy()
            ex = pl.scan_parquet(extra).join(keys, on=ID_COLS, how="semi").collect()
        else:
            ex = extra
        d = d.join(ex, on=ID_COLS, how="left")
    return d


def train_folds(tag: str = "m1", extra=None) -> list:
    """Bin the sampled training rows once (lgb.Dataset) and train each fold on a subset."""
    truth = truth_pairs()
    tp = truth.select(ID_COLS).with_columns(pl.lit(1, pl.Int8).alias("y"))
    feats = feature_names("train", extra)
    files = feat_files("train")
    sizes = [pl.scan_parquet(f).filter(in_train_sample(pl.col("a_idx"))).select(pl.len()).collect().item() for f in files]
    n = sum(sizes)
    X = np.empty((n, len(feats)), dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    fold = np.empty(n, dtype=np.int8)
    o = 0
    for f, m in zip(files, sizes):
        d = _read(f, extra, in_train_sample(pl.col("a_idx")))
        d = d.join(tp, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0), fold_of(pl.col("a_idx")).alias("fold"))
        X[o:o + m] = d.select(feats).to_numpy().astype(np.float32, copy=False)
        y[o:o + m] = d["y"].to_numpy()
        fold[o:o + m] = d["fold"].to_numpy()
        o += m
        del d
    print(f"[train] rows {n:,} pos {int(y.sum()):,} feats {len(feats)}", flush=True)
    full = lgb.Dataset(X, y, feature_name=feats, free_raw_data=True, params={"max_bin": PARAMS["max_bin"], "verbose": -1})
    full.construct()
    del X
    models = []
    for k in range(N_FOLDS):
        t = time.time()
        dtr = full.subset(np.where(fold != k)[0])
        dva = full.subset(np.where(fold == k)[0])
        m = lgb.train(PARAMS, dtr, 4000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])
        print(f"[train] fold {k}: best_iter {m.best_iteration} logloss {m.best_score['valid_0']['binary_logloss']:.5f} "
              f"{time.time() - t:.0f}s", flush=True)
        models.append(m)
    with open(os.path.join(WORK, f"{tag}_models.pkl"), "wb") as f:
        pickle.dump(models, f)
    imp = sorted(zip(feats, models[0].feature_importance("gain")), key=lambda x: -x[1])
    print("[train] top features:", [(n_, int(g)) for n_, g in imp[:25]], flush=True)
    return models


def predict_oof(models: list, tag: str = "m1", extra=None) -> pl.DataFrame:
    feats = feature_names("train", extra)
    out = []
    for f in feat_files("train"):
        d = _read(f, extra).with_columns(fold_of(pl.col("a_idx")).alias("fold"))
        X = d.select(feats).to_numpy()
        fold = d["fold"].to_numpy()
        p = np.zeros(d.height, np.float32)
        for k, m in enumerate(models):
            sel = fold == k
            if sel.any():
                p[sel] = m.predict(X[sel], num_iteration=m.best_iteration)
        out.append(d.select(ID_COLS).with_columns(pl.Series("p_raw", p)))
    oof = pl.concat(out)
    tp = truth_pairs().select(ID_COLS).with_columns(pl.lit(1, pl.Int8).alias("y"))
    oof = oof.join(tp, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(oof["p_raw"].to_numpy(), oof["y"].to_numpy())
    oof = oof.with_columns(pl.Series("p", iso.predict(oof["p_raw"].to_numpy()).astype(np.float32)))
    with open(os.path.join(WORK, f"{tag}_iso.pkl"), "wb") as f:
        pickle.dump(iso, f)
    oof.write_parquet(os.path.join(WORK, f"{tag}_oof.parquet"))
    return oof


def predict_test(models: list, tag: str = "m1", extra=None) -> pl.DataFrame:
    feats = feature_names("train", extra)
    with open(os.path.join(WORK, f"{tag}_iso.pkl"), "rb") as f:
        iso = pickle.load(f)
    out = []
    for f in feat_files("test"):
        d = _read(f, extra)
        X = d.select(feats).to_numpy()
        p = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
        out.append(d.select(ID_COLS).with_columns(pl.Series("p_raw", p.astype(np.float32))))
    pred = pl.concat(out)
    pred = pred.with_columns(pl.Series("p", iso.predict(pred["p_raw"].to_numpy()).astype(np.float32)))
    pred.write_parquet(os.path.join(WORK, f"{tag}_test.parquet"))
    return pred
