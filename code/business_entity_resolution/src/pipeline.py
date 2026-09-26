"""Stages: candidates -> features, per split and country (country is only a partition key)."""
from __future__ import annotations

import glob
import os
import time

import numpy as np
import polars as pl

from blocking import block_country, rank_and_prune
from features import build_records, pair_features, string_table
from io_utils import WORK
from mine import load_maps
from prepare import load

CHUNK = 2_000_000


def countries(A: pl.DataFrame) -> list:
    return A["country"].unique(maintain_order=True).to_list()


def candidates(split: str, A: pl.DataFrame, B: pl.DataFrame, rec: dict) -> None:
    for c in countries(A):
        t = time.time()
        p = block_country(A.filter(pl.col("country") == c), B.filter(pl.col("country") == c), rec)
        p = rank_and_prune(p)
        p.write_parquet(os.path.join(WORK, f"cand_{split}_{c}.parquet"))
        print(f"[cand] {split} {c}: {p.height:,} pairs in {time.time() - t:.0f}s", flush=True)


def context(p: pl.DataFrame, B_src: np.ndarray) -> pl.DataFrame:
    return p.with_columns(
        pl.len().over("a_idx").cast(pl.Float32).alias("ncand_a"),
        pl.len().over("b_idx").cast(pl.Float32).alias("ncand_b"),
        (pl.col("cheap").max().over("a_idx") - pl.col("cheap")).alias("gap_a"),
        (pl.col("cheap").max().over("b_idx") - pl.col("cheap")).alias("gap_b"),
        (pl.col("cheap").sort(descending=True).slice(1, 1).first().over("a_idx").fill_null(0)).alias("cheap2_a"),
        pl.Series("src_b", B_src[p["b_idx"].to_numpy()].astype(np.float32)),
    )


def featurize(split: str, A: pl.DataFrame, B: pl.DataFrame, rec: dict) -> None:
    for f in glob.glob(os.path.join(WORK, f"feat_{split}_*.parquet")):
        os.remove(f)
    SA, SB = string_table(A), string_table(B)
    posA, posB = np.arange(A.height), np.arange(B.height)
    B_src = B["src"].to_numpy()
    for c in countries(A):
        t = time.time()
        p = context(pl.read_parquet(os.path.join(WORK, f"cand_{split}_{c}.parquet")), B_src)
        p = p.with_columns(pl.col("^k_.*$").cast(pl.Float32), pl.col("rank_a", "rank_b").cast(pl.Float32))
        for i, s in enumerate(range(0, p.height, CHUNK)):
            ch = p.slice(s, CHUNK)
            f = pair_features(ch, SA, SB, rec, posA, posB)
            f = pl.concat([ch, f.drop("a_idx", "b_idx")], how="horizontal")
            f.write_parquet(os.path.join(WORK, f"feat_{split}_{c}_{i:03d}.parquet"))
        print(f"[feat] {split} {c}: {p.height:,} pairs in {time.time() - t:.0f}s", flush=True)


def run_split(split: str, steps=("cand", "feat")) -> None:
    A, B = load(split, "A"), load(split, "B")
    rec = build_records(split, A, B, load_maps()["generic"])
    if "cand" in steps:
        candidates(split, A, B, rec)
        if split == "train":
            import evaluate
            print("[cand] blocking report:", evaluate.blocking_report("train"), flush=True)
    if "feat" in steps:
        featurize(split, A, B, rec)


if __name__ == "__main__":
    import sys
    run_split(sys.argv[1], tuple(sys.argv[2:]) or ("cand", "feat"))
