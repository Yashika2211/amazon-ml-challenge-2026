"""Round-2 cluster features built from round-1 probabilities.

For a pair (a, b):
  p1, rank of p1 within a and within b, best competing probability on both sides,
  probability mass of a, number of confident candidates of a, and the similarity of b to
  a's best *other* candidates (S2/S3 copies of the same entity reinforce each other).
"""
from __future__ import annotations

import os

import numpy as np
import polars as pl
from rapidfuzz import fuzz, process

from io_utils import WORK
from prepare import load


def _cp(fn, x, y):
    return process.cpdist(x, y, scorer=fn, workers=-1, dtype=np.float32) / 100.0


def cluster_features(split: str, p1: pl.DataFrame) -> pl.DataFrame:
    d = p1.select("a_idx", "b_idx", pl.col("p").alias("p1"))
    d = d.with_columns(
        pl.col("p1").rank("ordinal", descending=True).over("a_idx").alias("r_a"),
        pl.col("p1").rank("ordinal", descending=True).over("b_idx").alias("r_b"),
    )
    top = (d.sort(["a_idx", "p1"], descending=[False, True]).group_by("a_idx", maintain_order=True)
             .agg(pl.col("b_idx").head(3).alias("tb"), pl.col("p1").head(3).alias("tp"),
                  pl.col("p1").sum().alias("psum_a"), (pl.col("p1") > 0.5).sum().alias("n50_a")))
    topb = (d.sort(["b_idx", "p1"], descending=[False, True]).group_by("b_idx", maintain_order=True)
              .agg(pl.col("p1").head(2).alias("tpb")))
    d = d.join(top, on="a_idx", how="left").join(topb, on="b_idx", how="left")
    # best other candidate of a (partner) and second-best other
    d = d.with_columns(
        pl.when(pl.col("r_a") == 1).then(pl.col("tp").list.get(1, null_on_oob=True)).otherwise(pl.col("tp").list.get(0)).fill_null(0).alias("pother_a"),
        pl.when(pl.col("r_b") == 1).then(pl.col("tpb").list.get(1, null_on_oob=True)).otherwise(pl.col("tpb").list.get(0)).fill_null(0).alias("pother_b"),
        pl.when(pl.col("r_a") == 1).then(pl.col("tb").list.get(1, null_on_oob=True)).otherwise(pl.col("tb").list.get(0)).alias("partner1"),
        pl.when(pl.col("r_a") <= 2).then(pl.col("tb").list.get(2, null_on_oob=True)).otherwise(pl.col("tb").list.get(1, null_on_oob=True)).alias("partner2"),
        pl.when(pl.col("r_a") <= 2).then(pl.col("tp").list.get(2, null_on_oob=True)).otherwise(pl.col("tp").list.get(1, null_on_oob=True)).fill_null(0).alias("ppartner2"),
    ).drop("tb", "tp", "tpb")
    B = load(split, "B").select("core", "addr")
    core, addr = B["core"], B["addr"]
    out = {}
    for k in ("partner1", "partner2"):
        pidx = d[k].fill_null(-1).to_numpy()
        has = pidx >= 0
        bi = d["b_idx"].to_numpy()
        n_sim = np.zeros(d.height, np.float32)
        a_sim = np.zeros(d.height, np.float32)
        if has.any():
            x = core.gather(bi[has]).to_list(); y = core.gather(pidx[has]).to_list()
            n_sim[has] = _cp(fuzz.token_set_ratio, x, y)
            x = addr.gather(bi[has]).to_list(); y = addr.gather(pidx[has]).to_list()
            a_sim[has] = _cp(fuzz.token_set_ratio, x, y)
        out[f"{k}_nsim"] = n_sim
        out[f"{k}_asim"] = a_sim
    d = d.with_columns([pl.Series(k, v) for k, v in out.items()])
    d = d.with_columns(
        (pl.col("p1") - pl.col("pother_b")).alias("margin_b"),
        (pl.col("p1") - pl.col("pother_a")).alias("margin_a"),
    ).drop("partner1", "partner2")
    return d.with_columns(pl.all().exclude("a_idx", "b_idx").cast(pl.Float32))


def build(split: str, tag: str) -> str:
    src = os.path.join(WORK, f"{tag}_oof.parquet" if split == "train" else f"{tag}_test.parquet")
    p1 = pl.read_parquet(src, columns=["a_idx", "b_idx", "p"])
    cf = cluster_features(split, p1)
    path = os.path.join(WORK, f"cluster_{split}.parquet")
    cf.write_parquet(path)
    return path
