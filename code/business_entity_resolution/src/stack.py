"""Round-2 cluster features built from round-1 probabilities.

For a pair (a, b):
  p1, rank of p1 within a and within b, best competing probability on both sides,
  probability mass of a, number of confident candidates of a, and the similarity of b to
  a's best *other* candidates (S2/S3 copies of the same entity reinforce each other).

Competitor context (branch disambiguation): many S1 share a name (branches); an S2/S3 copy
without an address is then ambiguous. The per-source copy-count prior (few S1 have zero
copies in a source) favours the branch that has no other confident copy from that source,
so we expose, for this S1 and for the strongest competing S1 of the record, the number of
confident copies from the record's source and in total (excluding the record itself).
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


def competitor_features(d: pl.DataFrame, src: np.ndarray) -> pl.DataFrame:
    """d: (a_idx, b_idx, p1). Adds own/competitor per-source confident-copy counts."""
    d = d.with_columns(pl.Series("s3", (src[d["b_idx"].to_numpy()] == 3)))
    conf = (pl.col("p1") > 0.5)
    stats = d.group_by("a_idx").agg(
        (conf & ~pl.col("s3")).sum().alias("c2"), (conf & pl.col("s3")).sum().alias("c3"),
        pl.col("p1").filter(~pl.col("s3")).sum().alias("m2"), pl.col("p1").filter(pl.col("s3")).sum().alias("m3"))
    d = d.join(stats, on="a_idx", how="left")
    me = conf.cast(pl.Int32)
    d = d.with_columns(
        (pl.when(pl.col("s3")).then(pl.col("c3")).otherwise(pl.col("c2")) - me).alias("own_same_src"),
        (pl.col("c2") + pl.col("c3") - me).alias("own_total"),
        (pl.when(pl.col("s3")).then(pl.col("m3")).otherwise(pl.col("m2")) - pl.col("p1")).alias("own_mass_same_src"),
    )
    # strongest competing S1 for the same record
    top2 = (d.sort(["b_idx", "p1"], descending=[False, True]).group_by("b_idx", maintain_order=True)
              .agg(pl.col("a_idx").head(2).alias("ta"), pl.col("p1").head(2).alias("tp")))
    d = d.join(top2, on="b_idx", how="left").with_columns(
        pl.when(pl.col("ta").list.get(0) == pl.col("a_idx")).then(pl.col("ta").list.get(1, null_on_oob=True))
          .otherwise(pl.col("ta").list.get(0)).alias("comp"),
        pl.when(pl.col("ta").list.get(0) == pl.col("a_idx")).then(pl.col("tp").list.get(1, null_on_oob=True))
          .otherwise(pl.col("tp").list.get(0)).alias("comp_p")).drop("ta", "tp")
    cs = stats.rename({"a_idx": "comp", "c2": "k2", "c3": "k3", "m2": "q2", "m3": "q3"})
    d = d.join(cs, on="comp", how="left")
    cme = (pl.col("comp_p").fill_null(0) > 0.5).cast(pl.Int32)
    d = d.with_columns(
        (pl.when(pl.col("s3")).then(pl.col("k3")).otherwise(pl.col("k2")) - cme).fill_null(-1).alias("comp_same_src"),
        (pl.col("k2") + pl.col("k3") - cme).fill_null(-1).alias("comp_total"),
        (pl.when(pl.col("s3")).then(pl.col("q3")).otherwise(pl.col("q2")) - pl.col("comp_p")).fill_null(-1).alias("comp_mass_same_src"),
        pl.col("comp").is_null().alias("no_comp"),
    )
    d = d.with_columns(
        (pl.col("own_same_src") - pl.col("comp_same_src")).alias("same_src_diff"),
        (pl.col("own_total") - pl.col("comp_total")).alias("total_diff"),
        # how many S1 find this record plausible
        (pl.col("p1") > 0.2).sum().over("b_idx").alias("n_amb_b"),
        pl.col("p1").sum().over("b_idx").alias("psum_b"),
    )
    return d.drop("s3", "c2", "c3", "m2", "m3", "k2", "k3", "q2", "q3", "comp", "comp_p")


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
    B = load(split, "B").select("core", "addr", "src")
    d = competitor_features(d, B["src"].to_numpy())
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
