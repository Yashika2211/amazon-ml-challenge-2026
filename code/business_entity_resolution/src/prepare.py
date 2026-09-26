"""Normalize all sources with the mined maps and cache them as parquet.

A = Source 1 (reference), B = Source 2 + Source 3 stacked (src column = 2 or 3).
"""
from __future__ import annotations

import os
import sys
import time

import polars as pl

from io_utils import WORK, load_source
from mine import load_maps
from normalize import normalize


def prepare(split: str, maps: dict) -> None:
    t = time.time()
    a = normalize(load_source(split, 1), maps).with_columns(pl.lit(1, pl.Int8).alias("src"))
    b = pl.concat([normalize(load_source(split, s), maps).with_columns(pl.lit(s, pl.Int8).alias("src"))
                   for s in (2, 3)])
    for name, df in (("A", a), ("B", b)):
        df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.Int32))
        df.write_parquet(os.path.join(WORK, f"{split}_{name}.parquet"))
    print(f"prepared {split}: A={a.height} B={b.height} in {time.time() - t:.0f}s", flush=True)


def load(split: str, name: str) -> pl.DataFrame:
    return pl.read_parquet(os.path.join(WORK, f"{split}_{name}.parquet"))


if __name__ == "__main__":
    maps = load_maps()
    for split in (sys.argv[1:] or ["train", "test"]):
        prepare(split, maps)
