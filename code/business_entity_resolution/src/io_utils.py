"""Loading helpers. All files are TSV without quoting."""
from __future__ import annotations

import os

import polars as pl

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DATA = os.environ.get("ER_DATA", os.path.join(ROOT, "student_resource", "dataset"))
WORK = os.environ.get("ER_WORK", os.path.join(ROOT, "work"))
OUT = os.environ.get("ER_OUT", os.path.join(ROOT, "output"))
os.makedirs(WORK, exist_ok=True)
os.makedirs(OUT, exist_ok=True)


def read_tsv(path: str) -> pl.DataFrame:
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False,
                       encoding="utf8-lossy", missing_utf8_is_empty_string=True)


def source_path(split: str, src: int) -> str:
    return os.path.join(DATA, split, f"{split}_source{src}.tsv")


def load_source(split: str, src: int) -> pl.DataFrame:
    df = read_tsv(source_path(split, src))
    return df.with_columns(pl.col("business_name").fill_null(""), pl.col("business_address").fill_null(""),
                           pl.col("country").fill_null(""))


def load_truth() -> pl.DataFrame:
    """Long format (s1_id, m_id) plus the list of all S1 ids."""
    gt = read_tsv(os.path.join(DATA, "train", "train_ground_truth.tsv"))
    return (gt.with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",").alias("m"))
              .explode("m").filter(pl.col("m") != "")
              .select(pl.col("source1_entity_id").alias("s1_id"), pl.col("m").alias("m_id")))
