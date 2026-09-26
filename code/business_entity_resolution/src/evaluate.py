"""OOF evaluation, threshold tuning, EXPERIMENTS.md logging and submission writing."""
from __future__ import annotations

import datetime
import glob
import json
import os

import polars as pl

from decide import apply_thresholds, expected_f05_select, grid_search, one_to_one, per_a_scores
from io_utils import OUT, ROOT, WORK
from model import truth_pairs
from prepare import load

EXP = os.path.join(ROOT, "EXPERIMENTS.md")


def truth_frames():
    tp = truth_pairs()
    A = load("train", "A").select(pl.col("idx").alias("a_idx"), "country")
    counts = tp.group_by("a_idx").agg(pl.len().alias("ntrue"))
    all_a = A.join(counts, on="a_idx", how="left").with_columns(pl.col("ntrue").fill_null(0))
    return tp, counts, all_a


def blocking_report(split: str = "train") -> dict:
    tp, _, all_a = truth_frames()
    cand = pl.concat([pl.read_parquet(f, columns=["a_idx", "b_idx"]) for f in glob.glob(os.path.join(WORK, f"cand_{split}_*.parquet"))])
    hit = tp.join(cand, on=["a_idx", "b_idx"], how="semi")
    rep = {"pairs": cand.height, "pairs_per_s1": round(cand.height / all_a.height, 2),
           "pair_recall": round(hit.height / tp.height, 5)}
    for c in all_a["country"].unique().sort().to_list():
        t = tp.filter(pl.col("country") == c)
        rep[f"pair_recall_{c}"] = round(hit.filter(pl.col("country") == c).height / t.height, 5)
    na, nb = all_a.height, load(split, "B").height
    rep["reduction_ratio"] = round(1 - cand.height / (na * nb), 8)
    return rep


def score_report(pred: pl.DataFrame, counts: pl.DataFrame, all_a: pl.DataFrame) -> dict:
    s = per_a_scores(pred, counts, all_a)
    rep = {"macro_f05": round(s["f05"].mean(), 5)}
    for c in s["country"].unique().sort().to_list():
        rep[f"f05_{c}"] = round(s.filter(pl.col("country") == c)["f05"].mean(), 5)
    rep["f05_singleton"] = round(s.filter(pl.col("ntrue") == 0)["f05"].mean(), 5)
    rep["f05_nonsingleton"] = round(s.filter(pl.col("ntrue") > 0)["f05"].mean(), 5)
    tp_ = pred["y"].sum()
    rep["micro_precision"] = round(tp_ / max(pred.height, 1), 5)
    rep["micro_recall"] = round(tp_ / counts["ntrue"].sum(), 5)
    rep["pred_empty_rate"] = round(1 - pred["a_idx"].n_unique() / all_a.height, 4)
    return rep


def tune(oof: pl.DataFrame) -> dict:
    _, counts, all_a = truth_frames()
    best, params = grid_search(oof, counts, all_a)
    t_abs, r, t_e = params
    # local refinement
    best2, params2 = grid_search(oof, counts, all_a,
                                 [round(t_abs + d, 3) for d in (-0.05, -0.025, 0, 0.025, 0.05)],
                                 sorted({max(0.0, r + d) for d in (-0.1, 0, 0.1)}),
                                 [round(t_e + d, 3) for d in (-0.05, -0.025, 0, 0.025, 0.05)])
    thr_pred = apply_thresholds(one_to_one(oof), *params2)
    rep_thr = score_report(thr_pred, counts, all_a)
    ef_pred = expected_f05_select(one_to_one(oof).filter(pl.col("p") >= 0.02))
    rep_ef = score_report(ef_pred, counts, all_a)
    use_ef = rep_ef["macro_f05"] > rep_thr["macro_f05"]
    return {"thresholds": params2, "threshold_report": rep_thr, "expected_f_report": rep_ef,
            "decision": "expected_f05" if use_ef else "thresholds"}


def decide(pred: pl.DataFrame, cfg: dict) -> pl.DataFrame:
    base = one_to_one(pred)
    if cfg["decision"] == "expected_f05":
        return expected_f05_select(base.filter(pl.col("p") >= 0.02))
    return apply_thresholds(base, *cfg["thresholds"])


def log_experiment(title: str, notes: str, blocking: dict, cfg: dict) -> None:
    new = not os.path.exists(EXP)
    with open(EXP, "a") as f:
        if new:
            f.write("# Experiments\n\nEvery run: blocking recall, candidates per S1, OOF macro-F0.5 "
                    "(GroupKFold by S1), precision/recall, by country and singleton status.\n\n")
        f.write(f"## {title} ({datetime.date.today()})\n\n{notes}\n\n")
        f.write("**Blocking:** " + ", ".join(f"{k}={v}" for k, v in blocking.items()) + "\n\n")
        f.write(f"**Decision:** {cfg['decision']}, thresholds (t_abs, r, t_empty) = {cfg['thresholds']}\n\n")
        for name in ("threshold_report", "expected_f_report"):
            f.write(f"- {name}: " + ", ".join(f"{k}={v}" for k, v in cfg[name].items()) + "\n")
        f.write("\n")


def write_submission(pred: pl.DataFrame, cfg: dict) -> None:
    A = load("test", "A").select(pl.col("idx").alias("a_idx"), pl.col("entity_id").alias("source1_entity_id"))
    Bid = load("test", "B").select(pl.col("idx").alias("b_idx"), pl.col("entity_id").alias("bid"))
    chosen = decide(pred, cfg).join(Bid, on="b_idx")
    m = chosen.group_by("a_idx").agg(pl.col("bid").sort().str.join(",").alias("matched_entity_ids"))
    res = A.join(m, on="a_idx", how="left").with_columns(pl.col("matched_entity_ids").fill_null(""))
    res.select("source1_entity_id", "matched_entity_ids").write_csv(os.path.join(OUT, "matching_results.tsv"), separator="\t", quote_style="never")
    cand = pl.concat([pl.read_parquet(f, columns=["a_idx", "b_idx"]) for f in glob.glob(os.path.join(WORK, "cand_test_*.parquet"))])
    c = cand.join(Bid, on="b_idx").group_by("a_idx").agg(pl.col("bid").sort().str.join(",").alias("candidate_entity_ids"))
    cres = A.join(c, on="a_idx", how="left").with_columns(pl.col("candidate_entity_ids").fill_null(""))
    cres.select("source1_entity_id", "candidate_entity_ids").write_csv(os.path.join(OUT, "candidate_pairs.tsv"), separator="\t", quote_style="never")
    with open(os.path.join(WORK, "decision.json"), "w") as f:
        json.dump(cfg, f)
    print(f"[submit] wrote {res.height:,} rows; non-empty {(res['matched_entity_ids'] != '').sum():,}", flush=True)
