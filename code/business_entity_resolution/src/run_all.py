"""One command: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python src/run_all.py [--from STAGE] [--title "run name"] [--notes "..."]

Stages: mine, prepare, train_data, test_data, fit, oof, tune, test, submit.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import polars as pl

import evaluate
import mine
import model
import pipeline
import prepare
from io_utils import WORK

STAGES = ["mine", "prepare", "train_data", "test_data", "fit", "oof", "tune", "test", "submit"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="mine", choices=STAGES)
    ap.add_argument("--to", dest="stop", default="submit", choices=STAGES)
    ap.add_argument("--title", default="run")
    ap.add_argument("--notes", default="")
    ap.add_argument("--tag", default="m1")
    args = ap.parse_args()
    todo = STAGES[STAGES.index(args.start): STAGES.index(args.stop) + 1]
    t0 = time.time()
    if "mine" in todo:
        mine.main()
    if "prepare" in todo:
        maps = mine.load_maps()
        for split in ("train", "test"):
            prepare.prepare(split, maps)
        for split in ("train", "test"):  # record arrays depend on the cache
            p = os.path.join(WORK, f"rec_{split}.npz")
            if os.path.exists(p):
                os.remove(p)
    if "train_data" in todo:
        pipeline.run_split("train")
    if "test_data" in todo:
        pipeline.run_split("test")
    models = None
    if "fit" in todo:
        models = model.train_folds(args.tag)
    if models is None and any(s in todo for s in ("oof", "test")):
        with open(os.path.join(WORK, f"{args.tag}_models.pkl"), "rb") as f:
            models = pickle.load(f)
    if "oof" in todo:
        model.predict_oof(models, args.tag)
    cfg = None
    if "tune" in todo:
        oof = pl.read_parquet(os.path.join(WORK, f"{args.tag}_oof.parquet"))
        cfg = evaluate.tune(oof)
        blk = evaluate.blocking_report("train")
        print(json.dumps({"blocking": blk, **cfg}, indent=1), flush=True)
        evaluate.log_experiment(args.title, args.notes, blk, cfg)
        with open(os.path.join(WORK, f"{args.tag}_decision.json"), "w") as f:
            json.dump(cfg, f)
    if "test" in todo:
        model.predict_test(models, args.tag)
    if "submit" in todo:
        if cfg is None:
            with open(os.path.join(WORK, f"{args.tag}_decision.json")) as f:
                cfg = json.load(f)
        pred = pl.read_parquet(os.path.join(WORK, f"{args.tag}_test.parquet"))
        evaluate.write_submission(pred, cfg)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
