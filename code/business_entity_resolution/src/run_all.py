"""One command: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv.

    python src/run_all.py [--from STAGE] [--to STAGE] [--no-stack] [--title "run name"] [--notes "..."]

Stages: mine, prepare, train_data, test_data, fit, oof, test, stack, tune, submit.
  fit/oof/test  round-1 LightGBM (tag m1): fold models, OOF on train, averaged test preds
  stack         round-2 (tag m2) on round-1 features + cluster features from m1 probabilities
  tune/submit   use m2 when stacking is on, else m1
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
import stack
from io_utils import WORK

STAGES = ["mine", "prepare", "train_data", "test_data", "fit", "oof", "test", "stack", "tune", "submit"]


def _load_models(tag):
    with open(os.path.join(WORK, f"{tag}_models.pkl"), "rb") as f:
        return pickle.load(f)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", default="mine", choices=STAGES)
    ap.add_argument("--to", dest="stop", default="submit", choices=STAGES)
    ap.add_argument("--no-stack", action="store_true")
    ap.add_argument("--title", default="run")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()
    todo = STAGES[STAGES.index(args.start): STAGES.index(args.stop) + 1]
    final = "m1" if args.no_stack else "m2"
    t0 = time.time()
    if "mine" in todo:
        mine.main()
    if "prepare" in todo:
        maps = mine.load_maps()
        for split in ("train", "test"):
            prepare.prepare(split, maps)
            p = os.path.join(WORK, f"rec_{split}.npz")  # record arrays depend on the cache
            if os.path.exists(p):
                os.remove(p)
    if "train_data" in todo:
        pipeline.run_split("train")
    if "test_data" in todo:
        pipeline.run_split("test")
    if "fit" in todo:
        model.train_folds("m1")
    if "oof" in todo:
        model.predict_oof(_load_models("m1"), "m1")
    if "test" in todo:
        model.predict_test(_load_models("m1"), "m1")
    if "stack" in todo and not args.no_stack:
        tr = pl.read_parquet(stack.build("train", "m1"))
        m2 = model.train_folds("m2", tr)
        model.predict_oof(m2, "m2", tr)
        del tr
        te = pl.read_parquet(stack.build("test", "m1"))
        model.predict_test(m2, "m2", te)
        del te
    cfg = None
    if "tune" in todo:
        oof = pl.read_parquet(os.path.join(WORK, f"{final}_oof.parquet"))
        cfg = evaluate.tune(oof)
        blk = evaluate.blocking_report("train")
        print(json.dumps({"blocking": blk, **cfg}, indent=1), flush=True)
        evaluate.log_experiment(f"{args.title} [{final}]", args.notes, blk, cfg)
        with open(os.path.join(WORK, f"{final}_decision.json"), "w") as f:
            json.dump(cfg, f)
    if "submit" in todo:
        if cfg is None:
            with open(os.path.join(WORK, f"{final}_decision.json")) as f:
                cfg = json.load(f)
        pred = pl.read_parquet(os.path.join(WORK, f"{final}_test.parquet"))
        evaluate.write_submission(pred, cfg)
    print(f"done in {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
