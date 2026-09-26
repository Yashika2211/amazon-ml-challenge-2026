# Business Entity Resolution (Amazon ML Challenge 2026)

Blocking + LightGBM pair classifier + one-to-one / F0.5-optimal decision layer.
Every step is learned from the provided training files only (no external data, APIs,
gazetteers or pretrained models).

## Reproduce

```bash
python3.11 -m venv .venv && .venv/bin/pip install -r code/business_entity_resolution/requirements.txt
# data expected at student_resource/dataset/{train,test} (override with ER_DATA=/path/to/dataset)
.venv/bin/python code/business_entity_resolution/src/run_all.py
# -> output/matching_results.tsv, output/candidate_pairs.tsv
cd student_resource && python3 utils/validate_submission.py \
    --matching ../output/matching_results.tsv --candidate ../output/candidate_pairs.tsv --test-dir dataset/test
```

Stages can be resumed with `--from {mine,prepare,train_data,test_data,fit,oof,tune,test,submit}`.
Intermediate files go to `work/` (override with `ER_WORK`), outputs to `output/` (`ER_OUT`).

**Hardware used:** Apple M-series, 10 cores, 16 GB RAM. Peak RAM about 10 GB.
Runtime figures are recorded in `EXPERIMENTS.md`.

## Pipeline

| Stage | File | What it does |
| --- | --- | --- |
| scorer | `src/metrics.py`, `src/decide.py` | exact per-S1 F0.5, macro average (unit test: `src/test_metrics.py`) |
| mine | `src/mine.py` | mines from TRAIN positives: Indic-transliteration token dictionary, per-country address component aliases (state names/codes/native script, city spellings), generic noise-token rates |
| prepare | `src/normalize.py`, `src/prepare.py` | NFKC + anyascii, junk prefixes, domains/hashtags, DBA/AKA split, digit-letter confusion, legal-form canonicalization, core/compact names; address bag, components, numbers |
| blocking | `src/blocking.py` | per country: union of rare-name-token, name-bigram, compact-name, address-number, address-bigram and name x address keys; numeric cheap ranking; keep top-K per S1 plus top-k S1 per S2/S3 record |
| features | `src/features.py`, `src/pipeline.py` | rapidfuzz string similarities on several name forms, IDF-weighted set overlap, sibling detector (distinctive unmatched tokens), legal-form agreement, address number/component overlap, rank/gap context |
| model | `src/model.py` | LightGBM, 3-fold GroupKFold by S1 id, isotonic calibration on OOF |
| decision | `src/decide.py`, `src/evaluate.py` | one-to-one assignment of S2/S3 records, per-S1 thresholds or expected-F0.5 prefix, tuned on OOF macro-F0.5 |

`country` is used only to partition blocking and to pick the mined alias table; it is never
a model feature, and unseen countries (France) go through the same code path.

## Hand-written maps (allowed; listed for audit)

`src/maps.py`: legal forms (EN/IN/FR), name stop words and honorifics, street-type
abbreviations (EN/IN/FR), address stop words, null tokens. Everything else is mined from
training positives in `src/mine.py`.

## Library licenses

| Library | License |
| --- | --- |
| polars, polars-runtime-32 | MIT |
| pyarrow | Apache-2.0 |
| numpy, scipy, scikit-learn, joblib, threadpoolctl | BSD-3-Clause |
| lightgbm | MIT |
| numba, llvmlite | BSD-2-Clause |
| rapidfuzz | MIT |
| anyascii | ISC |

No pretrained model is used.
