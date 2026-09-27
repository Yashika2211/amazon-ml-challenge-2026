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

Stages (resume with `--from STAGE`, stop with `--to STAGE`):

| Stage | What it does |
| --- | --- |
| `mine` | alias tables and noise-token rates from train positives |
| `prepare` | normalized parquet cache for train and test |
| `blocker` | trains the learned blocking ranker on 10% of train S1 |
| `train_data`, `test_data` | candidates (blocking) and pair features |
| `fit`, `oof`, `test` | round-1 LightGBM: fold models, out-of-fold predictions, test predictions |
| `stack` | round-2 LightGBM with cluster and competitor features |
| `pseudo` | countries without training data (France): mine aliases and noise-token rates from test pseudo-positives, redo test inference |
| `stack2` | round-3 LightGBM with cluster features recomputed from round-2 probabilities |
| `tune`, `submit` | decision thresholds on OOF macro-F0.5, write both TSV files, append to `EXPERIMENTS.md` |

Intermediate files go to `work/` (override with `ER_WORK`), outputs to `output/` (`ER_OUT`).

**Hardware used:** Apple M-series laptop, 10 cores, 16 GB RAM. Peak RAM about 9 GB.
**Runtime:** about 5 hours end to end (blocking about 1.5 h, features 15 min, three LightGBM rounds about 1 h,
pseudo stage about 45 min). Per-run numbers are in `EXPERIMENTS.md`.
**Disk:** about 40 GB in `work/`.

## Pipeline

| Stage | File | What it does |
| --- | --- | --- |
| scorer | `src/metrics.py`, `src/decide.py` | exact per-S1 F0.5, macro average (unit test: `src/test_metrics.py`) |
| mine | `src/mine.py` | mines from TRAIN positives: Indic-transliteration token dictionary, per-country address component aliases (state names/codes/native script, city spellings), generic noise-token rates |
| prepare | `src/normalize.py`, `src/prepare.py` | NFKC + anyascii, junk prefixes, domains/hashtags, DBA/AKA split, digit-letter confusion, legal-form canonicalization, core/compact names; address bag, components, numbers |
| blocking | `src/blocking.py`, `src/blocker.py` | per country: union of 10 key types (rare name tokens, name bigram, compact name, address number x street token, street bigrams, name x address, name x number, 1-deletion typo keys); a learned LightGBM ranker scores raw key matches; keep top 25 per S1 plus top 5 S1 per S2/S3 record |
| features | `src/features.py`, `src/pipeline.py` | rapidfuzz string similarities on several name forms, IDF-weighted set overlap, sibling detector (distinctive unmatched tokens), legal-form agreement, address number/component overlap, rank/gap context |
| model | `src/model.py`, `src/stack.py` | LightGBM, 3-fold GroupKFold by S1 id, isotonic calibration on OOF; rounds 2 and 3 add cluster and competitor-branch features from the previous round's probabilities |
| kernels | `src/fastops.py` | numba merge-intersection kernels for weighted set overlap |
| decision | `src/decide.py`, `src/evaluate.py` | one-to-one assignment of S2/S3 records, per-S1 thresholds or expected-F0.5 prefix, tuned on OOF macro-F0.5 |

`country` is used only to partition blocking and to pick the mined alias table; it is never
a model feature, and unseen countries (France) go through the same code path.

## Hand-written maps (allowed; listed for audit)

`src/maps.py`: legal forms (EN/IN/FR, including SASU = SAS and EURL = SARL), French name
abbreviations (frs, ets), name stop words and honorifics, street-type abbreviations (EN/IN/FR),
address stop words, null tokens. Everything else is mined from the provided files in `src/mine.py`:
train positives for US/India, and for countries absent from train (France) label-free test
pseudo-positives plus S2/S3-vs-S1 token over-representation. No external data is used.

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
