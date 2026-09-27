# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** YAAD
**Team Members:** Yashika Bhatia
**Submission Date:** 2026-09-27

---

## 1. Executive Summary

We resolve Source 2 and Source 3 records to Source 1 entities with a cascade: text normalization that mines its alias tables from training positives, a multi-key blocking stage ranked by a learned LightGBM blocker, a LightGBM pair classifier with a second stacked round that reasons about competing S1 records, and an F0.5-optimal decision layer with one-to-one assignment. The main innovations are the learned blocking ranker (pair recall at 25 candidates per S1 rises from 98.0% to 99.98% of what the keys can reach), alias-name splitting for DBA / AKA / FKA / "formerly" / "née" names, and pseudo-positive mining that adapts the pipeline to France, which has no training data. Out-of-fold macro-F0.5 on train: **0.9874**.

---

## 2. Methodology

### 2.1 Problem Analysis

Measured on the provided files:

- **Scale.** Train has 2.21M S1, 5.03M S2 and 5.29M S3 records (US + India). Test has 1.73M S1 (India 810k, US 663k, France 259k), 4.89M S2 and 5.08M S3. France has no training rows.
- **Ground truth.** 5.6% of S1 are singletons. Each S2/S3 id appears in at most one S1 list, and positives are always the same country. S2 and S3 copy counts per S1 are independent (correlation -0.02); per source, 1 copy is the most common count (38%) and 0 copies is rare (8%).
- **Distractors.** 26% of S2/S3 records match nothing. Most are near-duplicates of an S1 at a slightly shifted house number (111 vs 116, 11502 vs 11507), or siblings with an extra distinctive name token.
- **Branches.** About half of all S1 share their exact core name with another S1 of the same country (chains / branches). An S2/S3 copy with an empty address (3.0-3.6% of records in train) is then ambiguous between branches. This is the largest single source of error: 70% of rejected true candidates have an empty address.
- **Name noise.** Native Indic scripts (23% of India S2 names), domain and hashtag names, junk prefixes, token reordering, duplicated words, digit-for-letter typos, inserted generic words, truncation, invented brand names at the same address, and alias names marked by "dba", "aka", "f/k/a", "formerly (known as)" and "née" (about 70k train S3 names).
- **Address noise.** Component reordering, state as name / code / native script, city aliases (Bombay/Mumbai, Calcutta/Kolkata), number reformatting, null tokens, PO boxes, empty addresses. In France the last field mixes région and département (Gironde vs Nouvelle-Aquitaine).
- **Test vs train.** Test has more S2/S3 records per S1 (5.75 vs 4.68), fewer empty addresses (2.4-2.9%) and, in the US, fewer shared names (40% vs 48% of S1).

### 2.2 Solution Strategy

**Approach Type:** Blocking + learned blocking ranker + stacked gradient-boosted pair classifier + global decision layer
**Core Innovation:** A learned blocker that ranks raw key matches before pruning, alias-name splitting, competitor-aware stacking, and pseudo-positive adaptation for an unseen country.

Pipeline (one command, `src/run_all.py`):

1. **Mine** alias tables from train positives: Indic transliteration dictionary (526 tokens), per-country address component aliases (state codes, native-script states, city spellings), generic noise-token insertion rates.
2. **Normalize** names and addresses (NFKC, anyascii, legal-form canonicalization, alias-name split, digit-letter confusion, compact form; address bag, components, numbers).
3. **Learned blocker** trained on un-pruned key matches of 10% of train S1.
4. **Candidates** per country: union of 10 key types, scored by the learned blocker, keep top 12 per S1 plus top 3 S1 per S2/S3 record (18.8 candidates per S1 on train).
5. **Features** (numeric, country-agnostic).
6. **Round-1 LightGBM**, 3-fold GroupKFold by S1, isotonic calibration on out-of-fold scores.
7. **Round-2 LightGBM** on round-1 features plus cluster and competitor features.
8. **Pseudo stage** for countries without training data (France): mine aliases and noise tokens from confident test pairs, redo test inference with the same models.
9. **Decision**: one-to-one assignment, then per-S1 expected-F0.5 prefix selection with an empty threshold, tuned on OOF macro-F0.5.
10. **Test prior-shift correction** (`src/shift.py`): test contains up to twice as many same-name near-duplicates at a shifted house number per S1 as train, while exact copies grow only slightly. Per country and pair bucket, match odds are multiplied by (growth of the exact-copy bucket) / (growth of the bucket), capped at 1. Label-free; a leaderboard probe that moved these odds the other way lost 0.003.

---

## 3. Candidate Generation (Blocking)

- **Blocking keys used (within country only):**
  - `t`: the 3 rarest core-name tokens (document frequency capped).
  - `b`: bigram of the 2 rarest core-name tokens.
  - `c`, `p`, `s`: compact name (alphanumerics without legal form), full, 6-char prefix and 6-char suffix. These catch domain and hashtag names.
  - `a`: (address number, one of the 3 rarest street tokens).
  - `d`: bigrams of the 3 rarest street / locality tokens.
  - `x`: (rarest name token, rarest address token).
  - `y`: (one of the 2 rarest name tokens, address number).
  - `e`: 1-deletion neighbourhood of the 2 rarest name tokens (typo tolerance).
  - Keys whose block product |S1| x |S2/S3| exceeds a per-type cap are skipped.
- **Ranking and pruning:** a LightGBM blocker scores every raw key match from key-hit counts per type, IDF-weighted name / address / component overlap, number agreement, empty-address flags and the hand-weighted score. We keep the top 12 candidates per S1 and add any pair in the top 3 S1 of its S2/S3 record. The learned ranker keeps 99.7% of key-reachable true pairs within the top 10, so the candidate set could shrink from 34 to 18.8 per S1 for a 0.09-point recall cost.
- **Candidate pairs generated:** train 41,397,191 (18.8 per S1); test 35,882,653 (20.7 per S1: France 5.24M, India 17.05M, US 13.59M).
- **Reduction ratio:** 99.9998% of the full cross product within country.
- **How true matches were not lost:**

| Stage | Pair recall (train) |
| --- | --- |
| Union of all keys, before pruning | India 98.67%, US 99.06% |
| Hand-weighted score, top 25 per S1 | 97.97% of reachable pairs |
| Learned blocker, top 25 per S1 | 99.98% of reachable pairs |
| Final candidate set | 98.79% (India 98.58%, US 98.94%) at 18.8 candidates per S1 |

---

## 4. Matching Model

**Features used (all numeric; `country` is never a feature):**

- **Name features:** rapidfuzz ratio, partial ratio, token sort and token set on the core name; ratio, partial ratio and Jaro-Winkler on the compact name; token set on the full name; ratios on a transliteration-robust consonant skeleton; similarity to the DBA / AKA / FKA alternate name. IDF-weighted Jaccard, overlap and coverage on name tokens (core plus alternate name), plus a generic-token-discounted overlap. Sibling detector: count and max document frequency of distinctive unmatched tokens on each side. Generic-token mass, first-token match, compact containment, legal-form agree / conflict / missing, corpus frequency of the core name, native-script flag, lengths.
- **Address features:** token set, token sort and partial ratio. IDF-weighted overlap on address tokens and on components (city / state after aliasing). Number-set Jaccard and intersection, first-number equality and containment, number conflict. One-digit-edit number match (typo drift), log absolute and relative first-number difference, near-shift flag (distractor detector). Empty flags.
- **Context:** blocker score and its rank within the S1 and within the S2/S3 record, gap to the best candidate on both sides, candidate counts, key-hit counts per key type, source (S2 / S3).
- **Round-2 cluster and competitor features:** round-1 probability and ranks; best competing probability on the S1 side and on the record side; S1 probability mass and number of confident candidates; name and address similarity of the record to the S1's best other candidates (S2 and S3 copies of one entity reinforce each other). Confident same-source and total copy counts for this S1 and for the strongest competing S1 (per-source copy-count prior for branch disambiguation), and record ambiguity (number of S1 with p > 0.2, probability mass).

**Model type:** LightGBM binary classifier (MIT; no pretrained model): num_leaves 255, learning rate 0.06, early stopping on the held-out fold, 3-fold GroupKFold by S1 id on a 30% sample of S1 groups, out-of-fold predictions for every train pair, isotonic calibration. Round 2 uses the same folds. A third round (cluster features recomputed from round-2 probabilities) tied with round 2 (0.98713 vs 0.98710) and is not used in the final run.

**Threshold selection method:** tuned on OOF macro-F0.5, never on AUC.

1. One-to-one: each S2/S3 record keeps only its highest-probability S1.
2. Per S1, we compare (a) keeping candidates with p >= t_abs, p >= r x p_max and p_max >= t_empty (grid search), and (b) choosing the probability-sorted prefix that maximizes expected F0.5, 1.25 x sum(p_top_k) / (0.25 x E[n_true] + k), against P(no match) = prod(1 - p_i), with a minimum p_max. We use whichever scores higher on OOF: expected-F0.5 prefix selection with p_max >= 0.5 (OOF 0.98741 vs 0.98737 for the best threshold rule t_abs=0, r=0.7, t_empty=0.5).

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro, out-of-fold on train):** 0.9874

| Run | Blocking recall | OOF macro-F0.5 | India | US | Singleton | Non-singleton |
| --- | --- | --- | --- | --- | --- | --- |
| v1 baseline | 97.12% | 0.97775 | 0.97509 | 0.97952 | 0.97835 | 0.97771 |
| v2 more keys, number features, stacking | 97.89% | 0.98437 | 0.98216 | 0.98584 | 0.98711 | 0.98420 |
| v2b competitor-branch features | 97.89% | 0.98467 | 0.98255 | 0.98609 | 0.98771 | 0.98449 |
| v3 learned blocker (34 cand/S1), alias fix, rounds 2+3 | 98.88% | 0.98713 | 0.98586 | 0.98798 | 0.98793 | 0.98708 |
| **v4 final**: 18.8 cand/S1, 30% training sample, round 2 | 98.79% | **0.98741** | 0.98620 | 0.98822 | 0.98819 | 0.98736 |

**Leave-one-country-out and France:** training on US only and scoring India as an unseen country gives 0.9631 (vs 0.9862 when India is in training), which shows how much an unseen country costs. Self-training on confident pseudo-labels did not help there (0.9623). France has no labels. We checked that France predictions look like US and India (predicted list size, empty rate, share of uncertain candidates). Before the pseudo stage France had about twice as many uncertain candidates per S1 (0.53 vs 0.23-0.31). The causes were French noise words ("& Fils", "& Associés", "& Cie"), SAS/SASU and SARL/EURL swaps, and région vs département in the last address field. The pseudo stage learned the four département to région aliases (Nord and Pas-de-Calais to Hauts-de-France, Gironde to Nouvelle-Aquitaine, Loire-Atlantique to Pays de la Loire) and the French noise words from test pseudo-positives, without labels.

- **Common false positives (wrong merges):**
  - Near-duplicate distractors: same name at a slightly shifted house number (addressed by number-shift features).
  - Identical names with an empty-address copy, assigned to the wrong branch.
  - Invented brand names at the same address that belong to a different entity.
  - Siblings whose extra token looks generic ("Golden Projects LLP" vs "Golden Services LLP").
- **Common false negatives (missed matches):**
  - Empty-address copies of names shared by several branches (70% of rejected true candidates). The count prior resolves only part of these; the rest are genuinely ambiguous.
  - Copies whose name and address are both heavily mutated (truncated name plus dropped house number).
  - Blocking misses (about 1% of pairs), mostly common names whose keys exceed the block caps.

**Leaderboard checks (public subset):** cycle 3 with France country-specific noise weights 0.9807; the same without them 0.9760; the latter with more permissive shifted-number matches 0.9730. The France weights stay on, and the decoy correction goes in the conservative direction. Train OOF overstates the test score because test is denser in decoys (uncertain candidates per S1: train 0.20-0.22, test 0.27-0.33 for India/US, 0.72 for France).

**Loss decomposition (v2, OOF):** blocking misses 0.0066, rejected true candidates 0.0069, wrong matches 0.0022. Measured by adding back or removing each error type.

---

## 6. Conclusion

A normalization layer mined from training positives, a learned blocker and a stacked, competitor-aware LightGBM take OOF macro-F0.5 from 0.9778 to 0.9874, with blocking recall close to the key-union ceiling. The biggest lessons: rank blocking candidates with a model rather than a formula, split alias names before matching, and treat an S2/S3 record's competing S1 candidates as evidence. Remaining errors are dominated by empty-address copies of chain businesses, which no content feature can fully disambiguate.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/`:

| File | Role |
| --- | --- |
| `src/run_all.py` | entry point, stages mine to submit, resumable with `--from STAGE` |
| `src/io_utils.py` | TSV loading, paths |
| `src/maps.py` | hand-written maps: legal forms (EN/IN/FR), street types (EN/IN/FR), stop words, null tokens |
| `src/normalize.py` | name and address normalization |
| `src/mine.py` | alias mining from train positives; pseudo-positive mining for unseen countries |
| `src/prepare.py` | normalized parquet cache |
| `src/blocking.py`, `src/blocker.py` | blocking keys, learned blocking ranker, pruning |
| `src/fastops.py` | numba merge-intersection kernels for weighted set overlap |
| `src/features.py`, `src/pipeline.py` | record arrays, pair features, chunked feature generation |
| `src/model.py` | LightGBM GroupKFold training, OOF, calibration, test prediction |
| `src/stack.py` | round-2 cluster and competitor features |
| `src/decide.py`, `src/evaluate.py` | decision layer, exact macro-F0.5, experiment log, submission writer |
| `src/metrics.py`, `src/test_metrics.py` | reference scorer and unit test (README example = 0.714) |
| `src/package.py` | builds the submission zip |

Reproduce: `python src/run_all.py` (see `README.md`). Runtime on a 10-core, 16 GB laptop: about 5 hours end to end.

### B. Additional Results

Blocking recall at K per S1 on held-out train S1 groups (share of key-reachable true pairs):

| K | 5 | 10 | 15 | 20 | 25 | 30 | 50 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Hand-weighted | 0.8638 | 0.9531 | 0.9688 | 0.9755 | 0.9797 | 0.9822 | 0.9869 |
| Learned blocker | 0.9403 | 0.9973 | 0.9991 | 0.9996 | 0.9998 | 0.9999 | 0.99995 |

Full per-run logs are in `EXPERIMENTS.md`.
