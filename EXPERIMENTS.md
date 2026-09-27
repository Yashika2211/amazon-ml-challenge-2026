# Experiments

Every run: blocking recall, candidates per S1, OOF macro-F0.5 (GroupKFold by S1), precision/recall, by country and singleton status.

## Baseline v1: 8 blocking keys, K=15/kb=3, LightGBM, thresholds (2026-09-27)

First end-to-end run. Numeric cheap ranking; 3-fold GroupKFold on a 20% S1 sample; isotonic calibration.

**Blocking:** pairs=43409046, pairs_per_s1=19.67, pair_recall=0.9712, pair_recall_India=0.96334, pair_recall_US=0.97645, reduction_ratio=0.99999809

**Decision:** thresholds, thresholds (t_abs, r, t_empty) = (0.15, 0.7, 0.675)

- threshold_report: macro_f05=0.97775, f05_India=0.97509, f05_US=0.97952, f05_singleton=0.97835, f05_nonsingleton=0.97771, micro_precision=0.99589, micro_recall=0.94487, pred_empty_rate=0.0592
- expected_f_report: macro_f05=0.97728, f05_India=0.9747, f05_US=0.979, f05_singleton=0.959, f05_nonsingleton=0.97836, micro_precision=0.99608, micro_recall=0.94358, pred_empty_rate=0.0575

## v2: +4 blocking keys, number edit/shift features, round-2 stacking [m2] (2026-09-27)

Blocking adds name x number, 3-token address bigrams, 1-deletion typo keys; digit-edit-1 and house-number shift features; dotted acronym collapse; 30% training sample; round-2 cluster features; hybrid expected-F decision.

**Blocking:** pairs=45606134, pairs_per_s1=20.67, pair_recall=0.97894, pair_recall_India=0.97342, pair_recall_US=0.98263, reduction_ratio=0.999998

**Decision:** expected_f05, thresholds (t_abs, r, t_empty) = (0.0, 0.7, 0.45), expected-F (miss_rate, t_empty) = (0.0, 0.4)

- threshold_report: macro_f05=0.98432, f05_India=0.9821, f05_US=0.98581, f05_singleton=0.98665, f05_nonsingleton=0.98419, micro_precision=0.99739, micro_recall=0.95886, pred_empty_rate=0.0582
- expected_f_report: macro_f05=0.98437, f05_India=0.98216, f05_US=0.98584, f05_singleton=0.98711, f05_nonsingleton=0.9842, micro_precision=0.9978, micro_recall=0.95786, pred_empty_rate=0.0583

## v2b: competitor-branch count features in round 2 [m2] (2026-09-27)

Same candidates/features as v2; round-2 adds own vs strongest-competitor confident copy counts per source, record ambiguity (n S1 with p>0.2, p mass).

**Blocking:** pairs=45606134, pairs_per_s1=20.67, pair_recall=0.97894, pair_recall_India=0.97342, pair_recall_US=0.98263, reduction_ratio=0.999998

**Decision:** expected_f05, thresholds (t_abs, r, t_empty) = (0.0, 0.7, 0.45), expected-F (miss_rate, t_empty) = (0.02, 0.5)

- threshold_report: macro_f05=0.98463, f05_India=0.98248, f05_US=0.98606, f05_singleton=0.98606, f05_nonsingleton=0.98454, micro_precision=0.99743, micro_recall=0.95985, pred_empty_rate=0.0581
- expected_f_report: macro_f05=0.98467, f05_India=0.98255, f05_US=0.98609, f05_singleton=0.98771, f05_nonsingleton=0.98449, micro_precision=0.99783, micro_recall=0.9589, pred_empty_rate=0.0583

## v3: alias-marker fix, learned blocker K=25/kb=5, numba kernels, French pseudo-maps, larger LightGBM [m2] (2026-09-27)

FKA/formerly/nee/DBA: alias splitting (~70k train S3 names); French legal forms and abbreviations; learned LightGBM blocking ranker; top-25 per S1 + top-5 per record; competitor-branch round-2 features; pseudo-positive mining of France address aliases and noise tokens; num_leaves 255, lr 0.06, 20% sample.

**Blocking:** pairs=75089429, pairs_per_s1=34.03, pair_recall=0.98884, pair_recall_India=0.9866, pair_recall_US=0.99033, reduction_ratio=0.9999967

**Decision:** expected_f05, thresholds (t_abs, r, t_empty) = (0.0, 0.7, 0.5), expected-F (miss_rate, t_empty) = (0.0, 0.5)

- threshold_report: macro_f05=0.98707, f05_India=0.98577, f05_US=0.98793, f05_singleton=0.98712, f05_nonsingleton=0.98706, micro_precision=0.99741, micro_recall=0.96742, pred_empty_rate=0.0577
- expected_f_report: macro_f05=0.9871, f05_India=0.98582, f05_US=0.98796, f05_singleton=0.98712, f05_nonsingleton=0.9871, micro_precision=0.99773, micro_recall=0.96658, pred_empty_rate=0.0577

## v3b: France country-specific noise weights, round-3 stacking [m3] (2026-09-27)

Same candidates/features as v3; France aliases re-mined from raw components; France generic rates from over-representation and pseudo-positives (France pairs only); round 3 recomputes cluster/competitor features from round-2 probabilities.

**Blocking:** pairs=75089429, pairs_per_s1=34.03, pair_recall=0.98884, pair_recall_India=0.9866, pair_recall_US=0.99033, reduction_ratio=0.9999967

**Decision:** expected_f05, thresholds (t_abs, r, t_empty) = (0.0, 0.7, 0.5), expected-F (miss_rate, t_empty) = (0.02, 0.5)

- threshold_report: macro_f05=0.98709, f05_India=0.98581, f05_US=0.98794, f05_singleton=0.98793, f05_nonsingleton=0.98704, micro_precision=0.99744, micro_recall=0.96746, pred_empty_rate=0.0578
- expected_f_report: macro_f05=0.98713, f05_India=0.98586, f05_US=0.98798, f05_singleton=0.98793, f05_nonsingleton=0.98708, micro_precision=0.99776, micro_recall=0.96665, pred_empty_rate=0.0578

## Findings log (2026-09-27)

- **Leaderboard calibration.** The first public leaderboard upload (cycle 3/3b file) scored **0.980665**. The label-free expected-F0.5 estimate on test for the same predictions was 0.979-0.980, so the estimator tracks the leaderboard closely. It estimates France at 0.957-0.966 and US/India at about 0.983, which explains most of the gap to train OOF (0.9871).
- **Leave-one-country-out (train US, score India with real labels), round-1 features:** plain 0.9631. Adding unseen-country generic weights from S2/S3-vs-S1 token over-representation scored 0.9283, so those weights were **rejected** and disabled (`mine_pseudo(country_generic=False)`). Cycle 3c = cycle 3b without them: test estimate 0.9791 -> 0.9805 (France 0.957 -> 0.967).
- **Test is denser than train.** Test has 5.75 S2/S3 records per S1 vs 4.68, and about twice as many near-duplicate records at a house number shifted by at most 20 (0.47 vs 0.26 per S1). Uncertain candidates per S1 (0.1 < p < 0.9): train OOF India 0.20 / US 0.22; test India 0.27 / US 0.33 / France 0.72.
- **France.** Category-word swaps at the same address ("sportive" vs "primaire") are not over-represented in S2/S3 (ratio about 0.85, baseline 0.89), unlike inserted noise words ("international" 30x, "participations", "associes", "developpement"). France S1 contains many same-number, same-city entities on different streets.

