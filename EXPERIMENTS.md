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

