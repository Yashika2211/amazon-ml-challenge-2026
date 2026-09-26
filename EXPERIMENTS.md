# Experiments

Every run: blocking recall, candidates per S1, OOF macro-F0.5 (GroupKFold by S1), precision/recall, by country and singleton status.

## Baseline v1: 8 blocking keys, K=15/kb=3, LightGBM, thresholds (2026-09-27)

First end-to-end run. Numeric cheap ranking; 3-fold GroupKFold on a 20% S1 sample; isotonic calibration.

**Blocking:** pairs=43409046, pairs_per_s1=19.67, pair_recall=0.9712, pair_recall_India=0.96334, pair_recall_US=0.97645, reduction_ratio=0.99999809

**Decision:** thresholds, thresholds (t_abs, r, t_empty) = (0.15, 0.7, 0.675)

- threshold_report: macro_f05=0.97775, f05_India=0.97509, f05_US=0.97952, f05_singleton=0.97835, f05_nonsingleton=0.97771, micro_precision=0.99589, micro_recall=0.94487, pred_empty_rate=0.0592
- expected_f_report: macro_f05=0.97728, f05_India=0.9747, f05_US=0.979, f05_singleton=0.959, f05_nonsingleton=0.97836, micro_precision=0.99608, micro_recall=0.94358, pred_empty_rate=0.0575

