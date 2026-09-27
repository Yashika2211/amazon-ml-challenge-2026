"""Comprehensive error analysis and model improvement pipeline.

This script performs:
1. Error analysis: worst segments, patterns, concentration
2. Feature audit: correlation, importance, coverage
3. New feature engineering: interaction terms, aggregations, target encoding
4. Hyperparameter tuning with Optuna
5. Ensemble building: LightGBM + XGBoost + CatBoost
6. Pseudo-labeling for test set
"""
from __future__ import annotations

import os
import pickle
import sys
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold

# Add src to path
sys.path.insert(0, os.path.dirname(__file__))

from decide import per_a_scores, macro_f05
from io_utils import WORK, load_truth
from model import ID_COLS, feature_names, truth_pairs, N_FOLDS
from prepare import load

print("=" * 80)
print("COMPREHENSIVE ERROR ANALYSIS & MODEL IMPROVEMENT")
print("=" * 80)

# ============================================================================
# 1. ERROR ANALYSIS
# ============================================================================

def error_analysis():
    """Identify worst-performing segments and error patterns."""
    print("\n" + "=" * 80)
    print("1. ERROR ANALYSIS - Finding Worst Segments")
    print("=" * 80)
    
    # Load OOF predictions from best model (m2)
    oof = pl.read_parquet(os.path.join(WORK, "m2_oof.parquet"))
    
    # Load truth and get per-S1 scores
    A = load("train", "A").select("idx", "entity_id", "country")
    truth_df = load_truth()
    
    # Get truth counts per S1
    truth_counts = (truth_df.group_by("s1_id").agg(pl.len().alias("n_true"))
                    .join(A.select(pl.col("entity_id").alias("s1_id"), "idx", "country"), 
                          on="s1_id"))
    
    # Get predictions per S1
    # Apply decision thresholds from m2
    with open(os.path.join(WORK, "m2_decision.json"), "r") as f:
        import json
        cfg = json.load(f)
    
    # Simple threshold decision for analysis
    pred_df = oof.filter(pl.col("p") > 0.5).select("a_idx", "b_idx")
    B = load("train", "B").select("idx", "entity_id")
    
    pred_matches = pred_df.join(A.select(pl.col("idx").alias("a_idx"), pl.col("entity_id").alias("s1_id")), 
                                  on="a_idx")
    pred_matches = pred_matches.join(B.select(pl.col("idx").alias("b_idx"), pl.col("entity_id").alias("m_id")), 
                                      on="b_idx")
    
    # Compute per-S1 precision and recall
    # Convert to expected format (a_idx instead of s1_id)
    pred_for_scoring = pred_matches.select(
        pl.col("a_idx"),
        pl.col("m_id")
    ).join(A.select(pl.col("idx").alias("a_idx")), on="a_idx")
    
    truth_for_scoring = truth_df.join(
        A.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("a_idx")),
        on="s1_id"
    ).join(
        B.select(pl.col("entity_id").alias("m_id"), pl.col("idx").alias("b_idx")),
        on="m_id"
    ).select("a_idx", "b_idx")
    
    # Create labels
    pred_with_labels = pred_matches.select("a_idx", "b_idx").join(
        truth_for_scoring.with_columns(pl.lit(1).alias("y")),
        on=["a_idx", "b_idx"],
        how="left"
    ).with_columns(pl.col("y").fill_null(0))
    
    per_s1_scores = per_a_scores(pred_with_labels.select("a_idx", "b_idx", "y").rename({"b_idx": "m_id"}), 
                                  truth_counts.select("idx", "n_true"),
                                  A.select("idx", "country"))
    
    per_s1_scores = per_s1_scores.with_columns([
        (1 - pl.col("f05")).alias("error")
    ]).sort("error", descending=True)
    
    print(f"\nTotal S1 records: {per_s1_scores.height:,}")
    print(f"Perfect F0.5 (= 1.0): {(per_s1_scores['f05'] == 1.0).sum():,}")
    print(f"F0.5 >= 0.99: {(per_s1_scores['f05'] >= 0.99).sum():,}")
    print(f"F0.5 >= 0.95: {(per_s1_scores['f05'] >= 0.95).sum():,}")
    print(f"F0.5 < 0.9: {(per_s1_scores['f05'] < 0.9).sum():,}")
    print(f"F0.5 < 0.8: {(per_s1_scores['f05'] < 0.8).sum():,}")
    
    # Worst 100 cases
    worst_100 = per_s1_scores.head(100)
    print(f"\n=== WORST 100 S1 RECORDS ===")
    print(f"Mean F0.5: {worst_100['f05'].mean():.4f}")
    print(f"Mean Precision: {worst_100['precision'].mean():.4f}")
    print(f"Mean Recall: {worst_100['recall'].mean():.4f}")
    
    # Segment by country
    print(f"\n=== ERROR BY COUNTRY ===")
    for country in ["India", "US"]:
        segment = per_s1_scores.filter(pl.col("country") == country)
        print(f"\n{country}: {segment.height:,} records")
        print(f"  Mean F0.5: {segment['f05'].mean():.4f}")
        print(f"  Bottom 10% F0.5: {segment['f05'].quantile(0.1):.4f}")
        print(f"  Bottom 25% F0.5: {segment['f05'].quantile(0.25):.4f}")
    
    # Segment by truth size
    print(f"\n=== ERROR BY TRUTH SIZE ===")
    per_s1_with_size = per_s1_scores.join(truth_counts.select("idx", "n_true"), on="idx", how="left")
    per_s1_with_size = per_s1_with_size.with_columns(pl.col("n_true").fill_null(0))
    
    for bucket in [(0, 0), (1, 1), (2, 3), (4, 10), (11, 999)]:
        segment = per_s1_with_size.filter(
            (pl.col("n_true") >= bucket[0]) & (pl.col("n_true") <= bucket[1])
        )
        if segment.height > 0:
            print(f"\nTruth size {bucket[0]}-{bucket[1]}: {segment.height:,} records")
            print(f"  Mean F0.5: {segment['f05'].mean():.4f}")
            print(f"  Bottom 10% F0.5: {segment['f05'].quantile(0.1):.4f}")
    
    # Segment by prediction uncertainty (candidates in 0.1-0.9 range)
    uncertain = oof.filter((pl.col("p") > 0.1) & (pl.col("p") < 0.9)).group_by("a_idx").agg(pl.len().alias("n_uncertain"))
    per_s1_unc = per_s1_scores.join(uncertain.select(pl.col("a_idx").alias("idx"), "n_uncertain"), on="idx", how="left")
    per_s1_unc = per_s1_unc.with_columns(pl.col("n_uncertain").fill_null(0))
    
    print(f"\n=== ERROR BY UNCERTAINTY (candidates with 0.1 < p < 0.9) ===")
    for bucket in [(0, 0), (1, 2), (3, 5), (6, 999)]:
        segment = per_s1_unc.filter(
            (pl.col("n_uncertain") >= bucket[0]) & (pl.col("n_uncertain") <= bucket[1])
        )
        if segment.height > 0:
            print(f"\nUncertain candidates {bucket[0]}-{bucket[1]}: {segment.height:,} records")
            print(f"  Mean F0.5: {segment['f05'].mean():.4f}")
            print(f"  Mean uncertain count: {segment['n_uncertain'].mean():.2f}")
    
    # Save worst cases for detailed inspection
    worst_path = os.path.join(WORK, "worst_cases.parquet")
    worst_100_with_data = worst_100.join(A.select(pl.col("idx"), pl.col("entity_id").alias("s1_id")), on="idx")
    worst_100_with_data.write_parquet(worst_path)
    print(f"\nSaved worst 100 cases to: {worst_path}")
    
    # Analyze false negatives (missed matches) in worst cases
    worst_s1_ids = worst_100_with_data["s1_id"].to_list()
    truth_in_worst = truth_df.filter(pl.col("s1_id").is_in(worst_s1_ids))
    
    # Get all candidate pairs for worst S1
    worst_a_idx = worst_100["idx"].to_list()
    candidates_worst = oof.filter(pl.col("a_idx").is_in(worst_a_idx))
    
    # Join with truth to identify FN and FP
    truth_pairs_worst = truth_in_worst.join(A.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("a_idx")), 
                                             on="s1_id")
    truth_pairs_worst = truth_pairs_worst.join(B.select(pl.col("entity_id").alias("m_id"), pl.col("idx").alias("b_idx")), 
                                                on="m_id")
    truth_pairs_worst = truth_pairs_worst.select("a_idx", "b_idx").with_columns(pl.lit(True).alias("is_true"))
    
    candidates_worst = candidates_worst.join(truth_pairs_worst, on=["a_idx", "b_idx"], how="left")
    candidates_worst = candidates_worst.with_columns(pl.col("is_true").fill_null(False))
    
    # False negatives: true pairs with low score
    fn_pairs = candidates_worst.filter(pl.col("is_true") & (pl.col("p") < 0.5))
    print(f"\n=== FALSE NEGATIVES IN WORST 100 ===")
    print(f"Total FN pairs: {fn_pairs.height:,}")
    if fn_pairs.height > 0:
        print(f"FN score distribution:")
        print(f"  Mean p: {fn_pairs['p'].mean():.4f}")
        print(f"  Median p: {fn_pairs['p'].median():.4f}")
        print(f"  p < 0.1: {(fn_pairs['p'] < 0.1).sum():,}")
        print(f"  0.1 <= p < 0.3: {((fn_pairs['p'] >= 0.1) & (fn_pairs['p'] < 0.3)).sum():,}")
        print(f"  0.3 <= p < 0.5: {((fn_pairs['p'] >= 0.3) & (fn_pairs['p'] < 0.5)).sum():,}")
    
    return per_s1_scores, worst_100, candidates_worst


# ============================================================================
# 2. FEATURE AUDIT
# ============================================================================

def feature_audit():
    """Analyze feature importance and correlation with target."""
    print("\n" + "=" * 80)
    print("2. FEATURE AUDIT - Correlation & Importance")
    print("=" * 80)
    
    # Load features and labels
    oof = pl.read_parquet(os.path.join(WORK, "m2_oof.parquet"))
    tp = truth_pairs().select(ID_COLS).with_columns(pl.lit(1, pl.Int8).alias("y"))
    
    # Sample for faster correlation computation
    sample_size = min(100_000, oof.height)
    sampled_indices = np.random.choice(oof.height, sample_size, replace=False)
    
    # Load one chunk of features to get all feature names
    feat_files = [os.path.join(WORK, f) for f in os.listdir(WORK) if f.startswith("feat_train_") and f.endswith(".parquet")]
    cluster_file = os.path.join(WORK, "cluster_train_m2.parquet")
    
    feat_df = pl.read_parquet(feat_files[0])
    cluster_df = pl.read_parquet(cluster_file)
    
    # Merge with cluster features
    feat_df = feat_df.join(cluster_df, on=ID_COLS, how="left")
    feat_df = feat_df.join(tp, on=ID_COLS, how="left").with_columns(pl.col("y").fill_null(0))
    
    # Get feature names
    feat_cols = [c for c in feat_df.columns if c not in ID_COLS + ["y"]]
    
    print(f"\nTotal features: {len(feat_cols)}")
    
    # Load models to get feature importance
    with open(os.path.join(WORK, "m2_models.pkl"), "rb") as f:
        models = pickle.load(f)
    
    # Average feature importance across folds
    feat_importance = {}
    for feat in feat_cols:
        importance_values = []
        for model in models:
            try:
                idx = model.feature_name().index(feat)
                importance_values.append(model.feature_importance("gain")[idx])
            except (ValueError, IndexError):
                pass
        if importance_values:
            feat_importance[feat] = np.mean(importance_values)
        else:
            feat_importance[feat] = 0.0
    
    # Sort by importance
    feat_importance_sorted = sorted(feat_importance.items(), key=lambda x: -x[1])
    
    print("\n=== TOP 30 FEATURES BY IMPORTANCE ===")
    for i, (feat, imp) in enumerate(feat_importance_sorted[:30], 1):
        print(f"{i:2d}. {feat:40s} {imp:12,.0f}")
    
    print("\n=== BOTTOM 20 FEATURES BY IMPORTANCE ===")
    for i, (feat, imp) in enumerate(feat_importance_sorted[-20:], 1):
        print(f"{i:2d}. {feat:40s} {imp:12,.0f}")
    
    # Compute correlations on a sample
    print(f"\n=== COMPUTING CORRELATIONS ON SAMPLE (n={sample_size:,}) ===")
    sample_df = feat_df.sample(n=sample_size, seed=42)
    
    y = sample_df["y"].to_numpy()
    correlations = {}
    
    for feat in feat_cols:
        x = sample_df[feat].to_numpy()
        # Handle NaN/inf
        valid_mask = np.isfinite(x)
        if valid_mask.sum() > 100:
            corr, _ = spearmanr(x[valid_mask], y[valid_mask])
            if np.isfinite(corr):
                correlations[feat] = abs(corr)
            else:
                correlations[feat] = 0.0
        else:
            correlations[feat] = 0.0
    
    correlations_sorted = sorted(correlations.items(), key=lambda x: -x[1])
    
    print("\n=== TOP 30 FEATURES BY |CORRELATION| WITH TARGET ===")
    for i, (feat, corr) in enumerate(correlations_sorted[:30], 1):
        imp = feat_importance.get(feat, 0)
        print(f"{i:2d}. {feat:40s} |corr|={corr:.4f}  importance={imp:10,.0f}")
    
    # Save audit results
    audit_df = pl.DataFrame({
        "feature": list(feat_importance.keys()),
        "importance": list(feat_importance.values()),
        "correlation": [correlations.get(f, 0.0) for f in feat_importance.keys()]
    }).sort("importance", descending=True)
    
    audit_path = os.path.join(WORK, "feature_audit.parquet")
    audit_df.write_parquet(audit_path)
    print(f"\nSaved feature audit to: {audit_path}")
    
    return feat_importance_sorted, correlations_sorted


# ============================================================================
# 3. NEW FEATURE ENGINEERING
# ============================================================================

def engineer_new_features():
    """Create new advanced features."""
    print("\n" + "=" * 80)
    print("3. NEW FEATURE ENGINEERING")
    print("=" * 80)
    
    print("\nNew features to add:")
    print("1. Interaction terms: n_tset * a_tset, num_jac * n_tset, etc.")
    print("2. Ratio features: name_to_address_quality ratios")
    print("3. Target encoding: mean target by country, by core name frequency bucket")
    print("4. Aggregations: std/max/min of similarities across S1 candidates")
    print("5. Text length ratios and differences")
    print("6. Competition-based: rank differences, score gaps")
    print("7. Null/missing indicators")
    print("8. Higher-order interactions from cluster features")
    
    print("\n>>> Implementation in engineer_features.py")
    
    # Return placeholder - actual implementation follows
    return []


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\nStarting comprehensive analysis...")
    print(f"Working directory: {WORK}\n")
    
    # Run error analysis
    per_s1_scores, worst_100, candidates_worst = error_analysis()
    
    # Run feature audit
    feat_importance, correlations = feature_audit()
    
    # New features (implementation next)
    engineer_new_features()
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review worst_cases.parquet and feature_audit.parquet")
    print("2. Implement new features in engineer_features.py")
    print("3. Run hyperparameter optimization")
    print("4. Build ensemble models")
    print("5. Add pseudo-labeling")
