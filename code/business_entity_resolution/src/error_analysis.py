"""Direct error analysis on OOF predictions."""
from __future__ import annotations

import os
import sys
import pickle
import json

import numpy as np
import polars as pl
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(__file__))

from io_utils import WORK, load_truth
from prepare import load

print("=" * 80)
print("ERROR ANALYSIS - Current Model Performance")
print("=" * 80)

# Load data
print("\nLoading data...")
A = load("train", "A").select("idx", "entity_id", "country")
B = load("train", "B").select("idx", "entity_id")
truth_df = load_truth()

# Load OOF predictions
oof = pl.read_parquet(os.path.join(WORK, "m2_oof.parquet"))
print(f"OOF pairs: {oof.height:,}")

# Create truth pairs
truth_pairs = (truth_df
               .join(A.select(pl.col("entity_id").alias("s1_id"), pl.col("idx").alias("a_idx")), on="s1_id")
               .join(B.select(pl.col("entity_id").alias("m_id"), pl.col("idx").alias("b_idx")), on="m_id")
               .select("a_idx", "b_idx").with_columns(pl.lit(True).alias("is_match")))

print(f"Truth pairs: {truth_pairs.height:,}")

# Join OOF with truth
oof_labeled = oof.join(truth_pairs, on=["a_idx", "b_idx"], how="left")
oof_labeled = oof_labeled.with_columns(pl.col("is_match").fill_null(False))

n_pos = oof_labeled["is_match"].sum()
n_neg = oof_labeled.height - n_pos
print(f"Positive pairs: {n_pos:,}")
print(f"Negative pairs: {n_neg:,}")

#============================================================================
# 1. PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("1. PREDICTION SCORE DISTRIBUTION")
print("=" * 80)

print("\n=== ALL PAIRS ===")
print(f"Mean p: {oof_labeled['p'].mean():.4f}")
print(f"Median p: {oof_labeled['p'].median():.4f}")
print(f"Std p: {oof_labeled['p'].std():.4f}")
print(f"p < 0.1: {(oof_labeled['p'] < 0.1).sum():,} ({100 * (oof_labeled['p'] < 0.1).sum() / oof_labeled.height:.1f}%)")
print(f"0.1 <= p < 0.5: {((oof_labeled['p'] >= 0.1) & (oof_labeled['p'] < 0.5)).sum():,}")
print(f"0.5 <= p < 0.9: {((oof_labeled['p'] >= 0.5) & (oof_labeled['p'] < 0.9)).sum():,}")
print(f"p >= 0.9: {(oof_labeled['p'] >= 0.9).sum():,}")

print("\n=== TRUE MATCHES (should be high) ===")
true_matches = oof_labeled.filter(pl.col("is_match"))
print(f"Mean p: {true_matches['p'].mean():.4f}")
print(f"Median p: {true_matches['p'].median():.4f}")
print(f"p < 0.5: {(true_matches['p'] < 0.5).sum():,} ({100 * (true_matches['p'] < 0.5).sum() / true_matches.height:.2f}%) ← FALSE NEGATIVES")
print(f"p < 0.3: {(true_matches['p'] < 0.3).sum():,}")
print(f"p < 0.1: {(true_matches['p'] < 0.1).sum():,}")

print("\n=== FALSE MATCHES (should be low) ===")
false_matches = oof_labeled.filter(~pl.col("is_match"))
print(f"Mean p: {false_matches['p'].mean():.4f}")
print(f"Median p: {false_matches['p'].median():.4f}")
print(f"p >= 0.5: {(false_matches['p'] >= 0.5).sum():,} ({100 * (false_matches['p'] >= 0.5).sum() / false_matches.height:.2f}%) ← FALSE POSITIVES")
print(f"p >= 0.7: {(false_matches['p'] >= 0.7).sum():,}")
print(f"p >= 0.9: {(false_matches['p'] >= 0.9).sum():,}")

# ============================================================================
# 2. ERROR BY COUNTRY
# ============================================================================
print("\n" + "=" * 80)
print("2. ERROR BY COUNTRY")
print("=" * 80)

oof_with_country = oof_labeled.join(A.select(pl.col("idx").alias("a_idx"), "country"), on="a_idx")

for country in ["India", "US"]:
    country_pairs = oof_with_country.filter(pl.col("country") == country)
    country_true = country_pairs.filter(pl.col("is_match"))
    country_false = country_pairs.filter(~pl.col("is_match"))
    
    print(f"\n=== {country} ===")
    print(f"Total pairs: {country_pairs.height:,}")
    print(f"True matches: {country_true.height:,}")
    print(f"  Mean p: {country_true['p'].mean():.4f}")
    print(f"  FN (p < 0.5): {(country_true['p'] < 0.5).sum():,} ({100 * (country_true['p'] < 0.5).sum() / country_true.height:.2f}%)")
    print(f"False matches: {country_false.height:,}")
    print(f"  Mean p: {country_false['p'].mean():.4f}")
    print(f"  FP (p >= 0.5): {(country_false['p'] >= 0.5).sum():,} ({100 * (country_false['p'] >= 0.5).sum() / country_false.height:.2f}%)")

# ============================================================================
# 3. HARDEST CASES - FALSE NEGATIVES
# ============================================================================
print("\n" + "=" * 80)
print("3. HARDEST FALSE NEGATIVES (true matches with low scores)")
print("=" * 80)

fn_cases = oof_labeled.filter(pl.col("is_match") & (pl.col("p") < 0.5)).sort("p")
print(f"\nTotal FN cases: {fn_cases.height:,}")
print(f"Showing hardest 20...")

if fn_cases.height > 0:
    # Load feature file to inspect
    feat_sample = pl.read_parquet(os.path.join(WORK, "feat_train_India_000.parquet"))
    cluster_sample = pl.read_parquet(os.path.join(WORK, "cluster_train_m2.parquet"))
    
    feat_sample = feat_sample.join(cluster_sample, on=["a_idx", "b_idx"], how="left")
    
    fn_with_features = fn_cases.head(20).join(feat_sample, on=["a_idx", "b_idx"], how="left")
    fn_with_features = fn_with_features.join(A.select(pl.col("idx").alias("a_idx"), "country"), on="a_idx")
    
    # Show key features
    key_features = ["p", "country", "n_tset", "n_ratio", "a_tset", "num_jac", "num_inter", 
                    "nt_wjac", "legal_eq", "c_ratio", "p1", "margin_a", "margin_b"]
    
    available_features = [f for f in key_features if f in fn_with_features.columns]
    print(f"\nHardest 20 FN pairs (key features):")
    print(fn_with_features.select(["a_idx", "b_idx"] + available_features))

# ============================================================================
# 4. HARDEST CASES - FALSE POSITIVES  
# ============================================================================
print("\n" + "=" * 80)
print("4. HARDEST FALSE POSITIVES (false matches with high scores)")
print("=" * 80)

fp_cases = oof_labeled.filter(~pl.col("is_match") & (pl.col("p") >= 0.5)).sort("p", descending=True)
print(f"\nTotal FP cases: {fp_cases.height:,}")
print(f"Showing hardest 20...")

if fp_cases.height > 0:
    fp_with_features = fp_cases.head(20).join(feat_sample, on=["a_idx", "b_idx"], how="left")
    fp_with_features = fp_with_features.join(A.select(pl.col("idx").alias("a_idx"), "country"), on="a_idx")
    
    print(f"\nHardest 20 FP pairs (key features):")
    print(fp_with_features.select(["a_idx", "b_idx"] + available_features))

# ============================================================================
# 5. FEATURE IMPORTANCE
# ============================================================================
print("\n" + "=" * 80)
print("5. FEATURE IMPORTANCE ANALYSIS")
print("=" * 80)

# Load models
with open(os.path.join(WORK, "m2_models.pkl"), "rb") as f:
    models = pickle.load(f)

# Average feature importance
feat_names = models[0].feature_name()
importances = np.mean([m.feature_importance("gain") for m in models], axis=0)

feat_importance = sorted(zip(feat_names, importances), key=lambda x: -x[1])

print("\n=== TOP 40 FEATURES BY IMPORTANCE ===")
for i, (feat, imp) in enumerate(feat_importance[:40], 1):
    print(f"{i:3d}. {feat:45s} {imp:12,.0f}")

print("\n=== BOTTOM 15 FEATURES (candidates for removal) ===")
for i, (feat, imp) in enumerate(feat_importance[-15:], 1):
    print(f"{i:3d}. {feat:45s} {imp:12,.0f}")

# Save detailed results
audit_df = pl.DataFrame({
    "feature": [f[0] for f in feat_importance],
    "importance": [f[1] for f in feat_importance]
})
audit_path = os.path.join(WORK, "feature_importance.parquet")
audit_df.write_parquet(audit_path)
print(f"\nSaved feature importance to: {audit_path}")

# ============================================================================
# 6. UNCERTAINTY ANALYSIS
# ============================================================================
print("\n" + "=" * 80)
print("6. UNCERTAINTY ANALYSIS (pairs in 0.3-0.7 range)")
print("=" * 80)

uncertain = oof_labeled.filter((pl.col("p") >= 0.3) & (pl.col("p") <= 0.7))
print(f"\nUncertain pairs: {uncertain.height:,} ({100 * uncertain.height / oof_labeled.height:.2f}%)")
print(f"True matches: {uncertain['is_match'].sum():,} ({100 * uncertain['is_match'].sum() / uncertain.height:.1f}%)")
print(f"False matches: {(~uncertain['is_match']).sum():,}")

# Count uncertain per S1
uncertain_per_s1 = uncertain.group_by("a_idx").agg(
    pl.len().alias("n_uncertain"),
    pl.col("is_match").sum().alias("n_true_uncertain")
)

print(f"\nS1 records with uncertain candidates: {uncertain_per_s1.height:,}")
print(f"Mean uncertain per S1: {uncertain_per_s1['n_uncertain'].mean():.2f}")
print(f"Max uncertain per S1: {uncertain_per_s1['n_uncertain'].max()}")

high_uncertainty_s1 = uncertain_per_s1.filter(pl.col("n_uncertain") >= 5).sort("n_uncertain", descending=True)
print(f"\nS1 with >= 5 uncertain candidates: {high_uncertainty_s1.height:,}")
if high_uncertainty_s1.height > 0:
    print(high_uncertainty_s1.head(10))

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
print(f"\nResults saved to: {WORK}/")
print("Next: Review patterns and engineer new features to address weaknesses")
