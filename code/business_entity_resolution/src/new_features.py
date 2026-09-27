"""New advanced features to close the performance gap.

Based on error analysis:
- Add interaction terms between top features
- Target encoding (leak-free via out-of-fold)
- Ratio features (name quality / address quality)
- Per-S1 aggregation statistics
- Remove zero-importance features
"""
from __future__ import annotations

import os
import numpy as np
import polars as pl
from typing import Dict

from io_utils import WORK
from model import ID_COLS, truth_pairs, fold_of, in_train_sample


def add_interaction_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add multiplicative interaction features between top predictors."""
    
    interactions = []
    
    # Interactions with p1 (most important feature)
    if "p1" in df.columns:
        for feat in ["margin_b", "margin_a", "pother_b", "pother_a", "psum_b", "n50_a"]:
            if feat in df.columns:
                interactions.append(
                    (pl.col("p1") * pl.col(feat)).alias(f"p1_x_{feat}")
                )
    
    # String similarity interactions
    if "n_tset" in df.columns and "a_tset" in df.columns:
        interactions.append((pl.col("n_tset") * pl.col("a_tset")).alias("name_addr_qual"))
        interactions.append((pl.col("n_tset") + pl.col("a_tset")).alias("total_sim"))
    
    if "n_ratio" in df.columns and "num_jac" in df.columns:
        interactions.append((pl.col("n_ratio") * pl.col("num_jac")).alias("name_num_conf"))
    
    # Margin interactions (competition context)
    if "margin_a" in df.columns and "margin_b" in df.columns:
        interactions.append((pl.col("margin_a") * pl.col("margin_b")).alias("margin_product"))
        interactions.append(pl.min_horizontal("margin_a", "margin_b").alias("margin_min"))
        interactions.append((pl.col("margin_a") + pl.col("margin_b")).alias("margin_sum"))
    
    # Rank interactions
    if "rank_a" in df.columns and "rank_b" in df.columns:
        interactions.append((pl.col("rank_a") + pl.col("rank_b")).alias("rank_sum"))
        interactions.append(pl.max_horizontal("rank_a", "rank_b").alias("rank_max"))
    
    # Weighted overlap interactions
    if "nt_wjac" in df.columns and "at_wjac" in df.columns:
        interactions.append((pl.col("nt_wjac") * pl.col("at_wjac")).alias("name_addr_wjac"))
        interactions.append(
            (pl.col("nt_wjac") / (pl.col("at_wjac") + 0.01)).alias("name_addr_ratio")
        )
    
    # Copy count interactions (competitor features)
    if "own_same_src" in df.columns and "comp_same_src" in df.columns:
        interactions.append(
            ((pl.col("own_same_src") + 1) / (pl.col("comp_same_src") + 1)).alias("copy_count_ratio")
        )
    
    # IDF-based features
    if "at_maxidf_un_a" in df.columns and "at_maxidf_un_b" in df.columns:
        interactions.append(
            pl.max_horizontal("at_maxidf_un_a", "at_maxidf_un_b").alias("max_unmatched_idf")
        )
    
    if interactions:
        df = df.with_columns(interactions)
    
    return df


def add_ratio_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add ratio features comparing different aspects."""
    
    ratios = []
    
    # Name to address quality ratios
    if "n_tset" in df.columns and "a_tset" in df.columns:
        ratios.append(
            (pl.col("n_tset") / (pl.col("a_tset") + 0.01)).alias("name_to_addr_quality")
        )
    
    if "nt_wjac" in df.columns and "num_jac" in df.columns:
        ratios.append(
            (pl.col("nt_wjac") / (pl.col("num_jac") + 0.01)).alias("name_to_num_quality")
        )
    
    # Coverage ratios
    if "nt_wcov_a" in df.columns and "nt_wcov_b" in df.columns:
        ratios.append(
            pl.min_horizontal("nt_wcov_a", "nt_wcov_b").alias("min_name_coverage")
        )
        ratios.append(
            (pl.col("nt_wcov_a") / (pl.col("nt_wcov_b") + 0.01)).alias("name_coverage_ratio")
        )
    
    # Length ratios
    if "len_a" in df.columns and "len_b" in df.columns:
        ratios.append(
            (pl.col("len_a") / (pl.col("len_b") + 1)).alias("len_ratio")
        )
        ratios.append(
            (pl.col("len_a") - pl.col("len_b")).abs().alias("len_diff")
        )
    
    # Frequency ratios
    if "corefreq_a" in df.columns and "corefreq_b" in df.columns:
        ratios.append(
            (pl.col("corefreq_a") - pl.col("corefreq_b")).abs().alias("freq_diff")
        )
    
    # Token count ratios
    if "nt_na" in df.columns and "nt_nb" in df.columns:
        ratios.append(
            pl.min_horizontal("nt_na", "nt_nb").alias("min_name_tokens")
        )
        ratios.append(
            (pl.col("nt_na") / (pl.col("nt_nb") + 1)).alias("name_token_ratio")
        )
    
    if ratios:
        df = df.with_columns(ratios)
    
    return df


def add_aggregation_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add per-S1 and per-B aggregation statistics."""
    
    aggs = []
    
    # For features that benefit from knowing distribution across candidates
    agg_features = ["n_tset", "a_tset", "num_jac", "nt_wjac", "at_wjac", "n_ratio", "c_ratio"]
    
    for feat in agg_features:
        if feat in df.columns:
            # Per S1 statistics
            aggs.append(pl.col(feat).std().over("a_idx").alias(f"{feat}_std_a"))
            aggs.append(pl.col(feat).max().over("a_idx").alias(f"{feat}_max_a"))
            aggs.append((pl.col(feat) - pl.col(feat).mean().over("a_idx")).alias(f"{feat}_demean_a"))
            
            # Rank within S1
            aggs.append(
                pl.col(feat).rank("dense", descending=True).over("a_idx").alias(f"{feat}_rank_a")
            )
    
    if aggs:
        df = df.with_columns(aggs)
    
    return df


def add_confidence_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add confidence and uncertainty indicators."""
    
    conf = []
    
    # High confidence indicators
    if "p1" in df.columns:
        conf.append((pl.col("p1") > 0.95).cast(pl.Float32).alias("very_high_conf"))
        conf.append((pl.col("p1") < 0.05).cast(pl.Float32).alias("very_low_conf"))
        conf.append(((pl.col("p1") >= 0.3) & (pl.col("p1") <= 0.7)).cast(pl.Float32).alias("uncertain"))
    
    # Perfect match indicators
    if "n_tset" in df.columns:
        conf.append((pl.col("n_tset") == 100).cast(pl.Float32).alias("perfect_name"))
    
    if "num_first_eq" in df.columns and "a_tset" in df.columns:
        conf.append(
            ((pl.col("num_first_eq") == 1) & (pl.col("a_tset") > 80)).cast(pl.Float32).alias("same_num_addr")
        )
    
    # Conflict indicators
    if "legal_conf" in df.columns and "n_tset" in df.columns:
        conf.append(
            ((pl.col("legal_conf") == 1) & (pl.col("n_tset") > 90)).cast(pl.Float32).alias("legal_conflict_high_sim")
        )
    
    if conf:
        df = df.with_columns(conf)
    
    return df


def compute_target_encoding(split: str) -> Dict[str, pl.DataFrame]:
    """Compute out-of-fold target encoding for categorical features.
    
    Returns dictionaries mapping (category, fold) -> mean_target for leak-free encoding.
    """
    if split != "train":
        # For test, use full training statistics
        return {}
    
    print("\n=== COMPUTING TARGET ENCODING (OUT-OF-FOLD) ===")
    
    from prepare import load
    from io_utils import load_truth
    
    A = load("train", "A").select("idx", "country")
    B = load("train", "B").select("idx")
    
    # Get truth pairs
    truth_df = load_truth()
    truth_pairs_df = (truth_df
                      .join(A.select(pl.col("idx").alias("a_idx")), 
                            left_on=pl.col("entity_id"), right_on=A["entity_id"])
                      .join(B.select(pl.col("idx").alias("b_idx")), 
                            left_on=pl.col("m_id"), right_on=B["entity_id"])
                      .select("a_idx", "b_idx").with_columns(pl.lit(1).alias("y")))
    
    # For each fold, compute statistics on other folds
    encoding_maps = {}
    
    # Country encoding (simple since only 2 countries in train)
    country_encoding = []
    for fold in range(3):
        # Use other folds for encoding
        other_folds = A.filter(fold_of(pl.col("idx")) != fold)
        other_truth = truth_pairs_df.filter(fold_of(pl.col("a_idx")) != fold)
        
        # Compute mean target by country
        country_stats = (other_folds.join(A.select("idx", "country"), left_on="idx", right_on="idx")
                         .join(other_truth, left_on="idx", right_on="a_idx", how="left")
                         .with_columns(pl.col("y").fill_null(0))
                         .group_by("country")
                         .agg(pl.col("y").mean().alias("country_target_mean")))
        
        country_encoding.append((fold, country_stats))
    
    encoding_maps["country"] = country_encoding
    
    # Frequency bucket encoding
    A_with_freq = A.join(
        A.group_by("entity_id").agg(pl.len().alias("freq")),
        left_on="entity_id", right_on="entity_id"
    )
    
    # Create buckets: [1], [2-3], [4-10], [11+]
    A_with_freq = A_with_freq.with_columns(
        pl.when(pl.col("freq") == 1).then(pl.lit("freq_1"))
        .when(pl.col("freq") <= 3).then(pl.lit("freq_2_3"))
        .when(pl.col("freq") <= 10).then(pl.lit("freq_4_10"))
        .otherwise(pl.lit("freq_11plus"))
        .alias("freq_bucket")
    )
    
    freq_encoding = []
    for fold in range(3):
        other_folds = A_with_freq.filter(fold_of(pl.col("idx")) != fold)
        other_truth = truth_pairs_df.filter(fold_of(pl.col("a_idx")) != fold)
        
        freq_stats = (other_folds.join(other_truth, left_on="idx", right_on="a_idx", how="left")
                      .with_columns(pl.col("y").fill_null(0))
                      .group_by("freq_bucket")
                      .agg(pl.col("y").mean().alias("freq_target_mean")))
        
        freq_encoding.append((fold, freq_stats))
    
    encoding_maps["freq_bucket"] = freq_encoding
    
    print(f"Target encoding computed for {len(encoding_maps)} categorical features")
    
    return encoding_maps


def apply_new_features(split: str, tag: str) -> None:
    """Apply all new features to existing feature files."""
    
    print(f"\n=== APPLYING NEW FEATURES TO {split.upper()} ===")
    
    import glob
    feat_files = sorted(glob.glob(os.path.join(WORK, f"feat_{split}_*.parquet")))
    
    if not feat_files:
        print(f"No feature files found for {split}")
        return
    
    # Get target encoding
    # target_encoding = compute_target_encoding(split)
    
    output_dir = os.path.join(WORK, f"feat_v2_{split}")
    os.makedirs(output_dir, exist_ok=True)
    
    for i, feat_file in enumerate(feat_files):
        print(f"Processing {i+1}/{len(feat_files)}: {os.path.basename(feat_file)}")
        
        df = pl.read_parquet(feat_file)
        
        # Add new features
        df = add_interaction_features(df)
        df = add_ratio_features(df)
        df = add_aggregation_features(df)
        df = add_confidence_features(df)
        
        # Remove zero-importance features
        zero_importance = ["alt_a", "alt_b", "aempty_b", "c_contain", "no_comp"]
        df = df.drop([col for col in zero_importance if col in df.columns])
        
        # Save
        output_path = os.path.join(output_dir, os.path.basename(feat_file))
        df.write_parquet(output_path)
    
    print(f"New features saved to: {output_dir}")
    print(f"Feature count increased by ~{len(df.columns) - len(pl.read_parquet(feat_files[0]).columns)}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python new_features.py <train|test>")
        sys.exit(1)
    
    split = sys.argv[1]
    apply_new_features(split, "m2")
