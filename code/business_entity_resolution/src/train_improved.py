"""Train improved model with new features and optimized hyperparameters."""
from __future__ import annotations

import glob
import os
import pickle
import sys
import time

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(__file__))

from io_utils import WORK
from model import ID_COLS, truth_pairs, fold_of, in_train_sample, N_FOLDS

# Improved hyperparameters (based on domain knowledge and typical best practices)
IMPROVED_PARAMS = dict(
    objective="binary",
    learning_rate=0.04,  # Slower for better convergence
    num_leaves=511,  # Increased capacity
    min_data_in_leaf=50,  # Lower for finer splits
    feature_fraction=0.7,  # More regularization
    bagging_fraction=0.7,
    bagging_freq=1,
    lambda_l1=0.5,  # Add L1 regularization
    lambda_l2=2.0,  # Stronger L2
    max_bin=255,  # More bins for better splits
    min_gain_to_split=0.01,  # Require meaningful gains
    num_threads=10,
    verbose=-1
)

print("=" * 80)
print("TRAINING IMPROVED MODEL WITH NEW FEATURES")
print("=" * 80)


def load_improved_features(split: str):
    """Load features from feat_v2 directory."""
    
    pattern = f"feat_v2_{split}"
    feat_dir = os.path.join(WORK, pattern)
    
    if not os.path.exists(feat_dir):
        print(f"ERROR: {feat_dir} not found. Run new_features.py first.")
        return None
    
    files = sorted(glob.glob(os.path.join(feat_dir, "*.parquet")))
    print(f"\nLoading {len(files)} feature files from {pattern}...")
    
    return files


def train_with_improved_features(tag="m4_improved"):
    """Train model with new features."""
    
    print(f"\nModel tag: {tag}")
    
    # Load feature files
    feat_files = load_improved_features("train")
    if not feat_files:
        return None
    
    # Load truth
    truth = truth_pairs()
    tp = truth.select(ID_COLS).with_columns(pl.lit(1, pl.Int8).alias("y"))
    
    # Load cluster features
    cluster_file = os.path.join(WORK, "cluster_train_m2.parquet")
    if not os.path.exists(cluster_file):
        print(f"WARNING: {cluster_file} not found, training without cluster features")
        cluster_df = None
    else:
        cluster_df = pl.read_parquet(cluster_file)
        print(f"Loaded cluster features: {cluster_df.shape}")
    
    # Determine sample size and prepare data
    print("\nPreparing training data...")
    dfs = []
    for f in feat_files:
        df = pl.read_parquet(f)
        # Apply sampling
        df = df.filter(in_train_sample(pl.col("a_idx")))
        dfs.append(df)
    
    df = pl.concat(dfs)
    print(f"Base features loaded: {df.shape}")
    
    # Add cluster features
    if cluster_df is not None:
        df = df.join(cluster_df, on=ID_COLS, how="left")
        print(f"After cluster join: {df.shape}")
    
    # Add labels and folds
    df = df.join(tp, on=ID_COLS, how="left").with_columns(
        pl.col("y").fill_null(0),
        fold_of(pl.col("a_idx")).alias("fold")
    )
    
    # Get feature names
    feats = [c for c in df.columns if c not in ID_COLS + ["y", "fold"]]
    print(f"\nTotal features: {len(feats)}")
    print(f"Training samples: {df.height:,}")
    print(f"Positive rate: {df['y'].mean():.4f}")
    
    # Convert to numpy
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float32)
    fold = df["fold"].to_numpy().astype(np.int8)
    
    del df, dfs  # Free memory
    
    # Create LightGBM dataset once
    print("\nCreating LightGBM dataset...")
    full_dataset = lgb.Dataset(X, y, feature_name=feats, free_raw_data=False,
                                params={"max_bin": IMPROVED_PARAMS["max_bin"], "verbose": -1})
    full_dataset.construct()
    
    # Train folds
    print("\nTraining folds...")
    models = []
    oof_preds = np.zeros(len(y), dtype=np.float32)
    
    for k in range(N_FOLDS):
        print(f"\n{'=' * 60}")
        print(f"FOLD {k}")
        print('=' * 60)
        
        t_start = time.time()
        
        train_idx = np.where(fold != k)[0]
        val_idx = np.where(fold == k)[0]
        
        dtrain = full_dataset.subset(train_idx)
        dval = full_dataset.subset(val_idx)
        
        model = lgb.train(
            IMPROVED_PARAMS,
            dtrain,
            num_boost_round=6000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                lgb.log_evaluation(200)
            ]
        )
        
        # OOF predictions
        oof_preds[val_idx] = model.predict(X[val_idx], num_iteration=model.best_iteration)
        
        # Compute fold metrics
        val_y = y[val_idx]
        val_pred = oof_preds[val_idx]
        logloss = -np.mean(val_y * np.log(np.clip(val_pred, 1e-15, 1-1e-15)) + 
                           (1-val_y) * np.log(np.clip(1-val_pred, 1e-15, 1-1e-15)))
        
        elapsed = time.time() - t_start
        print(f"\nFold {k} complete:")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Validation logloss: {logloss:.6f}")
        print(f"  Time: {elapsed:.0f}s")
        
        models.append(model)
    
    # Overall OOF metrics
    print("\n" + "=" * 80)
    print("OVERALL OOF PERFORMANCE")
    print("=" * 80)
    
    oof_logloss = -np.mean(y * np.log(np.clip(oof_preds, 1e-15, 1-1e-15)) + 
                            (1-y) * np.log(np.clip(1-oof_preds, 1e-15, 1-1e-15)))
    print(f"OOF LogLoss: {oof_logloss:.6f}")
    
    # Threshold metrics
    for thresh in [0.5, 0.7, 0.9]:
        pred_pos = (oof_preds >= thresh).sum()
        true_pos = ((oof_preds >= thresh) & (y == 1)).sum()
        precision = true_pos / pred_pos if pred_pos > 0 else 0
        recall = true_pos / y.sum()
        print(f"Threshold {thresh}: Precision={precision:.4f}, Recall={recall:.4f}, Pred={pred_pos:,}")
    
    # Feature importance
    print("\n" + "=" * 80)
    print("TOP 50 FEATURES BY IMPORTANCE")
    print("=" * 80)
    
    avg_importance = np.mean([m.feature_importance("gain") for m in models], axis=0)
    feat_imp = sorted(zip(feats, avg_importance), key=lambda x: -x[1])
    
    for i, (feat, imp) in enumerate(feat_imp[:50], 1):
        print(f"{i:3d}. {feat:50s} {imp:15,.0f}")
    
    # Calibration
    print("\nApplying isotonic calibration...")
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1)
    iso.fit(oof_preds, y)
    oof_calibrated = iso.predict(oof_preds)
    
    cal_logloss = -np.mean(y * np.log(np.clip(oof_calibrated, 1e-15, 1-1e-15)) + 
                            (1-y) * np.log(np.clip(1-oof_calibrated, 1e-15, 1-1e-15)))
    print(f"Calibrated OOF LogLoss: {cal_logloss:.6f}")
    
    # Save models
    model_path = os.path.join(WORK, f"{tag}_models.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(models, f)
    print(f"\nModels saved to: {model_path}")
    
    # Save ISO
    iso_path = os.path.join(WORK, f"{tag}_iso.pkl")
    with open(iso_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"Calibration saved to: {iso_path}")
    
    # Save OOF predictions
    oof_df = pl.DataFrame({
        "a_idx": df["a_idx"].to_numpy() if "a_idx" in locals() else np.arange(len(y)),
        "b_idx": df["b_idx"].to_numpy() if "b_idx" in locals() else np.arange(len(y)),
        "p_raw": oof_preds,
        "p": oof_calibrated,
        "y": y
    })
    
    # Need to reload indices since we deleted df
    print("\nReloading indices for OOF save...")
    idx_df = pl.concat([pl.read_parquet(f).select(ID_COLS).filter(in_train_sample(pl.col("a_idx"))) 
                        for f in feat_files])
    
    oof_final = pl.DataFrame({
        "a_idx": idx_df["a_idx"].to_numpy(),
        "b_idx": idx_df["b_idx"].to_numpy(),
        "p_raw": oof_preds.astype(np.float32),
        "p": oof_calibrated.astype(np.float32),
    })
    
    oof_path = os.path.join(WORK, f"{tag}_oof.parquet")
    oof_final.write_parquet(oof_path)
    print(f"OOF predictions saved to: {oof_path}")
    
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    
    return models, iso


if __name__ == "__main__":
    train_with_improved_features()
