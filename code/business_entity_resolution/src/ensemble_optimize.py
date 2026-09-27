"""Hyperparameter optimization and ensemble modeling.

Uses Optuna for hyperparameter search and builds ensemble of:
- LightGBM (optimized)
- XGBoost
- CatBoost

Then blends predictions for final boost in performance.
"""
from __future__ import annotations

import os
import pickle
import sys
import time
from typing import List, Dict, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression

# Check if additional libraries are available
try:
    import optuna
    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False
    print("WARNING: optuna not installed. Run: pip install optuna")

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARNING: xgboost not installed. Run: pip install xgboost")

try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("WARNING: catboost not installed. Run: pip install catboost")

sys.path.insert(0, os.path.dirname(__file__))

from io_utils import WORK
from model import (ID_COLS, feature_names, truth_pairs, fold_of, 
                   in_train_sample, N_FOLDS, feat_files as get_feat_files)

print("=" * 80)
print("ENSEMBLE & HYPERPARAMETER OPTIMIZATION")
print("=" * 80)


def load_training_data(extra=None, use_v2_features=False):
    """Load and prepare training data for modeling."""
    
    truth = truth_pairs()
    tp = truth.select(ID_COLS).with_columns(pl.lit(1, pl.Int8).alias("y"))
    
    feat_pattern = f"feat_v2_train" if use_v2_features else f"feat_train"
    files = sorted([f for f in os.listdir(WORK) if f.startswith(feat_pattern) and f.endswith(".parquet")])
    files = [os.path.join(WORK, f) for f in files]
    
    if not files:
        print(f"ERROR: No feature files found matching: {feat_pattern}")
        return None, None, None, None
    
    print(f"Loading training data from {len(files)} files...")
    
    # Load and concatenate
    dfs = []
    for f in files[:3]:  # Limit for faster iteration during dev
        df = pl.read_parquet(f)
        if in_train_sample(pl.col("a_idx")) is not None:
            df = df.filter(in_train_sample(pl.col("a_idx")))
        dfs.append(df)
    
    df = pl.concat(dfs)
    
    # Add labels
    df = df.join(tp, on=ID_COLS, how="left").with_columns(
        pl.col("y").fill_null(0),
        fold_of(pl.col("a_idx")).alias("fold")
    )
    
    # Add cluster features if available
    if extra:
        if isinstance(extra, str) and os.path.exists(extra):
            ex = pl.read_parquet(extra)
            df = df.join(ex, on=ID_COLS, how="left")
    
    feats = [c for c in df.columns if c not in ID_COLS + ["y", "fold"]]
    
    X = df.select(feats).to_numpy().astype(np.float32)
    y = df["y"].to_numpy().astype(np.float32)
    fold = df["fold"].to_numpy().astype(np.int8)
    
    print(f"Data loaded: {X.shape[0]:,} rows, {X.shape[1]} features")
    print(f"Positive rate: {y.mean():.4f}")
    
    return X, y, fold, feats


def optimize_lightgbm(X, y, fold, n_trials=50):
    """Use Optuna to find best LightGBM hyperparameters."""
    
    if not HAS_OPTUNA:
        print("Optuna not available, using default parameters")
        return {
            "learning_rate": 0.06,
            "num_leaves": 255,
            "min_data_in_leaf": 100,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "lambda_l2": 1.0,
            "max_bin": 127
        }
    
    print(f"\n=== OPTIMIZING LIGHTGBM ({n_trials} trials) ===")
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "num_threads": 10,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 64, 512),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "bagging_freq": 1,
            "lambda_l1": trial.suggest_float("lambda_l1", 0, 2.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0, 2.0),
            "max_bin": trial.suggest_categorical("max_bin", [63, 127, 255]),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0, 1.0),
        }
        
        # Cross-validation
        scores = []
        for k in range(N_FOLDS):
            train_idx = fold != k
            val_idx = fold == k
            
            dtrain = lgb.Dataset(X[train_idx], y[train_idx])
            dval = lgb.Dataset(X[val_idx], y[val_idx])
            
            model = lgb.train(
                params,
                dtrain,
                num_boost_round=2000,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(0)
                ]
            )
            
            scores.append(model.best_score["valid_0"]["binary_logloss"])
        
        return np.mean(scores)
    
    study = optuna.create_study(direction="minimize", study_name="lightgbm_opt")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"\nBest trial:")
    print(f"  Loss: {study.best_trial.value:.6f}")
    print(f"  Params: {study.best_trial.params}")
    
    return study.best_trial.params


def train_xgboost(X, y, fold, feats):
    """Train XGBoost model."""
    
    if not HAS_XGB:
        print("XGBoost not available")
        return []
    
    print("\n=== TRAINING XGBOOST ===")
    
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 8,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "n_jobs": 10,
    }
    
    models = []
    for k in range(N_FOLDS):
        print(f"Fold {k}...")
        train_idx = fold != k
        val_idx = fold == k
        
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx], feature_names=feats)
        dval = xgb.DMatrix(X[val_idx], label=y[val_idx], feature_names=feats)
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=2000,
            evals=[(dval, "val")],
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        models.append(model)
        print(f"  Best iteration: {model.best_iteration}, Loss: {model.best_score:.6f}")
    
    return models


def train_catboost(X, y, fold, feats):
    """Train CatBoost model."""
    
    if not HAS_CATBOOST:
        print("CatBoost not available")
        return []
    
    print("\n=== TRAINING CATBOOST ===")
    
    params = {
        "loss_function": "Logloss",
        "learning_rate": 0.05,
        "depth": 8,
        "l2_leaf_reg": 3.0,
        "subsample": 0.8,
        "random_strength": 1.0,
        "border_count": 128,
        "thread_count": 10,
        "verbose": False,
    }
    
    models = []
    for k in range(N_FOLDS):
        print(f"Fold {k}...")
        train_idx = fold != k
        val_idx = fold == k
        
        train_pool = cb.Pool(X[train_idx], y[train_idx], feature_names=feats)
        val_pool = cb.Pool(X[val_idx], y[val_idx], feature_names=feats)
        
        model = cb.CatBoost(params)
        model.fit(
            train_pool,
            eval_set=val_pool,
            early_stopping_rounds=50,
            verbose=False
        )
        
        models.append(model)
        print(f"  Best iteration: {model.get_best_iteration()}, Loss: {model.get_best_score()['validation']['Logloss']:.6f}")
    
    return models


def blend_predictions(preds_list: List[np.ndarray], weights=None) -> np.ndarray:
    """Blend multiple model predictions."""
    
    if weights is None:
        weights = np.ones(len(preds_list)) / len(preds_list)
    
    weights = np.array(weights) / np.sum(weights)
    
    blended = np.zeros_like(preds_list[0])
    for pred, w in zip(preds_list, weights):
        blended += w * pred
    
    return blended


def main():
    """Run full optimization and ensemble pipeline."""
    
    # Check which libraries are available
    available_models = ["LightGBM"]
    if HAS_XGB:
        available_models.append("XGBoost")
    if HAS_CATBOOST:
        available_models.append("CatBoost")
    
    print(f"\nAvailable models: {', '.join(available_models)}")
    
    # Load data
    cluster_file = os.path.join(WORK, "cluster_train_m2.parquet")
    X, y, fold, feats = load_training_data(extra=cluster_file if os.path.exists(cluster_file) else None)
    
    if X is None:
        print("Failed to load data")
        return
    
    # Optimize LightGBM
    best_lgb_params = optimize_lightgbm(X, y, fold, n_trials=30)
    
    # Train models
    models_dict = {}
    
    # LightGBM with optimized params
    print("\n=== TRAINING OPTIMIZED LIGHTGBM ===")
    lgb_models = []
    lgb_params = {
        **best_lgb_params,
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "num_threads": 10,
        "bagging_freq": 1,
    }
    
    for k in range(N_FOLDS):
        train_idx = fold != k
        val_idx = fold == k
        
        dtrain = lgb.Dataset(X[train_idx], y[train_idx], feature_name=feats)
        dval = lgb.Dataset(X[val_idx], y[val_idx], feature_name=feats)
        
        model = lgb.train(
            lgb_params,
            dtrain,
            num_boost_round=4000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(100)
            ]
        )
        
        lgb_models.append(model)
        print(f"Fold {k}: iter={model.best_iteration}, loss={model.best_score['valid_0']['binary_logloss']:.6f}")
    
    models_dict["lightgbm"] = lgb_models
    
    # XGBoost
    if HAS_XGB:
        xgb_models = train_xgboost(X, y, fold, feats)
        if xgb_models:
            models_dict["xgboost"] = xgb_models
    
    # CatBoost
    if HAS_CATBOOST:
        cb_models = train_catboost(X, y, fold, feats)
        if cb_models:
            models_dict["catboost"] = cb_models
    
    # Generate OOF predictions for each model
    print("\n=== GENERATING OOF PREDICTIONS ===")
    oof_preds = {}
    
    for model_name, models in models_dict.items():
        print(f"OOF for {model_name}...")
        preds = np.zeros(len(y))
        
        for k, model in enumerate(models):
            val_idx = fold == k
            
            if model_name == "lightgbm":
                preds[val_idx] = model.predict(X[val_idx])
            elif model_name == "xgboost":
                dval = xgb.DMatrix(X[val_idx], feature_names=feats)
                preds[val_idx] = model.predict(dval)
            elif model_name == "catboost":
                preds[val_idx] = model.predict(X[val_idx], prediction_type="Probability")
        
        oof_preds[model_name] = preds
        
        # Compute logloss
        logloss = -np.mean(y * np.log(np.clip(preds, 1e-15, 1-1e-15)) + 
                           (1-y) * np.log(np.clip(1-preds, 1e-15, 1-1e-15)))
        print(f"  OOF LogLoss: {logloss:.6f}")
    
    # Try different blend weights
    if len(oof_preds) > 1:
        print("\n=== OPTIMIZING BLEND WEIGHTS ===")
        best_loss = float("inf")
        best_weights = None
        
        # Grid search over blend weights
        from itertools import product
        weight_options = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        
        model_names = list(oof_preds.keys())
        if len(model_names) == 2:
            for w1 in weight_options:
                w2 = 1 - w1
                blended = blend_predictions([oof_preds[model_names[0]], oof_preds[model_names[1]]], [w1, w2])
                loss = -np.mean(y * np.log(np.clip(blended, 1e-15, 1-1e-15)) + 
                                (1-y) * np.log(np.clip(1-blended, 1e-15, 1-1e-15)))
                if loss < best_loss:
                    best_loss = loss
                    best_weights = [w1, w2]
        
        if best_weights:
            print(f"Best blend weights: {dict(zip(model_names, best_weights))}")
            print(f"Best blend LogLoss: {best_loss:.6f}")
    
    # Save models
    output_path = os.path.join(WORK, "ensemble_models.pkl")
    with open(output_path, "wb") as f:
        pickle.dump({
            "models": models_dict,
            "best_lgb_params": best_lgb_params,
            "feature_names": feats,
            "blend_weights": best_weights if len(oof_preds) > 1 else None
        }, f)
    
    print(f"\nEnsemble models saved to: {output_path}")
    print("\n" + "=" * 80)
    print("OPTIMIZATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
