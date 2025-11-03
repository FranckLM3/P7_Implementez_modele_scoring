#!/usr/bin/env python3
"""
LightGBM training script with optimized hyperparameters (from legacy/MLflow.py)

This script uses hyperparameters found via Bayesian optimization:
- n_estimators=1000
- learning_rate=0.02
- max_depth=9
- num_leaves=44
- scale_pos_weight=11.39 (to handle class imbalance)
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from credit_scoring.data import load_dataset
from credit_scoring.metrics import compute_metrics


def build_pipeline() -> Pipeline:
    """Build complete pipeline with preprocessing + feature selection + optimized LightGBM"""
    # No need to define columns here, they will be inferred from data
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    
    # Preprocessor will be configured dynamically in main()
    preprocessor = ColumnTransformer(transformers=[])
    
    # Feature selector with LightGBM
    selector = SelectFromModel(LGBMClassifier(n_jobs=-1, random_state=123))
    
    # Optimized hyperparameters from legacy/MLflow.py
    clf = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        num_leaves=44,
        colsample_bytree=0.235,
        subsample=0.150,
        max_depth=9,
        reg_alpha=0.647,
        reg_lambda=0.884,
        min_split_gain=0.773,
        min_child_weight=34.7,
        scale_pos_weight=11.39,  # Important to handle class imbalance
        random_state=123,
        verbose=-1,
    )
    
    return Pipeline([
        ("preprocessor", preprocessor),
        ("selector", selector),
        ("classifier", clf),
    ]), numeric_transformer, categorical_transformer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LightGBM model")
    parser.add_argument("--data", type=str, default="data/model_dataset.csv")
    parser.add_argument("--out-dir", type=str, default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    df = load_dataset(args.data, low_memory=False)
    
    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Split features and target
    if "SK_ID_CURR" in df.columns:
        df = df.drop(columns=["SK_ID_CURR"])
    
    X = df.drop(columns=["TARGET"])
    y = df["TARGET"]
    
    print(f"Shape: {X.shape}, Target distribution: {y.value_counts().to_dict()}")
    
    # Infer column types
    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()
    
    print(f"Numeric features: {len(numeric_features)}, categorical: {len(categorical_features)}")
    
    # Build pipeline
    pipe, num_transformer, cat_transformer = build_pipeline()
    
    # Configure preprocessor with detected columns
    pipe.named_steps["preprocessor"].transformers = [
        ("num", num_transformer, numeric_features),
        ("cat", cat_transformer, categorical_features),
    ]
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=123, stratify=y
    )
    
    print("Training model...")
    pipe.fit(X_train, y_train)
    
    print("Evaluating...")
    y_proba = pipe.predict_proba(X_test)[:, 1]
    report = compute_metrics(y_test.to_numpy(), y_proba)
    
    # Save artifacts
    print("Saving artifacts...")
    joblib.dump(pipe, out_dir / "pipeline.joblib")
    (out_dir / "metrics.json").write_text(report.to_json())
    
    # Feature importance
    try:
        importances = pipe.named_steps["classifier"].feature_importances_
        # Get feature names after selection
        pre = pipe.named_steps["preprocessor"]
        cat_cols = pre.transformers_[1][2] if len(pre.transformers_) > 1 else []
        if len(cat_cols) > 0:
            cat_names = list(pre.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(cat_cols))
        else:
            cat_names = []
        num_names = list(pre.transformers_[0][2])
        all_names = num_names + cat_names
        
        selector = pipe.named_steps["selector"]
        support = selector.get_support()
        selected_names = [n for n, keep in zip(all_names, support) if keep]
        
        feat_df = pd.DataFrame({
            "feature": selected_names,
            "importance": importances
        }).sort_values("importance", ascending=False)
        
        feat_df.to_csv(out_dir / "feature_importance.csv", index=False)
        print(f"Top 10 features:\n{feat_df.head(10)}")
    except Exception as e:
        print(f"Could not save feature importance: {e}")
    
    print("\n" + "="*50)
    print("RESULTS")
    print("="*50)
    print(report.to_json())
    print(f"\nArtifacts saved in: {out_dir}/")


if __name__ == "__main__":
    main()
