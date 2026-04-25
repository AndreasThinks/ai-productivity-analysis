"""
retrain_classifier_expanded.py
------------------------------
Merge expansion positives into the training set and retrain the classifier.

Inputs:
  data/classifier_full_features.csv    — original 235 accounts (33 pos, 202 neg)
  data/expansion_features.csv          — 41 expansion positives

Output:
  data/classifier_model_expanded.pkl   — retrained model bundle
  data/classifier_predictions_expanded.csv  — CV predictions

Run:
  uv run --with scikit-learn --with pandas --with joblib \
         python3 -u scripts/retrain_classifier_expanded.py
"""

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_score

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

ORIG_FEATURES = os.path.join(DATA, "classifier_full_features.csv")
EXP_FEATURES  = os.path.join(DATA, "expansion_features.csv")
MODEL_OUT     = os.path.join(DATA, "classifier_model_expanded.pkl")
PREDS_OUT     = os.path.join(DATA, "classifier_predictions_expanded.csv")


def main():
    print("=" * 70)
    print("RETRAIN CLASSIFIER WITH EXPANSION POSITIVES")
    print("=" * 70)

    # Load original features
    orig = pd.read_csv(ORIG_FEATURES)
    print(f"\nOriginal features: {len(orig)} accounts")
    print(f"  Positives: {(orig['label'] == 1).sum()}")
    print(f"  Negatives: {(orig['label'] == 0).sum()}")

    # Load expansion features
    exp = pd.read_csv(EXP_FEATURES)
    print(f"\nExpansion features: {len(exp)} accounts")

    # Expansion features may not have a 'label' column — add it
    if "label" not in exp.columns:
        exp["label"] = 1

    # Check for duplicate logins
    dupes = set(orig["login"]) & set(exp["login"])
    if dupes:
        print(f"\nWarning: {len(dupes)} duplicate logins found. Dropping from expansion.")
        exp = exp[~exp["login"].isin(dupes)]

    # Add tool_type for consistency
    if "tool_type" not in orig.columns:
        orig["tool_type"] = "claude"
    if "tool_type" not in exp.columns:
        exp["tool_type"] = "claude"

    # Concatenate
    merged = pd.concat([orig, exp], ignore_index=True)
    print(f"\nMerged dataset: {len(merged)} accounts")
    print(f"  Positives: {(merged['label'] == 1).sum()}")
    print(f"  Negatives: {(merged['label'] == 0).sum()}")

    # Identify feature columns (exclude metadata)
    meta_cols = {"login", "label", "tool_type", "discovery_method", "marker_confidence"}
    feature_cols = [c for c in merged.columns if c not in meta_cols]
    print(f"\nFeature columns: {len(feature_cols)}")

    X = merged[feature_cols].values
    y = merged["label"].values

    # Impute missing values
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X)

    # Train Random Forest with same hyperparameters as original
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=4,
        random_state=42,
        class_weight="balanced",
    )

    # 5-fold stratified cross-validation
    print("\nRunning 5-fold stratified CV...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = cross_val_score(model, X_imp, y, cv=cv, scoring="roc_auc")
    print(f"  AUC mean: {auc_scores.mean():.3f} (±{auc_scores.std():.3f})")
    print(f"  AUC by fold: {[f'{s:.3f}' for s in auc_scores]}")

    # Fit on full data
    print("\nFitting on full training set...")
    model.fit(X_imp, y)

    # Feature importance
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=False).head(10)
    print("\nTop 10 features by importance:")
    for feat, imp in top10.items():
        print(f"  {feat}: {imp:.4f}")

    # Save predictions
    preds = model.predict_proba(X_imp)[:, 1]
    merged["predicted_prob"] = preds
    merged.to_csv(PREDS_OUT, index=False)
    print(f"\nPredictions saved: {PREDS_OUT}")

    # Save model bundle
    bundle = {
        "model": model,
        "imputer": imputer,
        "feature_cols": feature_cols,
        "model_name": "RandomForest_200_4_expanded",
        "training_date": datetime.now().isoformat(),
        "n_positives": int((merged["label"] == 1).sum()),
        "n_negatives": int((merged["label"] == 0).sum()),
        "cv_auc_mean": float(auc_scores.mean()),
        "cv_auc_std": float(auc_scores.std()),
    }
    joblib.dump(bundle, MODEL_OUT)
    print(f"Model bundle saved: {MODEL_OUT}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
