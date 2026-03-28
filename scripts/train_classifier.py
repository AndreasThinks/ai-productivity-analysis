"""
Train a binary classifier to detect AI coding tool usage from GitHub behavioural features.
"""

import sys
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.impute import SimpleImputer

# ── paths ──────────────────────────────────────────────────────────────────────
DATA_PATH   = "data/classifier_full_features.csv"
PRED_PATH   = "data/classifier_predictions.csv"
MODEL_PATH  = "data/classifier_model.pkl"

# ── 1. Load data ───────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1 — Loading data")
print("=" * 65)
df = pd.read_csv(DATA_PATH)
print(f"  Loaded {len(df)} rows × {df.shape[1]} columns")
print(f"  Positives (label=1): {(df['label']==1).sum()}")
print(f"  Negatives (label=0): {(df['label']==0).sum()}")
print(f"  marker_confidence distribution:\n{df['marker_confidence'].value_counts(dropna=False).to_string()}")

# ── 2. Drop leakage / metadata columns ────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2 — Dropping leakage / metadata columns")
print("=" * 65)

COLS_TO_DROP_EXPLICIT = [
    "login", "label", "marker_confidence", "has_claude_markers",
    "pre_commit_count", "post_commit_count",
    "discovery_method",   # encodes how account was found → leaks label
]
# Any column with 'marker' in the name
marker_cols = [c for c in df.columns if "marker" in c.lower()]

drop_set = set(COLS_TO_DROP_EXPLICIT) | set(marker_cols)
drop_set = {c for c in drop_set if c in df.columns}  # only existing cols
print(f"  Dropping: {sorted(drop_set)}")

# Save login + true label for predictions file before dropping
meta_df = df[["login", "label",
              df.columns[df.columns.str.contains("marker_confidence")][0]
              if any(df.columns.str.contains("marker_confidence")) else "label"]].copy()
meta_df = df[["login", "label"]].copy()
if "marker_confidence" in df.columns:
    meta_df["marker_confidence"] = df["marker_confidence"]

y = df["label"].values
X_raw = df.drop(columns=list(drop_set))

# ── 3. Feature list ────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3 — Feature columns used")
print("=" * 65)
feature_cols = list(X_raw.columns)
print(f"  Total features: {len(feature_cols)}")
for fc in feature_cols:
    print(f"    {fc}")

# ── 4. Impute missing values ───────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 4 — Imputing missing values (median)")
print("=" * 65)
imputer = SimpleImputer(strategy="median")
X = imputer.fit_transform(X_raw)
print(f"  Missing before imputation: {X_raw.isnull().sum().sum()}")
print(f"  Missing after  imputation: {np.isnan(X).sum()}")

# ── 5. Train / test split ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 5 — Stratified 80/20 train/test split (random_state=42)")
print("=" * 65)
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(len(y)), test_size=0.2, stratify=y, random_state=42
)
print(f"  Train: {len(y_train)} rows  (pos={y_train.sum()}, neg={(y_train==0).sum()})")
print(f"  Test : {len(y_test)} rows  (pos={y_test.sum()}, neg={(y_test==0).sum()})")

# ── 6. Define models ───────────────────────────────────────────────────────────
models = {
    "Logistic Regression": LogisticRegression(
        penalty="l2", C=1.0, max_iter=1000, random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42
    ),
    "Gradient Boosted Trees": GradientBoostingClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42
    ),
}

# ── 7. Train & evaluate on test set ───────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6+7 — Training models and evaluating on held-out test set")
print("=" * 65)

results = {}
trained_models = {}

for name, clf in models.items():
    print(f"\n  ── {name} ──")
    clf.fit(X_train, y_train)
    trained_models[name] = clf

    y_pred  = clf.predict(X_test)
    y_prob  = clf.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="macro", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="macro", zero_division=0)
    auc  = roc_auc_score(y_test, y_prob)
    cm   = confusion_matrix(y_test, y_pred)

    results[name] = dict(acc=acc, prec=prec, rec=rec, f1=f1, auc=auc, cm=cm)

    print(f"    Accuracy  : {acc:.3f}")
    print(f"    Precision : {prec:.3f}  (macro)")
    print(f"    Recall    : {rec:.3f}  (macro)")
    print(f"    F1        : {f1:.3f}  (macro)")
    print(f"    ROC-AUC   : {auc:.3f}")
    print(f"    Confusion matrix (rows=actual, cols=pred):")
    print(f"      TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"      FN={cm[1,0]}  TP={cm[1,1]}")

# ── 9. Cross-validation (primary reliability metric) ─────────────────────────
print("\n" + "=" * 65)
print("STEP 9 — 5-fold stratified cross-validation (full dataset)")
print("=" * 65)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}
for name, clf in models.items():
    scores = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")
    cv_results[name] = (scores.mean(), scores.std())
    print(f"  {name}: {scores.mean():.3f} ± {scores.std():.3f}  (folds: {np.round(scores,3)})")

# ── Determine best model by CV AUC ────────────────────────────────────────────
best_name = max(cv_results, key=lambda k: cv_results[k][0])
best_clf  = trained_models[best_name]
print(f"\n  Best model by CV AUC: {best_name}  ({cv_results[best_name][0]:.3f} ± {cv_results[best_name][1]:.3f})")

# ── 8. Feature importance for best model ─────────────────────────────────────
print("\n" + "=" * 65)
print(f"STEP 8 — Feature importance for best model: {best_name}")
print("=" * 65)

if hasattr(best_clf, "feature_importances_"):
    importances = best_clf.feature_importances_
elif hasattr(best_clf, "coef_"):
    importances = np.abs(best_clf.coef_[0])
else:
    importances = np.zeros(len(feature_cols))

feat_imp = pd.Series(importances, index=feature_cols).sort_values(ascending=False)
top10 = feat_imp.head(10)
print("  Top 10 features:")
for rank, (feat, imp) in enumerate(top10.items(), 1):
    print(f"    {rank:2d}. {feat:<45s}  {imp:.4f}")

top5_features = list(top10.head(5).index)

# ── 10. Save predictions ──────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 10 — Saving predictions to", PRED_PATH)
print("=" * 65)

# re-predict with best model over all data (for the full predictions file)
all_prob  = best_clf.predict_proba(X)[:, 1]
all_pred  = best_clf.predict(X)

pred_df = meta_df.copy()
pred_df["predicted_label"] = all_pred
pred_df["predicted_prob"]  = np.round(all_prob, 4)
# rename to match spec
pred_df = pred_df.rename(columns={"label": "true_label"})
if "marker_confidence" not in pred_df.columns:
    pred_df["marker_confidence"] = "unknown"
pred_df = pred_df[["login", "true_label", "predicted_label", "predicted_prob", "marker_confidence"]]
pred_df.to_csv(PRED_PATH, index=False)
print(f"  Saved {len(pred_df)} rows.")

# ── 11. Save best model ───────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 11 — Saving best model to", MODEL_PATH)
print("=" * 65)
joblib.dump({"model": best_clf, "imputer": imputer, "feature_cols": feature_cols,
             "model_name": best_name}, MODEL_PATH)
print(f"  Saved.")

# ── 12. Clean summary ─────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)
print(f"\n  Dataset: {len(y)} total  |  pos={y.sum()}  neg={(y==0).sum()}")
print(f"\n  Test-set results (80/20 stratified split):")
header = f"  {'Model':<26}  {'Acc':>5}  {'Prec':>5}  {'Rec':>5}  {'F1':>5}  {'AUC':>5}"
print(header)
print("  " + "-" * (len(header)-2))
for name, r in results.items():
    marker = " *" if name == best_name else "  "
    print(f"  {name:<26}  {r['acc']:.3f}  {r['prec']:.3f}  {r['rec']:.3f}  {r['f1']:.3f}  {r['auc']:.3f}{marker}")
print("  (* = best by CV AUC)")

print(f"\n  5-fold CV AUC (primary metric — small N={len(y)}):")
for name, (mean, std) in cv_results.items():
    marker = " *" if name == best_name else "  "
    print(f"    {name:<26}: {mean:.3f} ± {std:.3f}{marker}")

print(f"\n  Best model: {best_name}")
print(f"    Test Acc={results[best_name]['acc']:.3f}  Prec={results[best_name]['prec']:.3f}  "
      f"Rec={results[best_name]['rec']:.3f}  F1={results[best_name]['f1']:.3f}  AUC={results[best_name]['auc']:.3f}")

print(f"\n  Top 5 features ({best_name}):")
for rank, (feat, imp) in enumerate(feat_imp.head(5).items(), 1):
    print(f"    {rank}. {feat}  ({imp:.4f})")

# Caveats
best_cv_mean = cv_results[best_name][0]
if best_cv_mean < 0.70:
    print("\n  ⚠  WARNING: Best CV AUC < 0.70 — dataset too small for reliable")
    print("     classification. Collecting more labeled data is the priority.")
else:
    print(f"\n  CV AUC {best_cv_mean:.3f} ≥ 0.70 — moderate signal detected.")

print("\n  Files written:")
print(f"    {PRED_PATH}")
print(f"    {MODEL_PATH}")
print("=" * 65)

# Export key results as a module-level dict for AGENTS.md update script
_RESULTS = {
    "best_name": best_name,
    "best_test": results[best_name],
    "cv_results": cv_results,
    "top5": top5_features,
    "n_total": int(len(y)),
    "n_pos": int(y.sum()),
    "n_neg": int((y==0).sum()),
    "best_cv_mean": best_cv_mean,
}
