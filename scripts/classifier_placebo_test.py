"""
classifier_placebo_test.py
--------------------------
Pre-period placebo test for the AI adoption classifier.

Question: does the classifier capture pre-existing developer conscientiousness
rather than true AI tool adoption?

Method:
  1. Load the trained classifier and training features.
  2. Construct "pre-only" feature vectors: set post_* = pre_*, delta_* = 0.
     This mimics how pre-period scores are computed in the scraper.
  3. Score all accounts using only pre-period information.
  4. Split into quartiles by pre-only score.
  5. Compare pre-existing metrics (pre_commit_count, pre_message_length,
     pre_active_weeks, etc.) across quartiles.
  6. If the top quartile already had higher pre-existing quality, the
     classifier is confounded by conscientiousness.

Run:
  uv run --with scikit-learn --with pandas --with joblib \
         python3 -u scripts/classifier_placebo_test.py
"""

import os
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

CLASSIFIER_PKL = os.path.join(DATA, "classifier_model.pkl")
TRAIN_FEATURES_CSV = os.path.join(DATA, "classifier_full_features.csv")
OUT_CSV = os.path.join(DATA, "placebo_test_results.csv")


def main():
    print("=" * 70)
    print("CLASSIFIER PRE-PERIOD PLACEBO TEST")
    print("=" * 70)

    # Load classifier
    clf_bundle = joblib.load(CLASSIFIER_PKL)
    model = clf_bundle["model"]
    imputer = clf_bundle["imputer"]
    feat_cols = clf_bundle["feature_cols"]
    print(f"\nModel: {clf_bundle['model_name']}")
    print(f"Features: {len(feat_cols)}")

    # Load training features
    df = pd.read_csv(TRAIN_FEATURES_CSV)
    print(f"Training accounts: {len(df)}")

    # Build pre-only feature matrix
    # Strategy: for every feature column, if it starts with "post_", set it
    # equal to the corresponding "pre_" feature. If "delta_", set to 0.
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        if c.startswith("post_"):
            pre_col = "pre_" + c[5:]
            X[c] = df[pre_col] if pre_col in df.columns else 0.0
        elif c.startswith("delta_"):
            X[c] = 0.0
        else:
            # pre_* or other columns — use as-is
            X[c] = df[c] if c in df.columns else 0.0

    # Impute and score
    X_imp = imputer.transform(X)
    pre_scores = model.predict_proba(X_imp)[:, 1]
    df["pre_only_score"] = pre_scores

    print(f"\nPre-only score distribution:")
    print(f"  Mean: {pre_scores.mean():.3f}")
    print(f"  Std:  {pre_scores.std():.3f}")
    print(f"  Min:  {pre_scores.min():.3f}")
    print(f"  Max:  {pre_scores.max():.3f}")

    # Quartile split
    df["quartile"] = pd.qcut(df["pre_only_score"], 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])

    # Pre-existing metrics to compare
    pre_metrics = [
        ("pre_commit_count", "Pre-period commit count"),
        ("pre_mean_message_length", "Pre-period mean message length"),
        ("pre_active_weeks", "Pre-period active weeks"),
        ("pre_mean_commits_per_active_week", "Pre-period commits per active week"),
        ("pre_frac_conventional", "Pre-period frac conventional commits"),
        ("pre_frac_multiline", "Pre-period frac multiline commits"),
        ("pre_frac_has_bullets", "Pre-period frac bullet lists"),
        ("pre_mean_inter_commit_hours", "Pre-period mean inter-commit hours"),
    ]

    print("\n" + "=" * 70)
    print("QUARTILE COMPARISON (pre-only scores)")
    print("=" * 70)
    print(f"{'Metric':<45} {'Q1_mean':>10} {'Q4_mean':>10} {'diff':>10} {'p(t-test)':>10}")
    print("-" * 90)

    results = []
    for col, label in pre_metrics:
        if col not in df.columns:
            continue
        q1 = df[df["quartile"] == "Q1_low"][col].dropna()
        q4 = df[df["quartile"] == "Q4_high"][col].dropna()

        mean_q1 = q1.mean()
        mean_q4 = q4.mean()
        diff = mean_q4 - mean_q1

        # Welch's t-test
        if len(q1) > 1 and len(q4) > 1:
            tstat, pval = stats.ttest_ind(q4, q1, equal_var=False)
        else:
            pval = float("nan")

        sig = "*" if pval < 0.05 else ""
        print(f"{label:<45} {mean_q1:>10.2f} {mean_q4:>10.2f} {diff:>+10.2f} {pval:>10.4f} {sig}")

        results.append({
            "metric": label,
            "col": col,
            "q1_mean": mean_q1,
            "q4_mean": mean_q4,
            "diff": diff,
            "p_value": pval,
            "significant_05": pval < 0.05,
        })

    # Also compare by actual label
    print("\n" + "=" * 70)
    print("PRE-SCORE BY ACTUAL LABEL (ground truth)")
    print("=" * 70)
    if "label" in df.columns:
        for lbl, grp in df.groupby("label"):
            name = "Positive (AI user)" if lbl == 1 else "Negative (control)"
            print(f"  {name}: pre-only mean = {grp['pre_only_score'].mean():.3f}, "
                  f"n = {len(grp)}")

        pos_scores = df[df["label"] == 1]["pre_only_score"].dropna()
        neg_scores = df[df["label"] == 0]["pre_only_score"].dropna()
        if len(pos_scores) > 1 and len(neg_scores) > 1:
            tstat, pval = stats.ttest_ind(pos_scores, neg_scores, equal_var=False)
            print(f"\n  Welch t-test: t={tstat:.2f}, p={pval:.4f}")
            if pval < 0.05:
                print("  -> Ground-truth positives ALREADY score higher pre-only. "
                      "Classifier may capture pre-existing traits.")
            else:
                print("  -> No significant pre-existing difference. Classifier signal "
                      "is likely adoption-specific.")

    # Save results
    pd.DataFrame(results).to_csv(OUT_CSV, index=False)
    print(f"\nResults saved: {OUT_CSV}")

    # Overall interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    sig_count = sum(1 for r in results if r["significant_05"])
    total = len(results)
    print(f"Significant pre-existing differences (p<0.05): {sig_count}/{total}")

    if sig_count >= 3:
        print("\n⚠️  WARNING: Multiple pre-existing differences detected.")
        print("   The classifier appears to sort accounts by pre-existing developer")
        print("   quality/conscientiousness, not purely by AI adoption.")
        print("   The country-level IV may be confounded.")
    elif sig_count >= 1:
        print("\n⚡ CAUTION: Some pre-existing differences detected.")
        print("   The classifier may partially capture conscientiousness.")
        print("   Consider controlling for baseline activity in the panel model.")
    else:
        print("\n✓ PASS: No significant pre-existing differences.")
        print("   The classifier signal appears specific to AI adoption patterns.")

    print("\nRecommended next step: if confounded, add pre_commit_count or")
    print("pre_mean_message_length as a control in the panel regression.")


if __name__ == "__main__":
    main()
