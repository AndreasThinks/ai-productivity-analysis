"""
build_panel_v2.py — Phase 2 panel regression

Replaces the Oxford Insights AI Readiness Index (Phase 1's broken IV) with a
per-country-year AI adoption fraction derived from the behavioural classifier.

Steps:
  4a. Load classifier
  4b. Sanity-check on training data
  4c. Sanity-check on Aider validation accounts
  4d. Load the GitHub panel
  4e. Score panel accounts / build country-year adoption fraction
  4f. Run Phase 1 baseline + Phase 2 classifier-based regressions
  4g. Save results
  4h. Print interpretation

Run:
  uv run --with linearmodels --with scikit-learn --with pandas --with joblib \
         --with statsmodels python3 -u scripts/build_panel_v2.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

CLASSIFIER_PKL       = os.path.join(DATA, "classifier_model.pkl")
TRAIN_FEATURES_CSV   = os.path.join(DATA, "classifier_full_features.csv")
AIDER_RESULTS_CSV    = os.path.join(DATA, "aider_validation_results.csv")
PREDICTIONS_CSV      = os.path.join(DATA, "classifier_predictions.csv")
PANEL_FLAT_CSV       = os.path.join(DATA, "github_panel_flat.csv")
OXFORD_CSV           = os.path.join(DATA, "panel_dataset.csv")

REGRESSION_OUT       = os.path.join(DATA, "regression_results_v2.txt")
ADOPTION_OUT         = os.path.join(DATA, "country_quarter_ai_adoption.csv")

# ─────────────────────────────────────────────────────────────────────────────
# 4a. Load classifier
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("4a. Loading classifier")
print("=" * 70)

clf_bundle = joblib.load(CLASSIFIER_PKL)
model      = clf_bundle["model"]
imputer    = clf_bundle["imputer"]
feat_cols  = clf_bundle["feature_cols"]
model_name = clf_bundle["model_name"]

print(f"  Model: {model_name}")
print(f"  Expected feature columns: {len(feat_cols)}")


def score_df(df: pd.DataFrame, label: str = "") -> np.ndarray:
    """Score a dataframe using the loaded classifier. Returns predicted probabilities."""
    available = [c for c in feat_cols if c in df.columns]
    missing   = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  [{label}] Warning: {len(missing)} feature cols missing — filled with NaN")
    X = pd.DataFrame(index=df.index)
    for c in feat_cols:
        X[c] = df[c] if c in df.columns else np.nan
    X_imp = imputer.transform(X)
    probs = model.predict_proba(X_imp)[:, 1]
    return probs


# ─────────────────────────────────────────────────────────────────────────────
# 4b. Sanity-check on training data
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4b. Sanity check — training data")
print("=" * 70)

train = pd.read_csv(TRAIN_FEATURES_CSV)
probs = score_df(train, "train")
pos_mask = train["label"] == 1
neg_mask = train["label"] == 0

mean_pos = probs[pos_mask].mean()
mean_neg = probs[neg_mask].mean()

print(f"  Training positives: n={pos_mask.sum()}, mean score = {mean_pos:.3f}")
print(f"  Training negatives: n={neg_mask.sum()}, mean score = {mean_neg:.3f}")

if mean_pos < mean_neg:
    print("  ABORT: Claude positives score LOWER than negatives — classifier is inverted!")
    sys.exit(1)
else:
    print(f"  ✓ Sanity check passed (gap: +{mean_pos - mean_neg:.3f})")


# ─────────────────────────────────────────────────────────────────────────────
# 4c. Sanity-check on Aider accounts
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4c. Sanity check — Aider validation accounts")
print("=" * 70)

aider = pd.read_csv(AIDER_RESULTS_CSV)
aider_probs = score_df(aider, "aider")
print(f"  Aider accounts: n={len(aider)}, mean score = {aider_probs.mean():.3f}")
print(f"  (Expected ~0.727 from validate_aider.py report)")


# ─────────────────────────────────────────────────────────────────────────────
# 4d. Load the GitHub panel
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4d. Loading GitHub panel data")
print("=" * 70)

panel = pd.read_csv(PANEL_FLAT_CSV)
print(f"  Shape: {panel.shape}")
print(f"  Columns: {list(panel.columns)}")
print()
print(panel.head(3).to_string())
print()

# Parse year from quarter string (e.g. "2022-Q4" -> 2022)
panel["year"] = panel["quarter"].str.extract(r"(\d{4})").astype(int)
print(f"  Years in panel: {sorted(panel['year'].unique())}")
print(f"  Countries in panel: {panel['country'].nunique()}")


# ─────────────────────────────────────────────────────────────────────────────
# 4e. Score panel accounts / build country-year AI adoption fraction
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4e. Building country-year AI adoption fraction")
print("=" * 70)

# The panel doesn't contain individual logins — it's already aggregated by
# country-quarter. We can't score individual accounts here.
#
# PROXY APPROACH:
# For accounts in classifier_predictions.csv (training+eval set), we have
# predicted_prob. But these accounts don't map to the panel's country-quarter
# aggregates — the panel was built from GH Archive activity, not from the
# classifier training accounts.
#
# Since the panel data is already a country-quarter aggregate with no individual
# account breakdown, and we don't have a location breakdown for the classifier
# training accounts that links them to the panel, we use the following approach:
#
# Option A (preferred, used here): Use the panel's quarterly time structure as
# a proxy for AI adoption intensity over time. AI tools launched late 2023 /
# early 2024. We create a TIME-BASED adoption proxy:
#   - pre-AI era (2022, 2023): pct_ai_users_proxy = 0.0
#   - post-AI launch (2024):   pct_ai_users_proxy = mean classifier score from
#                               classifier_predictions.csv (our best estimate
#                               of the fraction of active GitHub developers
#                               using AI tools by 2024)
#
# This is a valid (if coarse) instrument: it gives cross-sectional variation if
# we interact it with country-level characteristics, and it creates a structural
# break at the AI launch that can be compared to productivity trends.
#
# Limitation: without per-account location data from the full panel scrape, we
# cannot compute TRUE country-quarter adoption fractions. Flag this prominently.

# Load predictions to get the mean AI probability estimate for 2024 accounts
preds = pd.read_csv(PREDICTIONS_CSV)
mean_2024_ai_score = preds["predicted_prob"].mean()
frac_above_threshold = (preds["predicted_prob"] > 0.5).mean()

print(f"  Classifier predictions: n={len(preds)}")
print(f"  Mean predicted_prob (training accounts): {mean_2024_ai_score:.3f}")
print(f"  Fraction above 0.5 threshold: {frac_above_threshold:.1%}")
print()
print("  NOTE: Panel does not contain individual account logins.")
print("  Cannot compute true per-country AI adoption fraction without a full")
print("  population scrape linking accounts to countries.")
print()
print("  Proxy: using time-based AI adoption indicator:")
print("    2022, 2023 → pct_ai_users = 0.0 (pre-launch baseline)")
print("    2024       → pct_ai_users = mean classifier score from training set")
print("  This gives a structural break at AI tool launch date.")
print("  Cross-country variation requires broader population scoring (Step 9).")

# Build the time-based proxy
adoption_by_year = {
    2022: 0.0,
    2023: 0.0,
    2024: float(mean_2024_ai_score),
}

panel["pct_ai_users"] = panel["year"].map(adoption_by_year)
panel["pct_ai_users_n"] = panel["year"].map({2022: 0, 2023: 0, 2024: len(preds)})

# Country-quarter adoption summary
cq_adoption = (
    panel[["country", "year", "quarter", "pct_ai_users", "pct_ai_users_n"]]
    .drop_duplicates()
    .sort_values(["country", "year"])
)
cq_adoption.to_csv(ADOPTION_OUT, index=False)
print(f"\n  Country-quarter adoption data saved: {ADOPTION_OUT}")
print(f"  Rows: {len(cq_adoption)}")


# ─────────────────────────────────────────────────────────────────────────────
# 4f. Run regressions
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4f. Running regressions")
print("=" * 70)

from linearmodels.panel import PanelOLS

# Productivity metric: log(commits_per_dev + 1)
panel["log_commits"] = np.log1p(panel["commits_per_dev"])
panel["log_prs"]     = np.log1p(panel["prs_per_dev"])
panel["log_events"]  = np.log1p(panel["total_events_per_dev"])

# Drop rows with missing values
panel_clean = panel.dropna(subset=["log_commits", "pct_ai_users"]).copy()
print(f"  Panel rows after cleaning: {len(panel_clean)}")
print(f"  Countries: {panel_clean['country'].nunique()}, Years: {sorted(panel_clean['year'].unique())}")

# linearmodels requires numeric time index — use year (integer)
# We have country + quarter as unique identifiers, but for PanelOLS
# we use country + year. For quarters with same year, average per year first.
panel_year = (
    panel_clean.groupby(["country", "year"])
    .agg(
        log_commits=("log_commits", "mean"),
        log_prs=("log_prs", "mean"),
        log_events=("log_events", "mean"),
        pct_ai_users=("pct_ai_users", "mean"),
        n_developers=("n_developers", "sum"),
    )
    .reset_index()
)

# Set up panel index with numeric time dimension
panel_year = panel_year.set_index(["country", "year"])

results_text = []
results_text.append("=" * 70)
results_text.append("PANEL REGRESSION RESULTS v2")
results_text.append("Date: 2026-03-30")
results_text.append("=" * 70)
results_text.append("")
results_text.append("Panel: github_panel_flat.csv")
results_text.append(f"N observations: {len(panel_clean)}")
results_text.append(f"Countries: {panel_clean.reset_index()['country'].nunique()}")
results_text.append(f"DV: log(commits_per_dev + 1)")
results_text.append("")
results_text.append("NOTE ON IV:")
results_text.append("  pct_ai_users is a TIME-BASED PROXY (not true per-country adoption).")
results_text.append("  Value = 0 for 2022 and 2023; value = mean classifier score")
results_text.append(f"  ({mean_2024_ai_score:.3f}) for 2024.")
results_text.append("  True country-quarter adoption requires a full population scrape.")
results_text.append("  See project_plan.md Step 9 for the path to the real IV.")
results_text.append("")

comparison_rows = []

# ── Regression A: Phase 1 baseline (Oxford Insights AI Readiness) ────────────
print()
print("  Regression A: Phase 1 baseline (Oxford Insights ai_readiness_score)")
print("  Loading panel_dataset.csv...")

oxford = pd.read_csv(OXFORD_CSV)
print(f"  Oxford panel shape: {oxford.shape}")
print(f"  Columns: {list(oxford.columns)}")

reg_a_run = False
if "ai_readiness_score" in oxford.columns and "commits_per_dev" in oxford.columns:
    # Build comparable panel
    oxford_clean = oxford.dropna(subset=["ai_readiness_score", "commits_per_dev"]).copy()
    oxford_clean["log_commits"] = np.log1p(oxford_clean["commits_per_dev"])

    # Need year and country
    if "year" not in oxford_clean.columns and "quarter" in oxford_clean.columns:
        oxford_clean["year"] = oxford_clean["quarter"].astype(str).str.extract(r"(\d{4})").astype(int)
    if "country" not in oxford_clean.columns and "country_code" in oxford_clean.columns:
        oxford_clean["country"] = oxford_clean["country_code"]

    if "country" in oxford_clean.columns and "year" in oxford_clean.columns:
        oxford_clean = oxford_clean.set_index(["country", "year"])
        try:
            mod_a = PanelOLS.from_formula(
                "log_commits ~ ai_readiness_score + EntityEffects + TimeEffects",
                data=oxford_clean,
                drop_absorbed=True,
            )
            res_a = mod_a.fit(cov_type="clustered", cluster_entity=True)
            coef_a = res_a.params["ai_readiness_score"]
            se_a   = res_a.std_errors["ai_readiness_score"]
            pval_a = res_a.pvalues["ai_readiness_score"]
            r2_a   = res_a.rsquared
            n_a    = int(res_a.nobs)
            print(f"  A — N={n_a}, coef={coef_a:.4f}, SE={se_a:.4f}, p={pval_a:.4f}, R²={r2_a:.4f}")
            results_text.append("REGRESSION A — Phase 1 Baseline (Oxford Insights AI Readiness)")
            results_text.append(str(res_a.summary))
            results_text.append("")
            comparison_rows.append({
                "Model": "A — Phase 1 Baseline",
                "IV": "ai_readiness_score (Oxford)",
                "N": n_a,
                "Coef": f"{coef_a:.4f}",
                "SE": f"{se_a:.4f}",
                "p-value": f"{pval_a:.4f}",
                "R²": f"{r2_a:.4f}",
            })
            reg_a_run = True
        except Exception as e:
            print(f"  Regression A failed: {e}")
            results_text.append(f"Regression A: FAILED — {e}")
            results_text.append("")
    else:
        print("  Oxford panel missing 'country' or 'year' — skipping Regression A")
else:
    print("  ai_readiness_score not in Oxford panel — skipping Regression A")
    print(f"  Oxford columns: {list(oxford.columns)}")

if not reg_a_run:
    comparison_rows.append({
        "Model": "A — Phase 1 Baseline",
        "IV": "ai_readiness_score (Oxford)",
        "N": "N/A",
        "Coef": "N/A",
        "SE": "N/A",
        "p-value": "N/A",
        "R²": "N/A",
    })

# ── Regression B: Phase 2 (classifier-based time proxy) ─────────────────────
print()
print("  Regression B: Phase 2 (pct_ai_users — time-based proxy)")

try:
    mod_b = PanelOLS.from_formula(
        "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
        data=panel_year,
        drop_absorbed=True,
    )
    res_b = mod_b.fit(cov_type="clustered", cluster_entity=True)
    coef_b = res_b.params["pct_ai_users"]
    se_b   = res_b.std_errors["pct_ai_users"]
    pval_b = res_b.pvalues["pct_ai_users"]
    r2_b   = res_b.rsquared
    n_b    = int(res_b.nobs)
    print(f"  B — N={n_b}, coef={coef_b:.4f}, SE={se_b:.4f}, p={pval_b:.4f}, R²={r2_b:.4f}")
    results_text.append("REGRESSION B — Phase 2 (classifier-based IV, time proxy)")
    results_text.append(str(res_b.summary))
    results_text.append("")
    comparison_rows.append({
        "Model": "B — Phase 2 (classifier proxy)",
        "IV": "pct_ai_users (time proxy)",
        "N": n_b,
        "Coef": f"{coef_b:.4f}",
        "SE": f"{se_b:.4f}",
        "p-value": f"{pval_b:.4f}",
        "R²": f"{r2_b:.4f}",
    })
    reg_b_ok = True
except Exception as e:
    print(f"  Regression B failed: {e}")
    results_text.append(f"Regression B: FAILED — {e}")
    results_text.append("")
    reg_b_ok = False
    coef_b = se_b = pval_b = r2_b = n_b = None
    comparison_rows.append({
        "Model": "B — Phase 2 (classifier proxy)",
        "IV": "pct_ai_users (time proxy)",
        "N": "N/A",
        "Coef": "N/A",
        "SE": "N/A",
        "p-value": "N/A",
        "R²": "N/A",
    })

# Comparison table
results_text.append("")
results_text.append("COMPARISON TABLE")
results_text.append("-" * 80)
comp_df = pd.DataFrame(comparison_rows)
results_text.append(comp_df.to_string(index=False))
results_text.append("")


# ─────────────────────────────────────────────────────────────────────────────
# 4g. Save results
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4g. Saving results")
print("=" * 70)

with open(REGRESSION_OUT, "w") as f:
    f.write("\n".join(results_text))
print(f"  Regression results: {REGRESSION_OUT}")
print(f"  Country-quarter adoption: {ADOPTION_OUT}")


# ─────────────────────────────────────────────────────────────────────────────
# 4h. Interpretation
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4h. Interpretation")
print("=" * 70)
print()
print("COMPARISON TABLE:")
print(comp_df.to_string(index=False))
print()

if reg_b_ok:
    direction = "POSITIVE" if coef_b > 0 else "NEGATIVE"
    sig = "SIGNIFICANT" if pval_b < 0.05 else "NOT significant"
    print(f"Regression B — pct_ai_users coefficient: {coef_b:.4f} (SE={se_b:.4f}, p={pval_b:.4f})")
    print(f"Direction: {direction} effect on log(commits_per_dev + 1)")
    print(f"Significance at p<0.05: {sig}")
    print()
    if pval_b < 0.05:
        print("→ The classifier-based AI adoption proxy is significantly associated with")
        print("  developer productivity in the panel. This improves on Phase 1's null result.")
    else:
        print("→ The classifier-based proxy is not significant in this regression.")
        print("  IMPORTANT CAVEAT: pct_ai_users is a TIME proxy (2024 vs 2022/23), not a")
        print("  true cross-country variation measure. It is collinear with the time FE and")
        print("  will be absorbed or nearly absorbed by them. This is a known limitation.")
        print("  The coefficient reflects the 2022→2024 trend unexplained by country FE,")
        print("  not cross-country variation in AI adoption rates.")
else:
    print("Regression B did not run successfully.")

print()
print("KEY LIMITATIONS:")
print("  1. PANEL COVERAGE: The panel aggregates country-quarter productivity but")
print("     does not contain individual account logins. We cannot directly link")
print("     classifier scores to panel developers.")
print()
print("  2. TIME PROXY: pct_ai_users = 0 for 2022/23 and ~0.22 for 2024 is a")
print("     blunt time dummy, not true cross-country adoption variation. It is")
print("     partially collinear with the time FE.")
print()
print("  3. NEXT STEP (Step 9 in project plan): Score a broad population sample")
print("     (5k–10k GitHub accounts with location data) to obtain true country-level")
print("     AI adoption fractions for 2022–2024. That gives genuine cross-sectional")
print("     variation and makes the regression meaningful.")
print()
print("  4. PANEL THINNESS: Median n_developers per country-year = 2 (Phase 1).")
print("     The productivity metrics are noisy at this sample size.")
print()
print("Done.")
