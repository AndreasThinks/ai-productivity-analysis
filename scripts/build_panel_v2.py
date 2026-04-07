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
# 4e. Build country-level AI adoption fraction from population scrape
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("4e. Building country-level AI adoption fraction from population scrape")
print("=" * 70)

POP_SCORES_V1  = os.path.join(DATA, "population_scores.csv")
POP_SCORES_V2  = os.path.join(DATA, "population_scores_v2.csv")
MIN_ACCOUNTS   = 15   # minimum scored accounts per country to include in regression C

# Load v1 and v2 population scores, deduplicate by login
scores_v1 = pd.read_csv(POP_SCORES_V1) if os.path.exists(POP_SCORES_V1) else pd.DataFrame()
scores_v2 = pd.read_csv(POP_SCORES_V2) if os.path.exists(POP_SCORES_V2) else pd.DataFrame()

print(f"  Population scores v1: {len(scores_v1)} accounts")
print(f"  Population scores v2: {len(scores_v2)} accounts")

scores_all = pd.concat([scores_v1, scores_v2], ignore_index=True)
scores_all = scores_all.drop_duplicates(subset="login", keep="first")
print(f"  Combined unique accounts: {len(scores_all)}")

# Per-country mean of post_classifier_score (post-period behaviour = the IV)
country_adoption = (
    scores_all.groupby("country")
    .agg(
        mean_ai_score=("post_classifier_score", "mean"),
        n_accounts=("login", "count"),
    )
    .reset_index()
)

# Countries meeting minimum threshold
country_adoption_filtered = country_adoption[
    country_adoption["n_accounts"] >= MIN_ACCOUNTS
].copy()

print(f"\n  Countries with >= {MIN_ACCOUNTS} accounts: {len(country_adoption_filtered)}")
print(country_adoption_filtered.sort_values("mean_ai_score", ascending=False).to_string(index=False))

# Build IV: pct_ai_users = 0 in 2022/2023 (pre-launch), = country mean score in 2024
# This gives cross-country AND cross-time variation (unlike the old global-mean proxy).
country_score_map = dict(zip(
    country_adoption_filtered["country"],
    country_adoption_filtered["mean_ai_score"]
))
country_n_map = dict(zip(
    country_adoption_filtered["country"],
    country_adoption_filtered["n_accounts"]
))

def get_pct_ai(row):
    if row["year"] < 2024:
        return 0.0
    return country_score_map.get(row["country"], float("nan"))

def get_pct_ai_n(row):
    if row["year"] < 2024:
        return 0
    return country_n_map.get(row["country"], 0)

panel["pct_ai_users"]   = panel.apply(get_pct_ai, axis=1)
panel["pct_ai_users_n"] = panel.apply(get_pct_ai_n, axis=1)

# Also keep the old time-proxy for Regression B (for comparison)
preds = pd.read_csv(PREDICTIONS_CSV)
mean_2024_ai_score   = preds["predicted_prob"].mean()
frac_above_threshold = (preds["predicted_prob"] > 0.5).mean()
panel["pct_ai_users_proxy"] = panel["year"].map({2022: 0.0, 2023: 0.0, 2024: float(mean_2024_ai_score)})

# Save adoption CSV (per-country real IV)
cq_adoption = (
    panel[["country", "year", "quarter", "pct_ai_users", "pct_ai_users_n"]]
    .drop_duplicates()
    .sort_values(["country", "year"])
)
cq_adoption.to_csv(ADOPTION_OUT, index=False)
print(f"\n  Country-quarter adoption data saved: {ADOPTION_OUT}")
print(f"  Rows: {len(cq_adoption)}")
print(f"\n  NOTE: pct_ai_users = 0 for 2022/2023 (pre-launch), per-country mean")
print(f"  post_classifier_score for 2024. Only countries with >= {MIN_ACCOUNTS} accounts")
print(f"  contribute to Regression C — all others get NaN and are dropped.")


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

# ── Regression B: Phase 2 (classifier-based time proxy — kept for comparison) ─
print()
print("  Regression B: Phase 2 (pct_ai_users_proxy — time-based proxy, BROKEN, for comparison only)")

panel_year_proxy = (
    panel_clean.groupby(["country", "year"])
    .agg(
        log_commits=("log_commits", "mean"),
        log_prs=("log_prs", "mean"),
        log_events=("log_events", "mean"),
        pct_ai_users=("pct_ai_users_proxy", "mean"),
        n_developers=("n_developers", "sum"),
    )
    .reset_index()
    .set_index(["country", "year"])
)

try:
    mod_b = PanelOLS.from_formula(
        "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
        data=panel_year_proxy,
        drop_absorbed=True,
    )
    res_b = mod_b.fit(cov_type="clustered", cluster_entity=True)
    coef_b = res_b.params["pct_ai_users"]
    se_b   = res_b.std_errors["pct_ai_users"]
    pval_b = res_b.pvalues["pct_ai_users"]
    r2_b   = res_b.rsquared
    n_b    = int(res_b.nobs)
    print(f"  B — N={n_b}, coef={coef_b:.4f}, SE={se_b:.4f}, p={pval_b:.4f}, R²={r2_b:.4f}")
    print(f"  (expected singular/garbage — this is the broken time-proxy for reference)")
    results_text.append("REGRESSION B — Phase 2 (classifier-based IV, time proxy — BROKEN BASELINE)")
    results_text.append("NOTE: pct_ai_users_proxy is collinear with time FE. Coefficient is meaningless.")
    results_text.append(str(res_b.summary))
    results_text.append("")
    comparison_rows.append({
        "Model": "B — Phase 2 (time proxy, broken)",
        "IV": "pct_ai_users_proxy (global mean in 2024)",
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
        "Model": "B — Phase 2 (time proxy, broken)",
        "IV": "pct_ai_users_proxy (global mean in 2024)",
        "N": "N/A",
        "Coef": "N/A",
        "SE": "N/A",
        "p-value": "N/A",
        "R²": "N/A",
    })

# ── Regression C: Phase 2 REAL IV (per-country adoption from population scrape) ─
print()
print("  Regression C: Phase 2 REAL IV (per-country mean classifier score)")
print(f"  Using countries with >= {MIN_ACCOUNTS} scored accounts only")

# Filter panel to countries in our scored set + drop NaN pct_ai_users (countries without scores)
panel_c_clean = panel_clean.dropna(subset=["pct_ai_users"]).copy()
panel_c_clean = panel_c_clean[panel_c_clean["country"].isin(country_score_map.keys())]

panel_year_c = (
    panel_c_clean.groupby(["country", "year"])
    .agg(
        log_commits=("log_commits", "mean"),
        log_prs=("log_prs", "mean"),
        log_events=("log_events", "mean"),
        pct_ai_users=("pct_ai_users", "mean"),
        n_developers=("n_developers", "sum"),
    )
    .reset_index()
    .set_index(["country", "year"])
)

print(f"  Panel C: {len(panel_year_c)} country-year obs, {panel_year_c.reset_index()['country'].nunique()} countries")

coef_c = se_c = pval_c = r2_c = n_c = None
reg_c_ok = False

if len(panel_year_c) < 10:
    print("  Too few observations for regression C — need more countries at >= 15 accounts")
    results_text.append("REGRESSION C: Skipped — insufficient country overlap between panel and scored accounts")
    results_text.append("")
    comparison_rows.append({
        "Model": "C — Phase 2 (REAL per-country IV)",
        "IV": f"pct_ai_users (pop. scrape, n>={MIN_ACCOUNTS})",
        "N": "insufficient",
        "Coef": "N/A", "SE": "N/A", "p-value": "N/A", "R²": "N/A",
    })
else:
    try:
        mod_c = PanelOLS.from_formula(
            "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
            data=panel_year_c,
            drop_absorbed=True,
        )
        res_c = mod_c.fit(cov_type="clustered", cluster_entity=True)
        coef_c = res_c.params["pct_ai_users"]
        se_c   = res_c.std_errors["pct_ai_users"]
        pval_c = res_c.pvalues["pct_ai_users"]
        r2_c   = res_c.rsquared
        n_c    = int(res_c.nobs)
        print(f"  C — N={n_c}, coef={coef_c:.4f}, SE={se_c:.4f}, p={pval_c:.4f}, R²={r2_c:.4f}")
        results_text.append("REGRESSION C — Phase 2 REAL IV (per-country classifier score from population scrape)")
        results_text.append(f"Countries included: {sorted(panel_year_c.reset_index()['country'].unique().tolist())}")
        results_text.append(f"Min accounts per country: {MIN_ACCOUNTS}")
        results_text.append(str(res_c.summary))
        results_text.append("")
        comparison_rows.append({
            "Model": "C — Phase 2 (REAL per-country IV)",
            "IV": f"pct_ai_users (pop. scrape, n>={MIN_ACCOUNTS})",
            "N": n_c,
            "Coef": f"{coef_c:.4f}",
            "SE": f"{se_c:.4f}",
            "p-value": f"{pval_c:.4f}",
            "R²": f"{r2_c:.4f}",
        })
        reg_c_ok = True
    except Exception as e:
        print(f"  Regression C failed: {e}")
        results_text.append(f"Regression C: FAILED — {e}")
        results_text.append("")
        comparison_rows.append({
            "Model": "C — Phase 2 (REAL per-country IV)",
            "IV": f"pct_ai_users (pop. scrape, n>={MIN_ACCOUNTS})",
            "N": "N/A",
            "Coef": "N/A", "SE": "N/A", "p-value": "N/A", "R²": "N/A",
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

print("Regression B (time proxy — for reference only, DO NOT interpret):")
if reg_b_ok:
    print(f"  coef={coef_b:.4f}, SE={se_b:.4f}, p={pval_b:.4f} — collinear with time FE, meaningless")
else:
    print("  Did not run.")

print()
print("Regression C (REAL per-country IV — primary result):")
if reg_c_ok:
    direction = "POSITIVE" if coef_c > 0 else "NEGATIVE"
    sig = "SIGNIFICANT (p<0.05)" if pval_c < 0.05 else "not significant (p>0.05)"
    print(f"  coef={coef_c:.4f}, SE={se_c:.4f}, p={pval_c:.4f}, R²={r2_c:.4f}, N={n_c}")
    print(f"  Direction: {direction} effect on log(commits_per_dev + 1)")
    print(f"  Significance: {sig}")
    print()
    if pval_c < 0.05:
        print("→ Countries with higher AI tool adoption (per classifier) show")
        print(f"  {'higher' if coef_c > 0 else 'lower'} developer productivity in 2024 vs their own baseline.")
        print("  This is the first valid Phase 2 result.")
    else:
        print("→ No significant association between per-country AI adoption and productivity.")
        print("  Possible reasons: panel thinness (median 2 devs/country-year), narrow")
        print("  country set (only those with >=15 scored accounts), or genuine null.")
        print("  Increase scrape coverage before concluding.")
else:
    print("  Regression C did not run — insufficient country overlap between panel and scored set.")

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
