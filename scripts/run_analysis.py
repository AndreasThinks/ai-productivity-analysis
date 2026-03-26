"""
run_analysis.py
---------------
Loads the panel dataset and runs:
  1. Descriptive statistics
  2. OLS regression: log(total_events_per_dev) ~ ai_readiness_score + country FE
  3. Correlation matrix heatmap  → data/figures/correlation_matrix.png
  4. Scatter plot AI readiness vs productivity  → data/figures/scatter_ai_vs_productivity.png
  5. Regression summary  → data/regression_results.txt
"""

import pathlib
import warnings

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")

ROOT = pathlib.Path(__file__).parent.parent
DATA = ROOT / "data"
FIGURES = DATA / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
panel = pd.read_csv(DATA / "panel_dataset.csv")
panel["log_total_events_per_dev"] = np.log(panel["total_events_per_dev"].clip(lower=0.01))

print("=" * 60)
print("PANEL DATASET — DESCRIPTIVE STATISTICS")
print("=" * 60)
print(f"Countries covered ({panel['country'].nunique()}): {sorted(panel['country'].unique())}")
print(f"Years covered: {sorted(panel['year'].unique())}")
print(f"Total observations: {len(panel)}")
print()
print("Summary stats per variable:")
print(
    panel[
        ["ai_readiness_score", "commits_per_dev", "prs_per_dev",
         "creates_per_dev", "comments_per_dev", "total_events_per_dev", "n_developers"]
    ].describe().round(3).to_string()
)

# ---------------------------------------------------------------------------
# OLS regression with country fixed effects
# Note: small sample (32 obs, 24 countries) — results are illustrative only
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("OLS REGRESSION: log(total_events_per_dev) ~ ai_readiness_score + country FE")
print("=" * 60)

# Need at least 2 obs per country for FE to absorb country dummies
# Keep all observations; country FE via C(country) in formula
model = smf.ols(
    "log_total_events_per_dev ~ ai_readiness_score + C(country)",
    data=panel,
).fit()

print(model.summary())

# Save regression results
results_path = DATA / "regression_results.txt"
with open(results_path, "w") as f:
    f.write("OLS REGRESSION RESULTS\n")
    f.write("Dependent variable: log(total_events_per_dev)\n")
    f.write("Independent variable: ai_readiness_score\n")
    f.write("Fixed effects: country dummies (C(country))\n")
    f.write(f"N observations: {int(model.nobs)}\n")
    f.write(f"N countries: {panel['country'].nunique()}\n\n")
    f.write(str(model.summary()))
    f.write("\n\n--- KEY COEFFICIENT ---\n")
    if "ai_readiness_score" in model.params:
        coef = model.params["ai_readiness_score"]
        pval = model.pvalues["ai_readiness_score"]
        ci = model.conf_int().loc["ai_readiness_score"]
        f.write(f"ai_readiness_score: coef={coef:.4f}, p={pval:.4f}, 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]\n")
print(f"\nRegression results saved to {results_path}")

# ---------------------------------------------------------------------------
# Key findings summary
# ---------------------------------------------------------------------------
print()
print("=" * 60)
print("KEY FINDINGS")
print("=" * 60)

if "ai_readiness_score" in model.params:
    coef = model.params["ai_readiness_score"]
    pval = model.pvalues["ai_readiness_score"]
    ci = model.conf_int().loc["ai_readiness_score"]
    print(f"AI Readiness → log(events/dev): coef = {coef:.4f}")
    print(f"  p-value = {pval:.4f}  (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
    if pval < 0.05:
        direction = "positive" if coef > 0 else "negative"
        print(f"  Result: STATISTICALLY SIGNIFICANT ({direction} relationship, p<0.05)")
        print(f"  Interpretation: A 1-point increase in AI readiness score is associated with")
        print(f"  a {coef*100:.1f}% change in total events per developer.")
    else:
        print(f"  Result: Not statistically significant (p={pval:.3f} > 0.05)")
        print(f"  Interpretation: No reliable relationship detected in this small sample.")

print(f"\nR² = {model.rsquared:.4f}")
print(f"Adjusted R² = {model.rsquared_adj:.4f}")
print()
print("CAVEATS:")
print("  - Sample is very small (n=32 obs, 24 countries, only 8 countries have 2 time points)")
print("  - GitHub sample used 100 actors per year → ~24-26 have locatable profiles")
print("  - Per-developer metrics are noisy due to low per-country actor counts")
print("  - This is a proof-of-concept; scale data collection before drawing conclusions")

# ---------------------------------------------------------------------------
# Correlation matrix heatmap
# ---------------------------------------------------------------------------
corr_cols = [
    "ai_readiness_score", "commits_per_dev", "prs_per_dev",
    "creates_per_dev", "comments_per_dev", "total_events_per_dev",
]
corr = panel[corr_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    center=0,
    vmin=-1,
    vmax=1,
    ax=ax,
    linewidths=0.5,
)
ax.set_title("Correlation Matrix: AI Readiness & Developer Productivity Metrics", fontsize=11)
plt.tight_layout()
corr_path = FIGURES / "correlation_matrix.png"
fig.savefig(corr_path, dpi=150)
plt.close(fig)
print(f"\nCorrelation matrix heatmap saved to {corr_path}")

# ---------------------------------------------------------------------------
# Scatter plot: AI readiness vs total_events_per_dev
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 7))

colors = {"2022": "#2196F3", "2024": "#FF5722"}
for year, grp in panel.groupby("year"):
    ax.scatter(
        grp["ai_readiness_score"],
        grp["total_events_per_dev"],
        label=str(year),
        color=colors.get(str(year), "grey"),
        s=80,
        alpha=0.8,
        zorder=3,
    )
    for _, row in grp.iterrows():
        ax.annotate(
            f"{row['country']} ({int(row['year'])})",
            (row["ai_readiness_score"], row["total_events_per_dev"]),
            textcoords="offset points",
            xytext=(5, 3),
            fontsize=7,
            alpha=0.85,
        )

# Add trend line
x = panel["ai_readiness_score"]
y = panel["total_events_per_dev"]
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, m * x_line + b, "k--", alpha=0.4, label=f"OLS trend (slope={m:.3f})")

ax.set_xlabel("AI Readiness Score (Oxford Insights)", fontsize=12)
ax.set_ylabel("Total Events per Developer", fontsize=12)
ax.set_title("AI Readiness vs Developer Productivity\n(one point per country-year)", fontsize=13)
ax.legend(title="Year")
ax.grid(True, alpha=0.3)
plt.tight_layout()

scatter_path = FIGURES / "scatter_ai_vs_productivity.png"
fig.savefig(scatter_path, dpi=150)
plt.close(fig)
print(f"Scatter plot saved to {scatter_path}")
