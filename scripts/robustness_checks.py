"""
robustness_checks.py — Systematic sensitivity analysis for the Phase 2 panel regressions.

Checks:
  R1. Drop baseline_log_commits control — does the negative sign dissolve?
  R2. Threshold IV: pct_above_0.5 (fraction scoring >0.5) instead of mean score
  R3. Median IV: median classifier score per country
  R4. Alternative DVs: log(prs_per_dev+1), log(total_events_per_dev+1)
  R5. Honest 2024-only OLS cross-section (no FE machinery — what panel is actually doing)
  R6. Country-level permutation placebo with v3 data (1000 permutations)

Run:
  uv run --with linearmodels --with scikit-learn --with pandas --with joblib \
         --with statsmodels python3 -u scripts/robustness_checks.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import joblib
from scipy import stats

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(BASE, "data")

from linearmodels.panel import PanelOLS

# ─────────────────────────────────────────────────────────────────────────────
# Load shared data
# ─────────────────────────────────────────────────────────────────────────────

MIN_ACCOUNTS = 15
MIN_DEVS     = 5

print("Loading data...")

panel = pd.read_csv(os.path.join(DATA, "github_panel_flat.csv"))
panel["year"] = panel["quarter"].str.extract(r"(\d{4})").astype(int)
panel["log_commits"] = np.log1p(panel["commits_per_dev"])
panel["log_prs"]     = np.log1p(panel["prs_per_dev"])
panel["log_events"]  = np.log1p(panel["total_events_per_dev"])

# Load and combine population scores
scores_all = pd.concat([
    pd.read_csv(os.path.join(DATA, f)) if os.path.exists(os.path.join(DATA, f)) else pd.DataFrame()
    for f in ["population_scores.csv", "population_scores_v2.csv", "population_scores_v3.csv"]
], ignore_index=True).drop_duplicates(subset="login", keep="first")

print(f"  Panel: {len(panel)} rows, {panel['country'].nunique()} countries")
print(f"  Population scores: {len(scores_all)} unique accounts")

# Build per-country IV variants
country_stats = (
    scores_all.groupby("country")
    .agg(
        mean_score  =("post_classifier_score", "mean"),
        median_score=("post_classifier_score", "median"),
        pct_above_05=("post_classifier_score", lambda x: (x > 0.5).mean()),
        n_accounts  =("login", "count"),
    )
    .reset_index()
)
country_stats = country_stats[country_stats["n_accounts"] >= MIN_ACCOUNTS].copy()
print(f"  Countries with >= {MIN_ACCOUNTS} accounts: {len(country_stats)}")

# Attach IVs to panel
score_map   = dict(zip(country_stats["country"], country_stats["mean_score"]))
median_map  = dict(zip(country_stats["country"], country_stats["median_score"]))
thresh_map  = dict(zip(country_stats["country"], country_stats["pct_above_05"]))

def map_iv(row, mapping):
    if row["year"] < 2024:
        return 0.0
    return mapping.get(row["country"], float("nan"))

panel["iv_mean"]   = panel.apply(lambda r: map_iv(r, score_map),  axis=1)
panel["iv_median"] = panel.apply(lambda r: map_iv(r, median_map), axis=1)
panel["iv_thresh"] = panel.apply(lambda r: map_iv(r, thresh_map), axis=1)

# Filter to scored countries, drop missing
panel_c = panel.dropna(subset=["iv_mean"]).copy()
panel_c = panel_c[panel_c["country"].isin(score_map.keys())]

# Aggregate to country-year
def agg_panel(df, iv_col):
    agg = (
        df.groupby(["country", "year"])
        .agg(
            log_commits =(  "log_commits", "mean"),
            log_prs     =(  "log_prs",     "mean"),
            log_events  =(  "log_events",  "mean"),
            pct_ai_users=(   iv_col,       "mean"),
            n_developers=("n_developers",  "sum"),
        )
        .reset_index()
    )
    agg = agg[agg["n_developers"] >= MIN_DEVS].copy()
    return agg

py_mean   = agg_panel(panel_c, "iv_mean")
py_median = agg_panel(panel_c.dropna(subset=["iv_median"]), "iv_median")
py_thresh = agg_panel(panel_c.dropna(subset=["iv_thresh"]), "iv_thresh")

# Add baseline control (2022-23 mean per country)
def add_baseline(df):
    pre = df[df["year"] < 2024].groupby("country")["log_commits"].mean().reset_index()
    pre = pre.rename(columns={"log_commits": "baseline"})
    return df.merge(pre, on="country", how="left")

py_mean   = add_baseline(py_mean)
py_median = add_baseline(py_median)
py_thresh = add_baseline(py_thresh)

print()

# ─────────────────────────────────────────────────────────────────────────────
# Helper: run a PanelOLS and return summary row
# ─────────────────────────────────────────────────────────────────────────────

def run_panel(df, formula, label, dv_label="log_commits", weighted=False):
    df2 = df.set_index(["country", "year"]).copy()
    if weighted:
        w = df2[["n_developers"]].copy()
        mod = PanelOLS.from_formula(formula, data=df2, drop_absorbed=True, weights=w)
    else:
        mod = PanelOLS.from_formula(formula, data=df2, drop_absorbed=True)
    res = mod.fit(cov_type="clustered", cluster_entity=True)
    # Extract IV coefficient (first non-intercept param that isn't baseline)
    iv_param = [p for p in res.params.index if p not in ("baseline", "Intercept")][0]
    return {
        "Label":   label,
        "DV":      dv_label,
        "N":       int(res.nobs),
        "Countries": df2.reset_index()["country"].nunique(),
        "Coef":    res.params[iv_param],
        "SE":      res.std_errors[iv_param],
        "p":       res.pvalues[iv_param],
        "R2":      res.rsquared,
    }

rows = []

# ─────────────────────────────────────────────────────────────────────────────
# Reference: replicate C and C-W from build_panel_v2.py
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print("REFERENCE — replicating C and C-W (baseline_log_commits included)")
print("=" * 70)

r = run_panel(py_mean, "log_commits ~ pct_ai_users + baseline + EntityEffects + TimeEffects", "C-ref (mean IV + baseline)")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

# Weighted C-W
r = run_panel(py_mean, "log_commits ~ pct_ai_users + baseline + EntityEffects + TimeEffects",
              "C-W-ref (mean IV + baseline, weighted)", weighted=True)
rows.append(r)
r_cw = r
print(f"  {r_cw['Label']}: coef={r_cw['Coef']:.4f}, SE={r_cw['SE']:.4f}, p={r_cw['p']:.4f}, N={r_cw['N']}")

# ─────────────────────────────────────────────────────────────────────────────
# R1. Drop baseline_log_commits
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R1. Drop baseline_log_commits control")
print("=" * 70)

r = run_panel(py_mean, "log_commits ~ pct_ai_users + EntityEffects + TimeEffects", "R1a: mean IV, no baseline")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_mean, "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
              "R1b: mean IV, no baseline, weighted", weighted=True)
rows.append(r)
r_nb = r
print(f"  {r_nb['Label']}: coef={r_nb['Coef']:.4f}, SE={r_nb['SE']:.4f}, p={r_nb['p']:.4f}, N={r_nb['N']}")

# ─────────────────────────────────────────────────────────────────────────────
# R2. Threshold IV (pct_above_0.5)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R2. Threshold IV: fraction of accounts scoring > 0.5")
print("=" * 70)

r = run_panel(py_thresh, "log_commits ~ pct_ai_users + baseline + EntityEffects + TimeEffects",
              "R2a: threshold IV + baseline")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_thresh, "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
              "R2b: threshold IV, no baseline")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

# ─────────────────────────────────────────────────────────────────────────────
# R3. Median IV
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R3. Median classifier score per country as IV")
print("=" * 70)

r = run_panel(py_median, "log_commits ~ pct_ai_users + baseline + EntityEffects + TimeEffects",
              "R3a: median IV + baseline")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_median, "log_commits ~ pct_ai_users + EntityEffects + TimeEffects",
              "R3b: median IV, no baseline")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

# ─────────────────────────────────────────────────────────────────────────────
# R4. Alternative DVs
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R4. Alternative dependent variables")
print("=" * 70)

r = run_panel(py_mean, "log_prs ~ pct_ai_users + baseline + EntityEffects + TimeEffects",
              "R4a: DV=log_prs, mean IV + baseline", dv_label="log_prs")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_mean, "log_events ~ pct_ai_users + baseline + EntityEffects + TimeEffects",
              "R4b: DV=log_events, mean IV + baseline", dv_label="log_events")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_mean, "log_prs ~ pct_ai_users + EntityEffects + TimeEffects",
              "R4c: DV=log_prs, no baseline", dv_label="log_prs")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

r = run_panel(py_mean, "log_events ~ pct_ai_users + EntityEffects + TimeEffects",
              "R4d: DV=log_events, no baseline", dv_label="log_events")
rows.append(r); print(f"  {r['Label']}: coef={r['Coef']:.4f}, SE={r['SE']:.4f}, p={r['p']:.4f}, N={r['N']}")

# ─────────────────────────────────────────────────────────────────────────────
# R5. Honest 2024-only OLS cross-section
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R5. Honest 2024-only OLS cross-section (no panel FE)")
print("=" * 70)

import statsmodels.formula.api as smf

cross24 = py_mean[py_mean["year"] == 2024].copy()
cross24 = cross24.dropna(subset=["pct_ai_users", "log_commits", "baseline"])
print(f"  N = {len(cross24)} countries in 2024 cross-section")

if len(cross24) >= 10:
    # Raw
    ols_raw = smf.ols("log_commits ~ pct_ai_users", data=cross24).fit()
    print(f"  R5a (raw, no controls): coef={ols_raw.params['pct_ai_users']:.4f}, "
          f"SE={ols_raw.bse['pct_ai_users']:.4f}, p={ols_raw.pvalues['pct_ai_users']:.4f}, "
          f"N={int(ols_raw.nobs)}")
    rows.append({"Label":"R5a: 2024 OLS, no controls","DV":"log_commits","N":int(ols_raw.nobs),
                 "Countries":len(cross24),"Coef":ols_raw.params["pct_ai_users"],
                 "SE":ols_raw.bse["pct_ai_users"],"p":ols_raw.pvalues["pct_ai_users"],
                 "R2":ols_raw.rsquared})

    # With baseline pre-period control
    ols_ctrl = smf.ols("log_commits ~ pct_ai_users + baseline", data=cross24).fit()
    print(f"  R5b (+ baseline control): coef={ols_ctrl.params['pct_ai_users']:.4f}, "
          f"SE={ols_ctrl.bse['pct_ai_users']:.4f}, p={ols_ctrl.pvalues['pct_ai_users']:.4f}, "
          f"N={int(ols_ctrl.nobs)}")
    rows.append({"Label":"R5b: 2024 OLS + baseline","DV":"log_commits","N":int(ols_ctrl.nobs),
                 "Countries":len(cross24),"Coef":ols_ctrl.params["pct_ai_users"],
                 "SE":ols_ctrl.bse["pct_ai_users"],"p":ols_ctrl.pvalues["pct_ai_users"],
                 "R2":ols_ctrl.rsquared})

    # Weighted
    ols_w = smf.wls("log_commits ~ pct_ai_users + baseline",
                    data=cross24, weights=cross24["n_developers"]).fit()
    print(f"  R5c (weighted + baseline): coef={ols_w.params['pct_ai_users']:.4f}, "
          f"SE={ols_w.bse['pct_ai_users']:.4f}, p={ols_w.pvalues['pct_ai_users']:.4f}, "
          f"N={int(ols_w.nobs)}")
    rows.append({"Label":"R5c: 2024 WLS + baseline","DV":"log_commits","N":int(ols_w.nobs),
                 "Countries":len(cross24),"Coef":ols_w.params["pct_ai_users"],
                 "SE":ols_w.bse["pct_ai_users"],"p":ols_w.pvalues["pct_ai_users"],
                 "R2":ols_w.rsquared})

# ─────────────────────────────────────────────────────────────────────────────
# R6. Country-level permutation placebo (1000 permutations)
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("R6. Country-level permutation placebo (1000 permutations)")
print("=" * 70)

np.random.seed(42)

# Use the 2024 cross-section for permutation test — honest about what we're testing
cross24_perm = cross24.copy()
observed_coef = ols_ctrl.params["pct_ai_users"]  # use baseline-controlled OLS
print(f"  Observed coef (R5b): {observed_coef:.4f}")

perm_coefs = []
countries = cross24_perm["country"].values
iv_vals   = cross24_perm["pct_ai_users"].values

for _ in range(1000):
    shuffled = np.random.permutation(iv_vals)
    cross24_perm = cross24.copy()
    cross24_perm["pct_ai_users"] = shuffled
    m = smf.ols("log_commits ~ pct_ai_users + baseline", data=cross24_perm).fit()
    perm_coefs.append(m.params["pct_ai_users"])

perm_coefs = np.array(perm_coefs)
perm_p = np.mean(perm_coefs <= observed_coef)  # one-sided: how often is perm coef more negative?
perm_p_two = np.mean(np.abs(perm_coefs) >= np.abs(observed_coef))  # two-sided

print(f"  Permutation p (one-sided, more negative): {perm_p:.4f}")
print(f"  Permutation p (two-sided): {perm_p_two:.4f}")
print(f"  Null distribution: mean={perm_coefs.mean():.4f}, SD={perm_coefs.std():.4f}")
print(f"  Observed coef is at {stats.percentileofscore(perm_coefs, observed_coef):.1f}th percentile of null")

# ─────────────────────────────────────────────────────────────────────────────
# Summary table
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 70)
print("SUMMARY TABLE")
print("=" * 70)

summary = pd.DataFrame(rows)
summary["Sig"] = summary["p"].apply(lambda p: "***" if p<0.01 else "**" if p<0.05 else "*" if p<0.1 else "")

print(summary[["Label","DV","N","Countries","Coef","SE","p","Sig"]].to_string(index=False, float_format="{:.4f}".format))

# Save
out_path = os.path.join(DATA, "robustness_results.txt")
with open(out_path, "w") as f:
    f.write("ROBUSTNESS CHECKS — AI Productivity Analysis\n")
    f.write(f"Run date: 2026-04-25\n\n")
    f.write(summary[["Label","DV","N","Countries","Coef","SE","p","Sig"]].to_string(index=False))
    f.write(f"\n\nPermutation placebo (R6):\n")
    f.write(f"  Observed coef (2024 OLS + baseline): {observed_coef:.4f}\n")
    f.write(f"  Permutation p (two-sided, 1000 draws): {perm_p_two:.4f}\n")
    f.write(f"  Null SD: {perm_coefs.std():.4f}\n")

print(f"\nResults saved to {out_path}")
print("\nDone.")
