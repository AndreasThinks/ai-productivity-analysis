# Octopus Publication: Analysis

**Title:** Account-level difference-in-differences and country-level panel regression analysis of AI coding tool adoption effects on commit activity

**Keywords:** difference-in-differences, panel regression, FDR correction, robustness checks, placebo test, causal inference, fixed effects, multiple testing

**Links to:** Results — "Classifier performance, cross-tool generalisation, and raw commit activity outcomes"

---

## 1. Account-Level Difference-in-Differences

### Primary Estimates

OLS regression with pre-period control and HC3 robust standard errors. N = 235 (33 treated, 202 controls). Seven outcomes tested simultaneously; Benjamini-Hochberg FDR correction applied across the family.

| Outcome | AI Δ | Ctrl Δ | Reg. coef | SE | 95% CI | p-value | FDR q |
|---|---|---|---|---|---|---|---|
| Commits / active week | +13.1 | +0.0 | **+13.1** | 3.07 | [7.1, 19.1] | <0.001 | <0.001 |
| Inter-commit hours | −224 | +146 | **−275** | 37.6 | [−349, −201] | <0.001 | <0.001 |
| Active weeks | −11.3 | −0.2 | **−11.3** | 2.91 | [−17.0, −5.6] | <0.001 | <0.001 |
| PR has body (frac) | +0.31 | −0.01 | **+0.32** | 0.06 | [0.20, 0.44] | <0.001 | <0.001 |
| Message length (chars) | +54 | −3 | **+54** | 22.4 | [10, 98] | 0.017 | 0.030 |
| Test co-write rate | +0.14 | −0.01 | **+0.14** | 0.06 | [0.02, 0.26] | 0.019 | 0.030 |
| Conventional commits (frac) | +0.26 | +0.02 | +0.21 | 0.13 | [−0.04, 0.46] | 0.117 | 0.137 |

All six significant outcomes at p < 0.05 survive FDR correction (q < 0.05). Conventional commits is not significant after FDR correction.

### Robustness: High-Confidence Positives

Restricting the treated group to 25 high-confidence accounts (those identified via co-author trailer, with a known adoption timestamp):

| Outcome | Main coef | HC coef | HC SE | HC p |
|---|---|---|---|---|
| Commits / active week | +13.1 | +15.7 | 4.12 | <0.001 |
| Inter-commit hours | −275 | −335 | 52.3 | <0.001 |
| Active weeks | −11.3 | −13.2 | 3.87 | <0.001 |
| PR has body (frac) | +0.32 | +0.38 | 0.08 | <0.001 |
| Message length (chars) | +54 | +61 | 28.1 | 0.031 |
| Test co-write rate | +0.14 | +0.17 | 0.08 | 0.037 |

Point estimates are larger for the high-confidence group, consistent with these being more committed adopters.

### Robustness: Winsorised Estimates

Outcomes winsorised at 5th/95th percentiles to assess sensitivity to outliers:

| Outcome | Main coef | Winsorised coef | Direction change? |
|---|---|---|---|
| Commits / active week | +13.07 | +7.95 | No (−39% attenuation) |
| Inter-commit hours | −275.26 | −179.16 | No (−35% attenuation) |
| Active weeks | −11.25 | −9.19 | No (−18% attenuation) |
| Message length (chars) | +54.26 | +30.26 | No (−44% attenuation) |
| Conventional commits (frac) | +0.08 | +0.05 | No (−31% attenuation) |
| PR has body (frac) | +0.32 | +0.28 | No (−13% attenuation) |
| Test co-write rate | +0.14 | +0.11 | No (−21% attenuation) |

The two primary outcomes (commits per active week and inter-commit hours) attenuate by 35–39% under winsorisation, consistent with some influence from high-activity outliers in the treated group. No outcome changes sign. The effects are practically large even under winsorisation — roughly 8 additional commits per active week and a 3-day reduction in inter-commit time — supporting the robustness of the direction while confirming that headline estimates should be treated as upper bounds.

---

## 2. Country-Level Panel Regression

### Specifications

PanelOLS with country and time fixed effects, clustered standard errors at country level. Dependent variable: log(commits_per_dev + 1).

| Specification | Adoption variable | N | Countries | Coef | SE | p |
|---|---|---|---|---|---|---|
| A — Phase 1 Baseline | Oxford Insights AI Readiness Index | 88 | 51 | +0.067 | 0.090 | 0.462 |
| B — Time proxy (degenerate) | Global mean score in 2024 | 111 | 52 | — | — | 0.998 |
| **C — Primary** | Per-country classifier score | **59** | **20** | **−6.06** | **7.96** | **0.451** |

Specification B is collinear with the time fixed effect (adoption is constant across countries in 2024) and is included only as a diagnostic foil. Specification C is the primary result.

### Robustness Checks

**Leave-one-out validation.** Regression C re-estimated 20 times, each time dropping one country. Coefficient ranges from −7.2 (dropping Netherlands, highest adoption) to −2.1 (dropping India, largest sample). Sign remains negative in all 20 specifications. p-values range from 0.31 to 0.58.

**Classifier threshold sensitivity.** Regression C re-run at adoption thresholds of 0.3, 0.4, 0.6, and 0.7. The coefficient on the adoption variable remains statistically insignificant across all specifications (p > 0.35 in all cases). At the most liberal threshold (0.3), the point estimate moves closer to zero (−1.8, SE = 4.2).

**Placebo test.** Country-level adoption rates are randomly permuted across countries 1,000 times, re-estimating Regression C each time. The placebo distribution has mean −0.13 and SD 5.37. The 90% placebo interval spans [−8.76, +8.70]. Our observed coefficient of −6.06 falls within this interval: 26.9% of random permutations produce a coefficient at least as extreme in absolute value (permutation p = 0.269). The null result is fully consistent with the sampling variability arising from the narrow cross-country variation in adoption rates and the thin productivity panel.

**Parallel trends (Specification A only).** For the Oxford Insights baseline, pre-treatment (2022–2023) trends show no statistically significant difference between high- and low-AI-readiness countries (p = 0.34 for the year × AI-readiness interaction). This supports parallel trends for Specification A; the assumption cannot be formally tested for Specification C.

### Power Analysis

The panel has 59 observations across 20 countries and 3 time periods. With roughly 3 observations per country-cluster, effective degrees of freedom are limited. The cross-country range in adoption rates spans only 4.4 percentage points (6.3% to 10.7%), reducing the signal available for identifying an effect. A back-of-envelope power calculation (Cohen's d = 0.5, 80% power, α = 0.05) requires approximately 64 observations per group — more than the total panel observations available. The null result should be interpreted as reflecting insufficient power rather than the absence of an underlying effect.

---

*Code: scripts/run_analysis.py, notebooks/research_paper_v2.ipynb*
*Full regression output: data/regression_results_v2.txt*
