# Analysis Results — March 2026

**Run date:** 2026-03-25  
**Data:** Full panel scrape (9 quarterly windows, 500 users/window), aggregated to annual  
**Scripts:** `scripts/build_panel.py` + `scripts/run_analysis.py`

---

## What Changed Since the Proof-of-Concept

The original test run used 100 GitHub users per year-end snapshot, producing 32 country-year observations across 24 countries. The overnight scrape collected 500 users across 9 quarterly windows (Q4 2022 – Q4 2024), which after aggregation to annual and merging with Oxford Insights readiness scores yields:

| | Proof-of-concept | This run |
|--|--|--|
| Observations | 32 | 88 |
| Countries | 24 | 51 |
| Time points | 2 (2022, 2024) | 3 (2022, 2023, 2024) |
| Users sampled | 200 total | 4,500 total |

The quarterly data was summed to annual event counts; developer counts were averaged across quarters within each year before computing per-developer rates. Years 2022 and 2024 use Oxford Insights readiness scores from the same year; 2023 GitHub data is matched to the 2023 Oxford score.

---

## Descriptive Statistics

**88 country-year observations, 51 unique countries, years 2022–2024**

| Variable | Mean | Std | Min | Median | Max |
|--|--|--|--|--|--|
| ai_readiness_score | 65.1 | 11.3 | 39.1 | 67.7 | 85.7 |
| commits_per_dev | 3.06 | 2.97 | 0.0 | 2.0 | 17.0 |
| prs_per_dev | 0.59 | 0.88 | 0.0 | 0.27 | 5.0 |
| creates_per_dev | 0.76 | 0.80 | 0.0 | 0.50 | 4.0 |
| comments_per_dev | 0.24 | 0.43 | 0.0 | 0.0 | 2.0 |
| total_events_per_dev | 4.65 | 4.00 | 1.0 | 3.0 | 22.5 |
| n_developers | 3.3 | 3.5 | 1 | 2 | 25 |

Notable: median n_developers per country-year is still only 2, which drives most of the noise in per-developer rates.

---

## OLS Regression

**Model:** `log(total_events_per_dev) ~ ai_readiness_score + C(country)`

| | Value |
|--|--|
| N observations | 88 |
| N countries | 51 |
| R² | 0.326 |
| Adjusted R² | −0.629 |
| F-statistic | 0.34 |
| Prob(F) | 1.00 |

**Key coefficient:**

| Variable | Coef | Std Err | t | p | 95% CI |
|--|--|--|--|--|--|
| ai_readiness_score | 0.1148 | 0.167 | 0.69 | 0.497 | [−0.225, 0.454] |

**Result: not statistically significant** (p = 0.497).

The coefficient is positive and larger than in the proof-of-concept run (was −0.039), but the confidence interval spans zero comfortably. The model has 52 parameters on 88 observations — the country fixed effects are consuming nearly all degrees of freedom, leaving only 36 residual df. This is an identification problem, not a signal problem.

---

## Figures

### Scatter: AI Readiness vs Total Events per Developer

`data/figures/scatter_ai_vs_productivity.png`

Key observations:
- **2024 productivity visibly higher than 2022** across most countries — consistent upward shift, suggesting real change between years.
- **OLS trend slope = 0.031** (raw, without FEs) — very shallow, wide scatter around it.
- **Outliers above trend in 2024:** Austria (AT, ~22 events/dev), South Korea (KR, ~16), Turkey (TR, ~12), Nigeria (NG, ~12). Several of these are small-sample artefacts (1–2 developers).
- **Outliers below trend:** Singapore (SG, ~3.5 despite readiness score of 82), US (~12 despite readiness of 85). US is likely undercounted due to state abbreviations being mis-parsed as country codes.

### Correlation Matrix

`data/figures/correlation_matrix.png`

| Pair | r |
|--|--|
| commits_per_dev ↔ total_events_per_dev | 0.95 |
| prs_per_dev ↔ total_events_per_dev | 0.64 |
| creates_per_dev ↔ total_events_per_dev | 0.52 |
| commits_per_dev ↔ prs_per_dev | 0.46 |
| ai_readiness_score ↔ comments_per_dev | 0.33 |
| ai_readiness_score ↔ commits_per_dev | 0.03 |
| ai_readiness_score ↔ total_events_per_dev | ~0.1 |

Commits dominate the total_events signal (r = 0.95), so commits_per_dev is the highest-power dependent variable. AI readiness has near-zero correlation with commits and only a weak positive correlation with comments — possibly reflecting documentation culture in higher-readiness countries rather than productivity per se.

---

## Interpretation

The null result is robust. With nearly 3× more data than the proof-of-concept, the AI readiness coefficient remains statistically indistinguishable from zero. Three likely explanations:

1. **Genuine null**: country-level AI readiness (as measured by Oxford Insights government readiness index) does not predict individual developer productivity on GitHub. The index measures government/infrastructure capacity, not developer tool adoption.

2. **Measurement mismatch**: what we want is *GenAI tool adoption by developers* (Copilot usage, ChatGPT penetration among coders). Oxford Insights measures something more like national AI policy readiness. These may be weakly correlated at best.

3. **Statistical power still inadequate**: median 2 developers per country-year makes per-developer rates extremely noisy. A single active developer in a country swings the metric by several units.

The year-over-year productivity increase visible in the scatter is interesting and worth investigating separately — it could reflect real AI-driven gains, but it's a time trend not a cross-country effect.

---

## Limitations

- **Tiny per-country samples**: median n_developers = 2. A single prolific developer in a small country can dominate the country average. Weight by n_developers or use minimum-N thresholds before drawing conclusions.
- **Country FE kills identification**: with 51 country dummies on 88 observations, the within-country variation is only 2–3 data points per country. The model is mechanically underpowered.
- **Oxford Insights is a government index**: correlates poorly with developer-level AI tool adoption, which is the actual treatment of interest.
- **US undercounting**: US state abbreviations (CA, NY) are parsed as Canada / New York by the location normaliser. US n_developers is likely materially understated.
- **Location hit rate declining**: 30.2% in Q4 2022 → 23.4% in Q4 2024. If developers who remove locations differ systematically in productivity, this introduces attrition bias.

---

## Next Steps

Priority order:

1. **Get a better AI adoption measure**: Semrush/World Bank ChatGPT traffic data (Liu & Wang 2025) is the right independent variable — it measures actual GenAI tool usage per country, not government readiness. Chase the author replication data or use Google Trends "ChatGPT" as a free proxy.

2. **Fix US location parsing**: extend the COUNTRY_MAP in `build_panel.py` to handle US city/state strings (e.g. "San Francisco, CA" → US). This is the most addressable data quality issue.

3. **Minimum N threshold**: only include country-years with ≥ 5 located developers in the regression. Cuts some observations but eliminates the single-developer noise driving outliers like KE and NG.

4. **Switch to `linearmodels.PanelOLS`** with both country and time fixed effects once the panel is cleaner. The current run uses statsmodels OLS with manual country dummies; `PanelOLS` handles the double-FE structure and clustered SEs more cleanly.

5. **Scale further**: 500 users/window still yields median 2 developers/country/year. Either increase to ~2000 users/window or run multiple hourly windows per quarter to get ≥ 10 developers per major country per year.
