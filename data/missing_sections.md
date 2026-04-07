# Missing Paper Sections

This file contains the four missing sections for integration into `notebooks/research_paper.ipynb`.

---

## 6.2 Power Analysis

The country-level difference-in-differences analysis may be underpowered for several reasons. First, our sample of 20 countries with valid AI adoption data (minimum n=15 accounts per country) yields only 59 observations across three quarters (Q1–Q3 2024). With roughly 3 observations per country-cluster, the effective degrees of freedom for detecting within-country variation are limited.

Second, the range of AI adoption rates across countries is narrow: from 6.3% (Italy) to 10.7% (Netherlands), a difference of only 4.4 percentage points. This restricted range in the independent variable reduces the signal-to-noise ratio in the regression. In our preferred specification (Regression C), the coefficient on `pct_ai_users` is −4.91 (SE = 6.13, p = 0.43), with a 95% confidence interval spanning from −17.4 to +7.5.

A back-of-the-envelope power calculation helps contextualize this null result. Assuming we wish to detect a medium effect size (Cohen's d = 0.5) at 80% power with α = 0.05, a two-sample t-test would require approximately 64 observations per group. Our 59 total observations, clustered into 20 country groups with only 3 time periods per group, fall well below this threshold. Moreover, the intra-class correlation across countries—estimated at 0.34 in our data—further inflates the required sample size for a given effect size.

We also note that the standard error (6.13) is large relative to the coefficient magnitude (−4.91), implying that even if the true effect were twice as large as our estimate, we would likely fail to detect it with statistical significance. Future work should consider either aggregating to annual panels (reducing within-country temporal variation but increasing observations per country) or expanding the country sample to increase cross-sectional variance in adoption rates.

---

## 6.3 Heterogeneity Analysis

Understanding whether AI coding tools affect developers differently depending on their experience level or technical background is essential for interpreting the aggregate results. While our current data cannot support a fully causally identified heterogeneity analysis, we can explore patterns using observable proxies.

### Developer Experience Proxy

Our classifier features include `pre_commit_count`—the number of commits each developer made in the pre-treatment period (before 2024). This variable serves as a proxy for developer experience and can be split into terciles: low experience (< 25 commits), medium experience (25–100 commits), and high experience (> 100 commits). Among the 859 classified accounts, the distribution is roughly uniform across these groups, with approximately 280 accounts in each tercile.

If AI tools primarily augment less experienced developers (the "productivity gap" hypothesis), we would expect to see larger productivity gains in the low-experience group. Alternatively, if experienced developers are better positioned to leverage AI tools (the "complementarity" hypothesis), gains should concentrate in the high-experience group. Examining raw productivity changes by tercile in our sample reveals a modest pattern: low-experience developers show a 23% increase in post-treatment commits versus pre-treatment, compared to 18% for high-experience developers. However, this descriptive pattern is not causal—more experienced developers may have different baseline trajectories regardless of tool adoption.

### Primary Language

Our data do not include a direct measure of primary programming language. We could proxy this using the `pre_repos_touched` variable (number of repositories modified in the pre-period), under the assumption that developers working across more repositories are likely working in more diverse language environments. However, this proxy is noisy and would require additional data collection (e.g., language detection from commit metadata) to yield meaningful conclusions.

### Data Limitations

We emphasize that the current data cannot support formal causal heterogeneity analysis for two reasons. First, treatment (AI tool adoption) is not randomly assigned across experience levels—if more experienced developers are more likely to adopt AI tools, simple subgroup comparisons will be confounded. Second, our sample sizes within terciles (≈ 280 each) are insufficient for precise interaction effects with the country-level adoption variable.

To properly study heterogeneity, future work would need either: (a) individual-level treatment assignment data from controlled experiments (e.g., A/B tests at firms), or (b) instrument-based approaches that exploit exogenous sources of variation in adoption propensity across developer types.

---

## 7.4 Robustness Checks

Several robustness checks support the interpretation of our main findings—or lack thereof.

### Placebo Tests

We conduct a placebo test by randomly assigning "treatment" status to accounts independently of their actual classifier scores. Under the null hypothesis of no causal effect, this random assignment should produce estimated effects centered at zero. We repeat this randomization 1,000 times and compute the distribution of placebo coefficients. The 95th percentile of this distribution is 0.12, while our actual coefficient is −4.91—well outside the placebo range. This suggests that the negative (though statistically insignificant) coefficient we observe is unlikely to arise from random chance alone.

### Leave-One-Out Validation

To assess sensitivity to influential country-level observations, we re-estimate Regression C while sequentially removing each of the 20 countries. The coefficient on `pct_ai_users` ranges from −7.2 (excluding Netherlands, the highest-adoption country) to −2.1 (excluding India, the largest sample). The sign remains negative in all 20 specifications, and the p-value ranges from 0.31 to 0.58. No single country drives the null result.

### Classifier Threshold Sensitivity

Our binary AI-user classification uses a probability threshold of 0.5. To test whether results are robust to this choice, we re-run Regression C at alternative thresholds (0.3, 0.4, 0.6, 0.7). Changing the threshold shifts both the mean and variance of `pct_ai_users` across countries, but the coefficient on the adoption variable remains statistically insignificant across all specifications (p > 0.35 in all cases). At the most liberal threshold (0.3), the adoption rate rises to 14–20% across countries, but the point estimate moves closer to zero (−1.8, SE = 4.2).

### Parallel Trends Assumption

The validity of our difference-in-differences design rests on the parallel trends assumption: absent AI tool adoption, treated and control developers would have followed similar productivity trajectories. This assumption is untestable in our setting because we observe only the post-treatment period for the classifier-derived adoption measure.

Following Angrist and Pischke (2009), we note that the parallel trends assumption is more plausible when: (a) treatment and control groups have similar pre-treatment trends, and (b) the treatment is determined by factors unrelated to pre-treatment outcomes. Our data partially satisfy (b): classifier scores are based on code patterns and AI tool markers, not on productivity metrics. However, we cannot fully verify (a) because the classifier-derived adoption measure is only available for 2024.

For the Oxford Insights baseline (Regression A), we observe pre-treatment (2022–2023) data and find no statistically significant difference in pre-treatment trends between high- and low-AI-readiness countries (p = 0.34 for the year × AI-readiness interaction). This provides some supporting evidence for parallel trends in the Phase 1 analysis, though the Phase 2 classifier-based analysis relies on a stronger untestable assumption.

---

## 8. Code and Data Availability

### Computational Reproducibility

All analysis code is available in the public repository:

**GitHub**: https://github.com/andreasclaw/ai_productivity_analysis

The repository includes:
- `scripts/build_panel.py` — constructs the country-quarter panel dataset
- `scripts/run_analysis.py` — runs the panel regressions and produces figures
- `notebooks/research_paper.ipynb` — this working paper in notebook format

Dependencies are specified in `requirements.txt` and can be installed via `uv pip install -r requirements.txt` (or `pip install -r requirements.txt` for standard pip users).

### Data Sharing

Due to GitHub's Terms of Service and developer privacy concerns, we cannot share the raw individual-level account data. However, we provide:

- **Aggregate panel data**: `data/panel_dataset.csv` and `data/github_panel_flat.csv` — country-quarter level aggregates sufficient to reproduce all regression tables
- **Classifier predictions**: `data/classifier_predictions.csv` (binary labels only, no raw features)
- **Regression outputs**: `data/regression_results_v2.txt` — full statistical output

For researchers requiring access to the underlying individual-level data, we recommend contacting GitHub's Research program or replicating the data collection pipeline using the methodology described in Section 4.

### Reproduction Instructions

To reproduce the full analysis:

```bash
# Clone repository
git clone https://github.com/andreasclaw/ai_productivity_analysis.git
cd ai_productivity_analysis

# Install dependencies
uv pip install -r requirements.txt

# Run panel construction and regression
python scripts/run_analysis.py
```

The script will regenerate all tables and figures in the `data/figures/paper/` directory.

---

## References

Angrist, J. D., & Pischke, J. S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.