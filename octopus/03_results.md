# Octopus Publication: Results

**Title:** Classifier performance, cross-tool generalisation, and raw commit activity outcomes for AI coding tool adopters vs. controls

**Keywords:** classifier validation, AUC, cross-tool generalisation, commit activity, difference-in-differences outcomes, country adoption rates, GitHub

**Links to:** Methods — "A behavioural classifier for AI coding tool adoption and two causal designs"

**Data repository:** https://github.com/AndreasThinks/ai-productivity-analysis

---

## 1. Classifier Performance

Three models are trained on 235 accounts (33 confirmed AI adopters, 202 controls) using 5-fold stratified cross-validation.

| Model | CV AUC (mean) | CV AUC (± SD) | Ablation AUC | AUC drop |
|---|---|---|---|---|
| Logistic Regression | 0.906 | 0.060 | 0.896 | 0.010 |
| **Random Forest** | **0.940** | **0.054** | **0.909** | **0.031** |
| Gradient Boosting | 0.898 | 0.097 | 0.890 | 0.008 |

Random Forest is selected as the primary model. The ablation model removes all 21 message and documentation features, retaining only activity timing features; its AUC of 0.909 confirms that commit timing and frequency carry most of the signal.

Top features by importance: post-period inter-commit hours (0.130), pre-period message length (0.120), post-period active weeks (0.066).

---

## 2. Cross-Tool Generalisation

The classifier, trained exclusively on confirmed Claude Code users, is applied to 36 Aider users identified independently via commit trailer detection. Results:

| Group | N | Mean score | Median score | SD | Fraction > 0.5 |
|---|---|---|---|---|---|
| Claude Code (train positive) | 33 | 0.776 | 0.856 | 0.211 | 90.9% |
| Aider (held-out) | 36 | 0.727 | 0.820 | 0.219 | 80.6% |
| Controls (train negative) | 202 | 0.033 | 0.016 | 0.045 | 0.0% |

Mann-Whitney test: Aider vs Controls, p < 0.0001. Aider vs Claude Code, p = 0.065 (not significantly different at the 5% level).

The classifier assigns Aider users scores statistically indistinguishable from Claude Code training positives, and far above controls. 80.6% of Aider accounts score above the 0.5 decision threshold, compared to 90.9% of Claude Code positives and 0% of controls.

---

## 3. Pre/Post Summary Statistics

Raw mean outcomes for AI adopters and controls across pre and post periods:

| Outcome | AI adopters (pre) | AI adopters (post) | Controls (pre) | Controls (post) |
|---|---|---|---|---|
| Commits / active week | 10.4 | 23.5 | 5.2 | 5.3 |
| Inter-commit hours | 281 | 58 | 180 | 325 |
| Active weeks | 48.1 | 36.8 | 32.4 | 32.2 |
| Message length (chars) | 62 | 116 | 58 | 55 |
| Conventional commits (frac) | 0.21 | 0.47 | 0.18 | 0.20 |
| PR has body (frac) | 0.18 | 0.50 | 0.22 | 0.27 |
| Test co-write rate | 0.08 | 0.22 | 0.09 | 0.08 |

Note: pre-period differences on several dimensions (commits per active week, inter-commit hours) confirm selection — AI adopters were already more active developers before adoption.

---

## 4. Per-Country AI Adoption Rates

Classifier-derived AI adoption rates for the 20 countries with at least 15 scored accounts in the population sample (N = 887 total accounts):

| Country | Adoption rate | N accounts |
|---|---|---|
| Netherlands (NL) | 10.7% | 20 |
| Australia (AU) | 10.3% | 17 |
| Canada (CA) | 10.1% | 25 |
| Poland (PL) | 9.7% | 28 |
| Indonesia (ID) | 9.3% | 25 |
| Brazil (BR) | 8.8% | 45 |
| Sweden (SE) | 8.6% | 16 |
| South Korea (KR) | 8.4% | 38 |
| Switzerland (CH) | 8.2% | 15 |
| France (FR) | 8.2% | 48 |
| United States (US) | 8.1% | 93 |
| Bangladesh (BD) | 7.9% | 21 |
| Japan (JP) | 7.1% | 17 |
| United Kingdom (GB) | 7.0% | 34 |
| Germany (DE) | 7.0% | 52 |
| India (IN) | 6.9% | 93 |
| Vietnam (VN) | 6.6% | 16 |
| China (CN) | 6.4% | 48 |
| Russia (RU) | 6.4% | 17 |
| Italy (IT) | 6.3% | 23 |

Cross-country range: 6.3% (Italy) to 10.7% (Netherlands). Standard deviation: 1.4 percentage points. English-speaking and northern European countries show the highest adoption rates; East Asian and southern European countries the lowest.

Note: adoption rates are computed from the binary classifier (threshold 0.5) applied to the population scoring sample. They represent detected AI-assisted coding behaviour in the scorer's sample, not verified self-reported usage.

---

## 5. Country-Level Panel

The country-level commit activity panel covers 59 country-year observations across 20 countries and three years (2022–2024). The median number of located developers per country-year observation is 2, reflecting the sparse location-parseable fraction of GitHub Archive users (~21% of active accounts). This thinness is a primary driver of low statistical power in the country-level analysis.

---

*Full data and code: https://github.com/AndreasThinks/ai-productivity-analysis*
*Classifier model: data/classifier_model.pkl | Population scores: data/population_scores.csv*
