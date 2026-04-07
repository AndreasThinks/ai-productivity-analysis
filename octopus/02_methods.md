# Octopus Publication: Methods

**Title:** A behavioural classifier for AI coding tool adoption and two causal designs for estimating its effects on developer commit activity

**Keywords:** behavioural classifier, random forest, difference-in-differences, panel regression, GitHub Archive, commit history, machine learning, causal inference

**Links to:** Research Problem — "How can AI coding tool adoption be measured at scale from public developer behaviour, and does adoption affect commit activity?"

**Code and data:** https://github.com/AndreasThinks/ai-productivity-analysis

---

## Overview

This publication describes two methodological contributions: (1) a behavioural classifier that identifies AI coding tool users from public GitHub commit history without requiring self-reported adoption data or proprietary telemetry, and (2) two causal designs — an account-level difference-in-differences and a country-level panel regression — that use the classifier to estimate the effects of AI tool adoption on developer commit activity.

---

## Part 1: Behavioural Classifier

### Data Sources

All data derive from GitHub Archive (gharchive.org), a public record of GitHub activity events available from 2011 onwards. We draw on three samples:

- **Classifier training sample.** Twelve hourly windows spanning November 2024, January 2025, and March 2025, yielding approximately 380,000 unique active developer accounts. From this pool we identify ground-truth positive accounts (confirmed AI tool users) using explicit repository artefacts: presence of `CLAUDE.md` files, `.claude/` configuration directories, or `Co-Authored-By: Claude <noreply@anthropic.com>` commit trailers.

- **Population scoring sample.** An additional 887 GitHub accounts with parseable location fields mapping to panel countries, used to derive per-country adoption rates. Each account is scored by the trained classifier.

- **Commit activity panel.** Nine quarterly hourly windows from Q4 2022 through Q4 2024, sampling 500 active developers per window. Used to construct country-level commit activity metrics.

### Ground Truth Labels

**Positive accounts (confirmed AI tool users)** are identified via two routes: (1) GitHub Code Search for repositories containing `CLAUDE.md` in the root, and (2) GH Archive scan for commit messages containing co-author trailers attributable to Claude Code or Aider. Accounts discovered via co-author trailer are assigned `marker_confidence = high` (adoption timestamp known); Code Search accounts are assigned `marker_confidence = low` (adoption date estimated conservatively as repository creation date). Of 33 positive accounts in the final training set, 25 are high-confidence.

**Negative accounts (non-adopters)** are randomly sampled from GH Archive active developers and filtered to accounts with commit activity in both the pre-period (January 2022 – December 2023) and post-period (January 2024 – present), with zero AI tool markers across full commit history.

### Features

We extract 43 behavioural features per account across three categories:

- **Message and documentation quality** (15 features): mean commit message length, fraction of multiline messages, fraction using conventional commit format, fraction mentioning tests, mean PR body length, fraction of PRs with a body.
- **Temporal and activity patterns** (15 features): active weeks, commits per active week, mean inter-commit hours, fraction of burst commits.
- **Temporal change features** (15 features, Δ = post − pre): the difference in each of the above between pre and post periods.

**Critical design constraint.** The explicit artefacts used to identify ground truth (CLAUDE.md files, co-author trailers) are excluded from the classifier feature set. The classifier must learn behavioural patterns correlated with AI adoption without having direct access to the markers used to define the training labels.

### Model Selection and Training

Three models are trained on the 235-account sample (33 positives, 202 negatives) using 5-fold stratified cross-validation: Logistic Regression, Random Forest, and Gradient Boosting. Models are evaluated on cross-validated AUC. Random Forest (CV AUC 0.940 ± 0.054) is selected as the primary model.

### Cross-Tool Validation

The classifier is trained exclusively on confirmed Claude Code users. To test whether it detects general AI-assisted coding behaviour rather than Claude-specific stylistic artefacts, it is applied to 36 Aider users identified independently via commit trailer detection. These accounts were not used in training.

### Writing-Style Ablation

To test whether the classifier is detecting behavioural change or merely Claude's distinctive commit message style, all 21 message and documentation features are removed and the model is re-trained on the remaining 22 activity features. This ablation model achieves AUC 0.909 — a drop of 3.1 points from the full model — confirming that commit timing and frequency carry most of the discriminative signal.

---

## Part 2: Causal Designs

### Account-Level Difference-in-Differences

**Setup.** Confirmed AI tool adopters (N = 33) are treated as the treatment group; controls (N = 202) as the comparison group. For each account we observe commit behaviour in the pre-period (January 2022 – December 2023) and post-period (January 2024 – present).

**Estimator.** For each outcome Y we estimate:

ΔYᵢ = α + β · Treatmentᵢ + γ · Yᵢᵖʳᵉ + εᵢ

where ΔYᵢ is the within-account change, Treatmentᵢ = 1 for AI adopters, and Yᵢᵖʳᵉ controls for baseline differences. Standard errors are HC3 heteroskedasticity-robust. The coefficient β estimates the average treatment effect on the treated: the additional change in the outcome for AI adopters relative to controls, conditional on their pre-period level.

**Outcomes.** Commits per active week (primary), inter-commit hours, active weeks, commit message length, fraction of conventional commits, fraction of PRs with a body, and test co-write rate. All seven outcomes are tested simultaneously; Benjamini-Hochberg FDR correction is applied across the family.

**Identifying assumption.** Parallel trends: absent AI tool adoption, treated and control accounts would have followed the same behavioural trajectory. AI adopters show significantly higher pre-period activity on several dimensions, indicating selection. The regression adjustment controls for pre-period levels but does not fully eliminate selection bias; estimates should be interpreted as upper bounds on the average treatment effect rather than unbiased causal estimates.

**Note on classifier–DiD relationship.** The classifier is trained using features that overlap with the DiD outcomes, creating a mechanical relationship between the two analyses. The DiD should be interpreted as evidence of the direction and magnitude of behavioural change conditional on classifier selection, not as a fully independent causal estimate.

### Country-Level Panel Regression

**Setup.** A country × year panel for 2022–2024 using GH Archive commit activity metrics (commits per located developer) across 20 countries with at least 15 classifier-scored accounts in the population sample.

**Adoption variable.** Per-country AI adoption rates derived from the population scoring sample: the fraction of accounts scoring above 0.5 on the classifier, computed separately for each country. Pre-2024 adoption is set to zero; 2024 values use the per-country classifier-derived rate.

**Model.** PanelOLS with country fixed effects and time fixed effects, clustered standard errors at the country level:

log(commits_per_dev)ᵢₜ = αᵢ + λₜ + β · pct_ai_usersᵢₜ + εᵢₜ

Three specifications are estimated: (A) Oxford Insights Government AI Readiness Index as the adoption variable (Phase 1 baseline); (B) global mean classifier score in 2024 (degenerate, included for reference); (C) per-country classifier-derived adoption rates (primary).

---

*Full code and data at: https://github.com/AndreasThinks/ai-productivity-analysis*
