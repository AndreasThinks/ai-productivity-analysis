# AI Adoption & Developer Productivity

Does AI coding tool adoption causally increase developer productivity? This project tries to find out — using GitHub as the lab.

Two-phase study. Phase 1 built a country-level panel regression linking AI readiness to GitHub activity metrics. Phase 2 (current) is building an account-level binary classifier to identify AI coding tool users from observable behavioural signals — commit patterns, message length, PR quality — without relying on explicit markers.

---

## Status

**Phase 1 — complete (null result).** Fixed-effects panel regression across 51 countries, 2022–2024. AI readiness coefficient p = 0.497 — not significant. Root cause: Oxford Insights measures government AI policy readiness, not whether developers are actually using Claude Code or Copilot today. Wrong independent variable.

**Phase 2 — in progress.** Building a classifier to label individual GitHub accounts as AI coding tool users or not. Ground truth from explicit artefacts (CLAUDE.md files, co-author commit trailers). Behavioural features: commit message length delta, PR description completeness, test co-write rate, conventional commit adoption. Early signal is strong — message length delta ~16x larger for confirmed AI users vs. controls. Classifier trained (RF AUC 0.940), Aider validation passed (0.727).

**Population scrape v3 — running.** Scores random GitHub accounts across 38 countries to build per-country AI adoption fractions. v1 completed 2,048/3,000 before stalling; v3 fixes the rate-limit retry spiral with capped backoff, proactive quota checks, and jitter. Country trim dropped 16 low-value countries (see `country_trim_analysis.md`). Systemd service + cron check-in every 3 hours.

**New analytical tools:**
- `scripts/classifier_placebo_test.py` — tests whether the classifier captures pre-existing developer conscientiousness rather than AI adoption. **Result: significant confound detected (6/8 pre-existing metrics, p<0.05).**
- `scripts/build_panel_v2.py` — now supports weighted regression (by n_developers), minimum-N thresholds, parallel trends diagnostics, and pre-period productivity controls.
- `scripts/retrain_classifier_expanded.py` — merges 41 expansion positives into training set. New model: 74 pos / 202 neg, AUC 0.936.
- `methodology_decisions.md` — living document recording every non-obvious analytical choice with rationale and alternatives considered.

---

## Research Design

### Phase 1 — Country Panel

Fixed-effects regression on a country × quarter panel, Q4 2022 – Q4 2024:

```
log(commits_per_dev) ~ ai_readiness_score + country_FE + time_FE
```

- **Country FE** absorb stable confounders (ecosystem maturity, developer culture)
- **Time FE** absorb global growth trends
- Productivity measured via GH Archive: commits, PRs, repo creation, per located developer
- AI adoption measured via Oxford Insights Government AI Readiness Index

**Results:** 88 country-year observations, 51 countries. Coefficient = 0.115, p = 0.497. Null. The independent variable is the problem — government readiness is three steps removed from a developer opening a coding agent.

### Phase 2 — Account-Level Classifier

A difference-in-differences design at the account level.

**Ground truth labelling:** GitHub Code Search for `CLAUDE.md` files + GH Archive scan for `Co-Authored-By: Claude` commit trailers → confirmed positive accounts.

**Behavioural features (no label leakage):**
- Δ commit message length (pre vs. post Nov 2023)
- Δ fraction of conventional commits
- Δ PR description completeness
- Δ test co-write rate
- Δ multiline commit fraction
- Burst commit patterns, hour-of-day entropy

**Explicit markers explicitly excluded from features** — CLAUDE.md presence, co-author trailer content, keyword matches on "Claude"/"Anthropic". These create the labels; they cannot also be classifier inputs.

**Model:** logistic regression first (interpretability), then gradient boosted trees.

**Integration:** replace `ai_readiness_score` in Phase 1 panel with fraction of located developers classified as AI users per country-quarter. Rerun PanelOLS.

---

## Preliminary Results (Phase 2 Test Runs — March 2026)

Two test scrapes completed (scraper v1 and v2.0, 40 accounts each). Both-window comparison (≥10 commits pre and post adoption): 4 positives, 20 negatives.

| Feature | AI users (Δ) | Controls (Δ) | Ratio |
|---------|-------------|--------------|-------|
| Mean message length | +53.8 chars | -3.4 chars | ~16x |
| Frac conventional commits | +0.26 | +0.02 | ~15x |
| Frac PR has body | +0.69 | +0.05 | ~15x |
| Test co-write rate | +0.19 | -0.01 | ~15x |

**Coverage caveat:** only 4 of 20 positive accounts pass the both-window threshold. Code Search (`filename:CLAUDE.md`) systematically surfaces recent adopters with thin pre-adoption history. Scraper v2.1 addresses this with multi-hour GH Archive co-author discovery (earlier adopters, more pre-window history) and a `marker_confidence` field to stratify by temporal split reliability. Full run (200 pos / 200 neg) pending.

---

## Data

### GitHub Productivity (GH Archive + GitHub API)

- 9 quarterly windows, Q4 2022 – Q4 2024
- 500 developers sampled per window; 26.3% location hit rate
- 54 countries covered
- Events: PushEvent (commits), PullRequestEvent, CreateEvent, IssueCommentEvent, ReleaseEvent
- All metrics normalised per located developer

### AI Adoption (Phase 1)

- Oxford Insights Government AI Readiness Index, 2021–2023
- Stanford HAI AI Index (supplementary)

Oxford Insights data is **not included** in this repo — download from [oxfordinsights.com](https://oxfordinsights.com/ai-readiness/ai-readiness-index/). Stanford HAI CSVs are included under `data/stanford_hai/`.

### GH Archive Cache

Cached scrape files (`data/classifier_cache*/`, `data/gharchive_*.jsonl`) are **not committed** — too large and keyed to specific run dates. Re-run the scraper scripts to reproduce.

---

## Reproducing

### Requirements

```bash
# All scripts use uv — install from https://docs.astral.sh/uv/
uv run scripts/scrape_github_panel.py   # Phase 1 panel scrape
uv run scripts/build_panel.py           # Merge GH data + AI readiness index
uv run scripts/run_analysis.py          # Fixed-effects regression

uv run scripts/scrape_classifier_full.py  # Phase 2 classifier scrape v2.1 (TEST_RUN=False for full run)
```

### Environment variables

```bash
export GITHUB_TOKEN=your_pat_here   # GitHub personal access token (5000 req/hr)
```

No other credentials required. Oxford Insights data must be downloaded separately and placed in `data/oxford_insights/`.

---

## File Structure

```
ai_productivity_analysis/
├── README.md
├── project_plan.md                         ← detailed methodology + findings (living doc)
├── country_trim_analysis.md                ← data-driven country drop recommendations
├── methodology_decisions.md                ← audit trail of analytical choices
├── data_source_assessment.md               ← AI adoption data source evaluation
├── analysis_results_march2026.md           ← full Phase 1 write-up
├── data/
│   ├── panel_dataset.csv                   ← 88 obs × 9 cols, Phase 1 panel
│   ├── github_panel_flat.csv               ← 347 rows, country × quarter
│   ├── regression_results.txt              ← OLS summary
│   ├── regression_results_v2.txt           ← PanelOLS with classifier IV
│   ├── classifier_test_features.csv        ← Phase 2 test run features (40 accounts)
│   ├── classifier_full_features.csv        ← Phase 2 full run features (235 accounts)
│   ├── classifier_model.pkl                ← trained RF classifier (33 pos)
│   ├── classifier_model_expanded.pkl       ← retrained RF classifier (74 pos)
│   ├── classifier_predictions_expanded.csv ← CV predictions for expanded model
│   ├── placebo_test_results.csv            ← pre-period placebo test output
│   ├── population_scores_v3.csv            ← v3 population scores (in progress)
│   ├── figures/
│   │   ├── correlation_matrix.png
│   │   └── scatter_ai_vs_productivity.png
│   ├── oxford_insights/                    ← not committed — download separately
│   └── stanford_hai/                       ← HAI Index CSVs, 2023–2024
└── scripts/
    ├── scrape_github_panel.py              ← Phase 1 panel scraper
    ├── build_panel.py                      ← merge pipeline
    ├── build_panel_v2.py                   ← Phase 2 panel + classifier IV
    ├── run_analysis.py                     ← regression + figures
    ├── scrape_classifier_sample.py         ← Phase 2 subsample scraper
    ├── scrape_classifier_full.py           ← Phase 2 full scraper (200+200 accounts)
    ├── scrape_expanded_positives.py        ← Expansion positive discovery
    ├── scrape_population_v3.py             ← Population scorer (38 countries, 3k target)
    ├── classifier_placebo_test.py          ← Pre-period classifier validation
    ├── retrain_classifier_expanded.py      ← Merge expansion + retrain classifier
    └── train_classifier.py                 ← RF classifier training
```

---

## Known Issues

- **Classifier confound (CRITICAL):** Placebo test shows 6/8 pre-existing metrics differ significantly between high/low pre-only score quartiles. Confirmed positives score 0.418 pre-only vs 0.033 for negatives (p<0.0001). The classifier captures developer conscientiousness, not purely AI adoption. **Mitigation:** `baseline_log_commits` control added to panel regression. **Limitation:** residual confounding may remain at account level within countries.
- **US undercounting:** state abbreviations (CA, NY, TX) in GitHub location fields aren't reliably mapped to country. US developer count is likely understated.
- **Location hit rate ~26%:** the visible subsample may not be representative. Sensitivity analysis needed.
- **Both-window coverage thin for positives:** Claude Code accounts skew recent (post-2023), limiting pre-adoption baseline data.
- **Commit velocity confounded by sampling:** negative accounts selected from active GH Archive windows skew toward prolific committers. Velocity features unreliable without controlling for baseline.
- **Rate-limit resilience:** v1/v2 scrapers could enter multi-hour sleep spirals on GitHub 403s. Fixed in v3 with capped backoff and proactive quota checks.
- **Panel thinness:** median 2 developers per country-year even after minimum-N filtering. Weighted regression mitigates but doesn't eliminate this.

---

## Licence

MIT
