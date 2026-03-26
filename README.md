# AI Adoption & Developer Productivity

Does AI coding tool adoption causally increase developer productivity? This project tries to find out — using GitHub as the lab.

Two-phase study. Phase 1 built a country-level panel regression linking AI readiness to GitHub activity metrics. Phase 2 (current) is building an account-level binary classifier to identify AI coding tool users from observable behavioural signals — commit patterns, message length, PR quality — without relying on explicit markers.

---

## Status

**Phase 1 — complete (null result).** Fixed-effects panel regression across 51 countries, 2022–2024. AI readiness coefficient p = 0.497 — not significant. Root cause: Oxford Insights measures government AI policy readiness, not whether developers are actually using Claude Code or Copilot today. Wrong independent variable.

**Phase 2 — in progress.** Building a classifier to label individual GitHub accounts as AI coding tool users or not. Ground truth from explicit artefacts (CLAUDE.md files, co-author commit trailers). Behavioural features: commit message length delta, PR description completeness, test co-write rate, conventional commit adoption. Early signal is promising — message length delta ~4x larger for confirmed AI users vs. controls.

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

## Preliminary Results (Phase 2 Test Run — March 2026)

Test scrape: 40 accounts (20 confirmed Claude Code users, 20 controls).

| Feature | AI users (Δ) | Controls (Δ) | Ratio |
|---------|-------------|--------------|-------|
| Mean message length | +53.8 chars | -3.4 chars | ~16x |
| Frac conventional commits | +0.26 | +0.02 | ~15x |
| Frac PR has body | +0.69 | +0.05 | ~15x |
| Test co-write rate | +0.19 | -0.01 | ~15x |

Coverage caveat: only 4 of 20 positive accounts had sufficient pre-adoption commit history for the both-window comparison (≥10 commits pre and post Nov 2023). CLAUDE.md accounts skew recent. Full scrape (200 pos / 200 neg) running to address this.

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

uv run scripts/scrape_classifier_full.py  # Phase 2 classifier scrape (set TEST_RUN flag)
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
├── data_source_assessment.md               ← AI adoption data source evaluation
├── analysis_results_march2026.md           ← full Phase 1 write-up
├── data/
│   ├── panel_dataset.csv                   ← 88 obs × 9 cols, Phase 1 panel
│   ├── github_panel_flat.csv               ← 347 rows, country × quarter
│   ├── regression_results.txt              ← OLS summary
│   ├── classifier_test_features.csv        ← Phase 2 test run features (40 accounts)
│   ├── figures/
│   │   ├── correlation_matrix.png
│   │   └── scatter_ai_vs_productivity.png
│   ├── oxford_insights/                    ← not committed — download separately
│   └── stanford_hai/                       ← HAI Index CSVs, 2023–2024
└── scripts/
    ├── scrape_github_panel.py              ← Phase 1 panel scraper
    ├── build_panel.py                      ← merge pipeline
    ├── run_analysis.py                     ← regression + figures
    ├── scrape_classifier_sample.py         ← Phase 2 subsample scraper
    └── scrape_classifier_full.py           ← Phase 2 full scraper (200+200 accounts)
```

---

## Known Issues

- **US undercounting:** state abbreviations (CA, NY, TX) in GitHub location fields aren't reliably mapped to country. US developer count is likely understated.
- **Location hit rate ~26%:** the visible subsample may not be representative. Sensitivity analysis needed.
- **Both-window coverage thin for positives:** Claude Code accounts skew recent (post-2023), limiting pre-adoption baseline data.
- **Commit velocity confounded by sampling:** negative accounts selected from active GH Archive windows skew toward prolific committers. Velocity features unreliable without controlling for baseline.

---

## Licence

MIT
