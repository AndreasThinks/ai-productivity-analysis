# AI Adoption & Developer Productivity

Do developers who adopt agentic AI coding tools ship more accepted work? This project
uses public GitHub data to find out. The paper lives in
`notebooks/research_paper_v3.ipynb` and is published at
[andreasthinks.me/papers/ai-productivity](https://andreasthinks.me/papers/ai-productivity/).

---

## Status (29 October 2026)

**Pull-request outcomes: stand, with caveats.** Confirmed Claude Code and Aider adopters
open and merge more pull requests than randomly sampled controls (+46 opened, +43 merged,
p < 0.001; +55 and +51 without the 12 adopters whose markers predate Claude Code). This
analysis splits every account at 1 January 2024. Adopters were found through recent
activity and merged PRs include self-merges, so the estimates are likely upper bounds.

**Commit tempo, classifier and country panel: withdrawn pending v4.** The v2.7 scraper
split 25 of the 33 adopters at a commit from March 2026, days before the scrape, giving
them a 2–3 week post-period against about two years for controls. That produced the
reported tempo effects and most of the classifier's 0.94 AUC (post-period active weeks
alone gives 0.89). See `methodology_decisions.md`, Decision 7, and the correction at the
top of the paper.

**v4 re-collection: code ready, not yet run.** `scripts/rescrape_v4.py` uses equal
26-week windows around each account's first agent-attributed commit, placebo dates for
controls, all public repositories, and stored commit authorship so agent-written commits
can be separated from human ones. `scripts/analyse_v4.py` produces the DiD and classifier
results from its caches.

---

## Reproducing

Dependencies are pinned with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest                                   # unit tests
uv run jupyter nbconvert --to notebook --execute notebooks/research_paper_v3.ipynb
```

The notebook reads `data/` next to it, or wherever `AI_PRODUCTIVITY_DATA` points.

### Running the v4 re-collection

It needs a GitHub personal access token and the real account list, which is not in the
repo (see Data and privacy below). Search is limited to 30 requests a minute, so expect
roughly 40 seconds per account, about 3–4 hours for all 311.

```bash
export GITHUB_TOKEN=...                          # classic PAT, public_repo scope is enough
mkdir -p private && cp /path/to/accounts_v4.csv private/
uv run scripts/rescrape_v4.py --limit 3         # smoke test: 3 adopters, 3 controls
uv run scripts/rescrape_v4.py                    # full run; safe to restart
uv run scripts/analyse_v4.py                     # writes data/v4_*.csv and data/v4_results.md
```

`private/accounts_v4.csv` has columns `login`, `group` (`adopter` or `control`) and
`tool` (`claude`, `aider` or empty). `private/pseudonym_salt.txt` must be the same salt
used for the committed data so that `account_id` values line up.

---

## Data and privacy

Committed data are pseudonymised: GitHub logins are replaced by `account_id`, a salted
SHA-256 hash (`scripts/pseudonymise.py`). Raw caches with commit messages and profile
locations, the login list, the salt and the login key live in `private/`, which is
gitignored. Earlier commits in this repository's history still contain raw data until the
history is rewritten.

Oxford Insights data is not redistributable; download it from
[oxfordinsights.com](https://oxfordinsights.com/ai-readiness/ai-readiness-index/) into
`data/oxford_insights/`. Stanford HAI CSVs are included under `data/stanford_hai/`.

The legacy scrapers (`scrape_*.py`, `validate_aider.py`, `pr_outcome_metrics.py --scrape`)
expect real logins and so only run against the private data.

---

## File structure

```
├── notebooks/research_paper_v3.ipynb     the paper; single source for the website copy
├── methodology_decisions.md               log of analytical choices (Decision 7: v4 design)
├── pyproject.toml, uv.lock                pinned dependencies
├── scripts/
│   ├── paths.py                           data and private directories (env-overridable)
│   ├── rescrape_v4.py                     v4 event-time collection
│   ├── analyse_v4.py                      v4 features, DiD and classifier
│   ├── pseudonymise.py                    login -> account_id
│   ├── pr_outcome_metrics.py              PR outcome DiD (v3)
│   ├── train_classifier.py, retrain_classifier_expanded.py, validate_aider.py
│   ├── scrape_*.py, build_panel*.py, run_analysis.py, robustness_checks.py
├── data/                                  pseudonymised derived data used by the paper
├── private/                               not committed: logins, salt, raw caches
└── tests/
```

Older working documents (`project_plan.md`, `analysis_results_march2026.md`,
`data_source_assessment.md`, `country_trim_analysis.md`, `octopus/`) describe earlier
phases and predate the October 2026 correction.

---

## Licence

MIT
