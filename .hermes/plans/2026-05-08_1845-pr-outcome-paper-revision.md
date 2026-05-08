# PR Outcome Paper Revision Implementation Plan

> **For Hermes:** Use subagent-driven-development skill if splitting this into independent coding/review work. For this pass, implement directly with strict TDD for script changes.

**Goal:** Convert the completed PR outcome scrape into paper-ready evidence, check whether it challenges the current paper framing, and prepare the code/data pipeline to rerun on the expanded 276-account cohort.

**Architecture:** Extend `scripts/pr_outcome_metrics.py` rather than adding a parallel notebook-only analysis. The script should own reproducible data generation, core DiD estimates, robustness specifications, multiple-testing correction, and caveat diagnostics. The paper/notebook can then consume stable CSV/TXT outputs instead of one-off scratch calculations.

**Tech Stack:** Python, pandas, statsmodels, pytest, uv.

---

## Current context

- Existing PR scrape completed for `data/classifier_full_features.csv`: 235 accounts, 33 treated, 202 controls.
- Main account-level PR DiD is strong for PR volume and merged PRs.
- Robustness checks show the PR volume signal survives dropping capped accounts, dropping zero-PR accounts, and restricting to high-confidence treated accounts.
- Merge-rate/time-to-merge effects are more fragile because zero-PR windows are currently encoded as zero-valued rates/latencies.
- Current paper `notebooks/research_paper_v3.ipynb` frames the country-level PR dependent variable as a null result. The new account-level PR evidence does not overturn the paper, but it challenges any comfortable interpretation that PR output is unchanged.
- Expanded classifier file exists: `data/classifier_predictions_expanded.csv` with 276 accounts, 74 positives, 202 controls. The PR scraper currently defaults to the older 235-account file.

---

## Task 1: Add configurable input/output paths

**Objective:** Let the PR outcome script run against either the old 235-account feature file or the expanded 276-account classifier file without editing constants.

**Files:**
- Modify: `scripts/pr_outcome_metrics.py`
- Test: `tests/test_pr_outcome_metrics.py`

**Steps:**
1. Write failing tests for loading a caller-specified features CSV and building outcomes from a caller-specified cache directory if needed.
2. Add CLI options:
   - `--features-path`, default `data/classifier_full_features.csv`
   - `--cache-dir`, default `data/pr_outcome_cache`
   - `--outcomes-path`, default `data/account_pr_outcomes.csv`
   - `--did-results-path`, default `data/account_pr_did_results.txt`
   - `--status-path`, default `data/pr_outcome_status.csv`
3. Thread these paths through `load_feature_accounts`, `scrape_accounts`, `build_outcome_dataset`, and `write_did_report`.
4. Run targeted tests.

**Verification:**
```bash
uv run --with pytest --with pandas --with statsmodels pytest tests/test_pr_outcome_metrics.py -q
```

---

## Task 2: Add reproducible robustness diagnostics

**Objective:** Promote the scratch double-checks into first-class report output.

**Files:**
- Modify: `scripts/pr_outcome_metrics.py`
- Test: `tests/test_pr_outcome_metrics.py`

**Required diagnostics:**
- Main DiD estimates for all `DID_METRICS`.
- Benjamini-Hochberg q-values across the main metrics.
- Sensitivity specs for key metrics:
  - `main`
  - `uncapped_only`: drop accounts with `n_prs >= max_prs_per_account`
  - `nonzero_prs_only`: keep accounts with `n_prs > 0`
  - `both_prepost_activity`: keep `pre_prs_opened > 0` and `post_prs_opened > 0`
  - `drop_zero_pre`: keep `pre_prs_opened > 0`
  - `high_conf_treated_only`: controls plus treated where `marker_confidence == "high"`
- Coverage diagnostics:
  - total accounts, treated/control counts
  - zero-PR accounts by label
  - capped accounts by label
  - PR-active accounts by label
  - both-window PR-active accounts by label
- Clear interpretive warning: PR-volume effects are primary; merge-rate/time-to-merge should be treated as secondary because zero-PR windows contaminate those metrics.

**Verification:**
- Unit test BH q-values on known p-values.
- Unit test sensitivity builder returns the expected spec names/counts on a toy dataset.
- Run targeted pytest.

---

## Task 3: Re-run analysis on existing cache for the expanded cohort

**Objective:** Build an expanded-cohort outcomes file using cached PRs where available and expose missing accounts cleanly.

**Command:**
```bash
uv run --with pandas --with statsmodels scripts/pr_outcome_metrics.py \
  --analyse \
  --features-path data/classifier_predictions_expanded.csv \
  --outcomes-path data/account_pr_outcomes_expanded.csv \
  --did-results-path data/account_pr_did_results_expanded.txt
```

**Expected:**
- Output will initially include only accounts with cache files.
- Controls should all be present from the completed 235-account scrape.
- The 41 additional high-confidence positives will likely be missing until scraped.

---

## Task 4: Scrape missing expanded positives only

**Objective:** Complete PR outcome coverage for the 276-account classifier cohort without redoing cached accounts.

**Command:**
```bash
GITHUB_TOKEN=... uv run --with pandas --with statsmodels scripts/pr_outcome_metrics.py \
  --scrape --analyse \
  --features-path data/classifier_predictions_expanded.csv \
  --outcomes-path data/account_pr_outcomes_expanded.csv \
  --did-results-path data/account_pr_did_results_expanded.txt \
  --status-path data/pr_outcome_status_expanded.csv
```

**Notes:**
- Existing cache files should be reused.
- Only missing expanded positives should require GitHub calls.
- Use the durable shell runner pattern if this looks long.

---

## Task 5: Update project documentation

**Objective:** Keep `project_plan.md` aligned with the new evidence.

**Files:**
- Modify: `project_plan.md`

**Content to add after verified expanded run:**
- The 235-account PR outcome result is strong but preliminary relative to the paper’s current 276-account classifier frame.
- Main PR-volume effects survive robustness; merge-rate/time-to-merge are secondary.
- Account-level PR evidence does not overturn the country-level PR null but forces a more careful interpretation: aggregation can hide account-level output shifts.
- Immediate next paper revision: update abstract/results/discussion to distinguish account-level PR output from country-level PR panel null.

---

## Paper framing decision

Use this language unless the expanded run contradicts it:

> The account-level PR outcome extension shows that confirmed AI coding-tool adopters open and merge substantially more PRs after adoption, with no clear increase in PR size or review burden. This does not invalidate the country-level PR null; instead, it reveals a scale mismatch. Country-level aggregate PR rates are too noisy to carry strong output claims, while account-level PR histories show visible adopters increasing accepted packaged work. The paper should therefore frame PR outcomes as evidence against a simple “AI reduces productivity” story, while preserving the caution that public GitHub traces measure workflow and accepted artifacts, not total productivity.

---

## Risks

- Expanded positives may be much more PR-heavy and slow to scrape.
- GitHub search caps can truncate prolific accounts at `max_prs_per_account=300`; uncapped-only robustness remains mandatory.
- Zero-PR windows make rates/latencies mathematically convenient but substantively ugly.
- The expanded 276-account file includes predicted/classifier metadata, while the old 235 file is the original labelled features file. Keep labels and discovery fields explicit.

---

## Done criteria

- Tests pass.
- Script can run against the expanded classifier file via CLI flags.
- Report includes main estimates, BH q-values, coverage, and robustness specs.
- Expanded-cohort analysis has either completed or a durable scrape is running.
- `project_plan.md` records the current state accurately.
