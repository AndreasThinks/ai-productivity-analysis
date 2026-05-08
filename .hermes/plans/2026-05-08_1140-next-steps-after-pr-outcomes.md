# Next steps after PR outcome metrics

## Goal

Turn the new account-level PR outcome scraper into evidence the paper can use, then revise the paper around the strongest empirical story.

## Current context

- PR outcome tooling is merged on `main`.
- Smoke scrape succeeded against real GitHub API data.
- Generated PR outputs are intentionally ignored by git.
- Country-level panel result remains null-to-negative and structurally thin.
- The strongest likely path is account-level before/after evidence with PR outcomes as accepted-output metrics.

## Proposed sequence

### 1. Run the full PR outcome scrape

Command:

```bash
GITHUB_TOKEN=... uv run --with pandas --with statsmodels \
  scripts/pr_outcome_metrics.py --scrape --analyse
```

Expected outputs:

- `data/pr_outcome_cache/<login>.json`
- `data/account_pr_outcomes.csv`
- `data/account_pr_did_results.txt`

Validation:

- Confirm treated/control counts are both nonzero.
- Check how many accounts have at least one pre or post PR.
- Inspect missingness: no PRs, post-only PRs, pre-only PRs.
- Re-run tests after scrape:

```bash
uv run --with pytest --with pandas --with statsmodels pytest tests/test_pr_outcome_metrics.py -q
```

### 2. Diagnose whether PR coverage is usable

Key checks:

- Share of classifier cohort with any authored PRs.
- Share with PRs in both pre and post windows.
- Treated/control balance among PR-active accounts.
- Whether PR-active accounts are a weird subset of the classifier cohort.

Decision rule:

- If PR-active N is healthy, make PR outcomes a main account-level result.
- If PR-active N is thin, frame it as supporting evidence and keep commit-tempo DiD as the main account-level result.

### 3. Add PR outcome figures/tables to paper notebook

Likely additions to `notebooks/research_paper_v3.ipynb` or successor paper notebook:

- PR outcome DiD coefficient table.
- Coefficient plot for merged PRs/month, merge rate, time-to-merge, PR size, review comments.
- Small missingness table: all classifier accounts vs PR-active accounts.
- Interpretation paragraph: commits measure workflow tempo, PRs measure accepted packaged work.

### 4. Reframe the paper's empirical story

Recommended framing:

- Country-level panel: careful null or ambiguous negative, likely measurement and aggregation limits.
- Account-level classifier: methodological contribution, validated cross-tool.
- Account-level DiD: AI users show changed development tempo.
- PR outcomes: accepted-output check. This determines whether the tempo shift corresponds to shipped work or just workflow rearrangement.

### 5. Clean up the no-op baseline control

`baseline_log_commits` is absorbed by country fixed effects in the panel. Either:

- remove it from the headline panel methods, or
- explicitly label it as a failed mitigation and move it to robustness discussion.

Do not let it imply we controlled for the placebo confound when the model mathematically cannot use it.

### 6. Final paper revision pass

After PR outcomes are known:

- Update abstract and conclusion.
- Replace stale v3 numbers where needed.
- Ensure every quantitative claim comes from current generated outputs.
- Remove or quarantine speculative productivity claims if PR outcomes do not support them.

## Files likely to change

- `notebooks/research_paper_v3.ipynb` or a new final paper notebook.
- `project_plan.md` after full PR scrape results are known.
- Possibly `README.md` if PR outcome results become canonical.
- Possibly `scripts/pr_outcome_metrics.py` if full scrape reveals API or missingness edge cases.

## Risks

- GitHub search API may cap at 1,000 results per query for prolific accounts. Current max is 300 PRs/account, enough for this cohort unless power users dominate.
- PR-active subset may be small or unbalanced.
- Time-to-merge and review comments are collaboration metrics, not pure productivity.
- Closed-unmerged PRs may include abandoned drafts, experiments, and stale branches.

## Immediate next action

Run the full PR scrape, inspect coverage, then decide whether PR outcomes enter the paper as headline evidence or robustness evidence.
