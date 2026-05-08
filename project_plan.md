# AI Productivity Analysis — Project Plan

*Living document. Updated as the project evolves. See AGENTS.md for coding conventions and agent instructions.*

---

## Project Status: Phase 2 — PR Outcome Extension Drafted, Paper Revision Pending (May 8, 2026)

**Where we are:**
- Classifier is done and validated (AUC 0.940, Aider generalisation confirmed).
- Population scrape v3 **COMPLETED April 25 morning**: 2,999 accounts scored across
  53 countries, 102 country-quarter groups with >=15 accounts.
- Adoption table built: `data/country_quarter_ai_adoption_v3.csv`.
- `build_panel_v2.py` ran clean. Headline numbers (April 25):
  - Regression A (Phase 1, Oxford IV): coef=0.067, p=0.46, N=88, null
  - Regression C (per-country IV + baseline): coef=-5.14, p=0.25, N=72, null, NEGATIVE
  - Regression C-W (weighted): coef=-7.56, p=0.055, borderline NEGATIVE
  - Regression D (parallel trends): p=0.62, passes
- Country trim analysis completed: dropping 16 low-value countries (54 → 38 panel).
  See `country_trim_analysis.md` for full rationale.
- v3 incorporates: 5-minute sleep cap, proactive /rate_limit checks, abuse detection
  vs quota exhaustion distinction, global cooldown after 3 consecutive 403s, jitter.
- Pre-period placebo test completed (`scripts/classifier_placebo_test.py`):
  **RESULT: 6/8 pre-existing metrics significantly differ (p<0.05). Classifier captures
  pre-existing developer conscientiousness, not purely AI adoption.** Mitigation:
  `baseline_log_commits` control added to panel regression, but robustness checks
  (April 25) confirmed it is fully absorbed by entity FE and contributes nothing.
- Classifier retrained with 41 expansion positives: 74 pos + 202 neg = 276 accounts.
  AUC 0.936 (±0.031), stable vs original 0.940. Saved as `classifier_model_expanded.pkl`.
- **PR outcome extension drafted and first full run completed May 8:** `scripts/pr_outcome_metrics.py` scrapes authored
  PRs across repositories via GitHub issue search, caches per-account PR details in
  `data/pr_outcome_cache/`, builds `data/account_pr_outcomes.csv`, and writes
  `data/account_pr_did_results.txt`. Hardened with atomic cache writes, per-account
  error handling, and `data/pr_outcome_status.csv`. Durable launch/check scripts:
  `run_pr_outcome_scrape.sh` and `check_pr_outcome_scrape.sh`. Tests live in
  `tests/test_pr_outcome_metrics.py`. Initial 235-account run found strong account-level
  PR-volume effects: opened PRs +57.6 (p=0.0011), merged PRs +55.6 (p=0.0011),
  merge rate +0.26 (p=0.00036), median time-to-merge −19.2h (p=0.027). After
  double-checking, the PR-volume effects survive FDR correction and robustness filters;
  merge-rate/time-to-merge are secondary because zero-PR windows distort rates/latencies.
  Script now supports custom `--features-path`, `--outcomes-path`, and report paths so the
  expanded 276-account classifier cohort can be rerun. Expanded positive scrape is in progress
  for the 41 high-confidence positives missing from the first cache.

**Critical scrape blocker: RESOLVED.** v3 hit 53 countries, well above the >=25
threshold. Regression C now runs cleanly with N=72 and 34 countries.

**New blocker: paper revision.** The octopus/ draft is stale (quotes old N=59 / 20-
country / coef=-6.06 numbers from pre-v3). Needs full rewrite to incorporate v3
results and the April 25 robustness findings (see below).

---

Phase 1 (panel regression with Oxford Insights AI Readiness Index) returned a null
result. The likely cause is that the independent variable measures government AI
policy readiness, not whether individual developers are actually using AI tools.
Phase 2 replaces it with an account-level classifier.

---

## Phase 1 Recap

**What we built:** Fixed-effects panel regression linking country-level AI readiness to GitHub developer productivity (commits, PRs, repo creation) across 2022–2024.

**What we found:**
- 88 country-year observations, 51 countries
- OLS with country FE: ai_readiness_score coefficient = 0.115, p = 0.497 — not significant
- R² = 0.33, Adj. R² = −0.63 — country FEs consume degrees of freedom at this sample size
- Real signal in 2024 vs 2022 productivity uplift, but that's a time trend not a cross-country effect
- Median n_developers per country-year = 2 — too thin to trust

**Root cause of null result:** Oxford Insights measures government AI readiness (policy, infrastructure, skills frameworks). That's three steps removed from whether a developer opened Claude Code this morning. Independent variable is wrong.

---

## Phase 2: Account-Level AI Usage Classifier

### Objective

Build a binary classifier that labels individual GitHub accounts as AI coding tool users or not, based on observable signals in their public commit history, repo structure, and behavioural patterns. Use the fraction of AI users per country-quarter as the independent variable in the panel model, replacing the Oxford Insights index.

### Classifier Strategy

Two-phase approach:

**Phase 2a — Rule-based marker detection (ground truth labelling)**
Identify confirmed AI tool users from explicit, hard artefacts left in repos and commit history. No training data needed. Output: a labeled set of high-confidence positives and negatives.

**Phase 2b — Behavioural feature classifier**
Train a model on the labeled set using behavioural features that generalise to accounts with no explicit markers. Output: a probability score per account.

---

## AI Tool Markers Research

Research conducted March 2025. Summary of detectable signals per tool.

### Claude Code (Anthropic)

**Confidence: High — multiple reliable markers**

Explicit artefacts:
- `CLAUDE.md` file in repo root
- `.claude/` config directory
- `AGENTS.md` with Claude Code / Hermes content
- `.hermes/` directory (Hermes agent users)

Commit / PR text:
- Co-author trailer: `Co-Authored-By: Claude <noreply@anthropic.com>`
- Commit messages referencing "Claude", "claude-code", "Anthropic"
- PR descriptions with Claude Code footer text

Workflow artefacts:
- GitHub Actions workflows referencing `claude-code` or Anthropic APIs
- `requirements.txt` / `package.json` referencing Anthropic SDK

**Detection approach:** File tree scan + commit message regex. High precision.

---

### Aider (Aider-AI)

**Confidence: High — consistent commit trailer**

Commit trailers:
- `Co-authored-by: aider (model-name) <noreply@aider.chat>`
- `Co-authored-by: aider (model-name) <aider@aider.chat>`
- Author name suffix: `(aider)` appended to commit author field

**Detection approach:** Commit trailer regex. Very reliable — Aider appends this by default and the pattern is well-documented.

---

### GitHub Copilot (Microsoft)

**Confidence: Medium — agent mode only, inline autocomplete invisible**

Agent mode (detectable):
- Commits authored by Copilot directly, human as co-author
- `Agent-Logs-Url` trailer in commit message (introduced March 2026)
- Actor: `copilot-swe-agent[bot]`
- Co-author: `Co-authored-by: Copilot <noreply@github.com>`

Inline autocomplete (NOT detectable):
- Leaves zero commit-level traces
- Majority of Copilot usage is inline autocomplete — this is a significant coverage gap

**Detection approach:** Commit trailer + actor name regex for agent mode only. Will systematically undercount Copilot users. Treat classifier output as "Copilot agent mode user" not "Copilot user."

---

### Cursor (Anysphere)

**Confidence: Low-Medium — config file marker, no commit trailer by default**

File artefacts:
- `.cursor/` directory in repo root (contains rules files)
- `.cursorrules` file in repo root
- `.cursor/rules/*.mdc` files

Commit trailers:
- No standard trailer by default
- Agent Trace spec (released January 2026, RFC status) — if adopted, would create `.agent-trace/` records with model attribution including `dev.cursor` metadata. Too early to rely on.

**Detection approach:** File tree scan for `.cursor/` or `.cursorrules`. Lower precision than Claude Code — these files sometimes exist without active Cursor use.

---

### Windsurf / Codeium

**Confidence: Low — no reliable markers found**

- No standard commit trailer
- No consistent config file left in repos
- Essentially invisible at the commit level with current information

**Detection approach:** Behavioural features only. Cannot reliably detect from explicit markers.

---

### Kiro (AWS)

**Confidence: Medium — author identity detectable**

- Commits authored as "Kiro Agent" identity — author name is the marker
- `.kiro/` config directories in repos
- No GPG signing yet (open feature request as of Feb 2026)

**Detection approach:** Commit author name match + file tree scan for `.kiro/`. Relatively low prevalence in the wild currently.

---

### Devin (Cognition)

**Confidence: Medium-High — agent identity detectable**

- Makes commits through GitHub integration under its own identity
- PRs created via Devin have characteristic structure
- Only major AI agent currently supporting GPG signed commits

**Detection approach:** Commit author / PR actor name match.

---

### Agent Trace Spec (Cursor / Cognition / others)

Vendor-neutral RFC published January 2026. Specifies `.agent-trace/` JSON records storing AI contribution attribution at line level, including model identifiers in `provider/model-name` format. Partners include Vercel, Cloudflare, Cognition, Cline, Amp. If broadly adopted this becomes a universal detection mechanism. Monitor for adoption — not yet reliable enough to use as primary signal.

---

## Classifier Methodology

### The Core Design Constraint

The explicit markers — CLAUDE.md files, co-author trailers, .claude/ directories — are how we *identify* ground truth users. They cannot also be classifier features. Training the classifier on the same signals used to create the labels is data leakage and produces a model that just rediscovers its own labels. It would also be useless for the actual goal: identifying AI users who leave no explicit traces.

The classifier must learn behavioural patterns that are *correlated with* Claude Code adoption without being *definitionally equivalent* to it. The research question underneath this is: even if Claude Code left no explicit traces, could we still identify its users from how their coding behaviour changed?

### Methodological Frame: Account-Level Difference-in-Differences

The design is a difference-in-differences at the account level.

- **Treatment**: confirmed Claude Code adoption, established from explicit markers
- **Pre-window**: 12 months before adoption (or pre-2023 globally as first pass)
- **Post-window**: 12 months after adoption (or 2024+ globally)
- **Outcome**: change in behavioural features between windows

The classifier learns what that change pattern looks like. It can then identify accounts that show the same pattern *without* the explicit markers — that's the generalisation step.

This is also a publishable methodological contribution in its own right: "Can we detect AI tool adoption from behavioural signals alone?" has real implications for measuring AI adoption rates at scale.

### Temporal Window Strategy

**Option A — Account-specific timing (preferred, run second)**
For each account: find date of first Claude Code marker. Pre-window = 12 months before. Post-window = 12 months after. Most principled — you're measuring change from each account's actual adoption date. More complex to implement. Thin post-window for recent adopters.

**Option B — Global cutoff (simpler, run first)**
Pre = Jan 2022 – Jun 2023. Post = Jan 2024 – present. Clean gap covering the ambiguous period where some developers had early API access but the CLI wasn't public.

Build with Option B first. Store each account's first-marker date so Option A can be run as a robustness check. If both produce similar classifiers, the temporal signal is robust.

### Behavioural Features (safe to use — no leakage risk)

In rough order of expected signal strength:

**Temporal change features (strongest)**
- Δ commit frequency per active week (pre vs post)
- Δ mean lines added per commit
- Δ commit size distribution (shift right = more medium/large commits)
- Δ PR description length
- Δ test file co-creation rate (commits touching impl + tests together)
- Δ documentation commit rate

**Level features (weaker alone, useful in combination)**
- Commit velocity absolute level
- PR description completeness score (has summary, mentions testing — regex heuristic)
- Cross-language activity breadth
- Repo creation rate
- Hour-of-day distribution entropy (AI lowers friction → flatter activity distribution)
- Burst pattern — commits clustered in short sessions

**Features to explicitly exclude (label leakage)**
- CLAUDE.md presence
- Co-author trailer content
- Any keyword match on "Claude", "Anthropic", "claude-code" in commit messages
- .claude/ or .hermes/ directory presence
- Any other explicit marker used to construct the label

### Scrape Design

**Stage 1 — Ground truth positive discovery**

Two parallel routes:
1. GitHub Code Search API: `filename:CLAUDE.md` → returns repos → resolve to account logins
2. GH Archive PushEvent data: scan commit messages for `Co-Authored-By: Claude <noreply@anthropic.com>` → extract actor logins

Deduplicate across both routes. Target 200–500 confirmed positive accounts.

**Stage 2 — Per-account deep scrape (GitHub REST API)**

For each account collect:
- User profile: created_at, location, public_repos count
- Up to 100 most recent repos: name, created_at, language, size
- For each repo: top-level file tree (CLAUDE.md detection; do NOT use as feature — label only)
- Up to 500 most recent commits per account: message, timestamp, additions, deletions
- Up to 100 most recent PRs: body length, created_at, merged_at, state

Cache everything in `data/classifier_cache/` keyed by login + date. Resume-safe.

**Stage 3 — Negative set construction**

From a random GH Archive sample, filter to accounts with:
- Activity before Nov 2023 (establishes pre-period baseline)
- Zero explicit AI tool markers across all tools
- Minimum 20 commits total (enough history to compute features)

**Rate limit:** 5000 req/hr with PAT. ~5 calls per account = ~1000 accounts/hr max.

---

## Classifier Build Plan

### Step 1 — Collect confirmed Claude Code accounts (ground truth positives)

Search GH Archive for accounts with:
- Any repo containing `CLAUDE.md`
- Any commit with `Co-Authored-By: Claude <noreply@anthropic.com>`
- Any repo with `.claude/` or `.hermes/` directories

Target: 200–500 confirmed positive accounts. AndreasThinks is a known positive for sanity-checking.

Collect pre/post history for each account (split at November 2023, Claude Code launch). Each account is its own control — before state is the counterfactual, after state is the treated state.

### Step 2 — Collect confirmed negative accounts

Accounts that:
- Have commit history before Nov 2023 but zero activity after (provably pre-AI era, no adoption)
- Have consistent low-quality commit messages throughout with no markers
- Explicitly have Copilot-only markers (separate class — not negative, but different tool)

Target: 500–1000 confirmed negative accounts.

### Step 3 — Extract behavioural features

For each account compute:

*Commit-level:*
- Mean / variance of lines added per commit
- Mean / variance of lines deleted per commit
- Churn ratio (lines deleted / lines added)
- Commit message length mean and variance
- Commit frequency per active week
- Commit message quality proxy (verb presence, sentence structure heuristic)

*PR-level:*
- PR description length mean
- PR description completeness (has summary, has testing notes — regex)
- PR merge rate
- Time from open to merge mean

*Timing:*
- Hour-of-day distribution entropy (AI users may show flatter distributions)
- Weekend vs weekday activity ratio
- Burst pattern — commits clustered in short windows

*Repository:*
- README length
- Repos with complete structure (tests/, docs/, CI)
- Language diversity

*Temporal change features (strongest signal):*
- Δ commit frequency pre/post Nov 2023
- Δ commit message length pre/post
- Δ PR description completeness pre/post
- Δ lines-per-commit pre/post

### Step 4 — Train classifier

- Train on high-confidence labeled set from Step 1+2
- Hold out 20% as test set before training
- Model: logistic regression first (interpretability), then gradient boosted trees
- Report precision, recall, F1 on held-out set
- Sanity check: does AndreasThinks score as positive?
- Spot-check 20 classifier-assigned positives manually

### Step 5 — Apply to random sample + validation against other tools

Apply the trained classifier (built on Claude Code ground truth) to a random sample of 5,000–10,000 GitHub accounts with sufficient activity history.

Read the confidence score distribution:
- Accounts scoring above threshold = predicted AI tool users
- Distribution shape tells us estimated population prevalence

Validation against other tools:
- Collect accounts with confirmed Aider markers (`noreply@aider.chat` trailers)
- Collect accounts with confirmed Copilot agent markers
- Collect accounts with Cursor `.cursor/` markers
- Run classifier over these sets — do they score higher than the random sample baseline?
- If yes: classifier is picking up general AI-assisted coding behaviour, not just Claude Code-specific patterns. That's actually useful — it means the behavioural features generalise.
- If no: classifier is overfitting to Claude Code-specific stylistic patterns. Need to revisit feature engineering.

This validation step answers whether we're measuring "Claude Code use" or "AI-assisted coding in general" — both are interesting, but they're different claims.

### Step 6 — Integration into panel model

Replace `ai_readiness_score` with `pct_ai_users_per_country_quarter` derived from classifier predictions. Rerun PanelOLS with country + time FE and clustered SEs. Compare results to Phase 1.

---

## Open Questions

1. **Sparsity of explicit markers**: How many CLAUDE.md files actually exist in public repos on GH Archive? Run a quick search before committing to this as the primary ground truth source.

2. **Tool specificity vs generality**: Do we want the classifier to identify Claude Code users specifically, or AI-assisted coders broadly? The former is more precise but thinner on labeled data. The latter is more robust but harder to validate.

3. **Account vs commit level**: Should the classifier score be binary (AI user / not) or continuous (fraction of commits that appear AI-assisted)? Continuous is richer but harder to validate.

4. **Selection bias**: Accounts that leave explicit markers (CLAUDE.md etc.) may be more careful or more experienced developers. The behavioral patterns of marker-leavers may not generalise to casual AI tool users.

5. **Temporal drift**: AI tool usage patterns will change as tools evolve. A classifier trained on 2024 data may not generalise to 2026 patterns.

---

---

## Subsample Scrape Results (March 2026)

### What was run

Script: `scripts/scrape_classifier_sample.py` (v2)
Ground truth positives: GitHub Code Search API (`filename:CLAUDE.md`) → 50 confirmed Claude Code user accounts
Negatives: GH Archive 2025-01-15 hour 3 → 50 active developer candidates (≥5 push events, no AI markers)
Deep scrape cap: 30 positives + 30 negatives
Commit history: via `/repos/{owner}/{repo}/commits` (full history, not 90-day events window)
Feature split: PRE = 2022-01-01 to 2024-01-01 / POST = 2024-01-01 onwards

### Output files

| File | Contents |
|------|----------|
| `data/classifier_positive_logins.csv` | 50 confirmed Claude Code accounts |
| `data/classifier_negative_logins.csv` | 50 negative candidates |
| `data/classifier_sample_raw.json` | Raw scraped data, 60 accounts |
| `data/classifier_sample_features.csv` | 59 rows × 17 feature columns |
| `data/gharchive_2025-01-15-3.jsonl` | GH Archive cache (192k events) |

### Key findings

**Coverage issue:** Only 27% of positives and 17% of negatives had commits in *both* time windows. Most negative accounts are newer (post-2024 only) — a GitHub account age confound rather than a true negative signal. Both-window filter left 8 positives and 5 negatives, too thin for reliable analysis. Confirmed: must filter to accounts with pre_commit_count > 0 in the full scrape.

**Commit message length — strongest signal observed:**

On both-window accounts (most valid comparison):
- Positives: pre mean = 58.8 chars, post mean = 106.3 chars → **Δ = +47.4 chars**
- Negatives: pre mean = 31.4 chars, post mean = 43.2 chars → **Δ = +11.8 chars**
- Ratio: positive delta is ~4× larger than negative delta

On full sample (confounded by account age but larger n):
- Positives: post mean = 104.2 chars vs negatives: 40.1 chars
- Delta: positives +88.5 chars on average, negatives +32.2 chars

This is the clearest signal in the data. Claude Code generates verbose, structured commit messages — this is a real and detectable behavioural shift.

**Commit velocity — confounded, not reliable:**
- Negatives show higher raw commit counts and larger deltas than positives in both windows
- Likely because the negative sample skews toward very active developers (selected by ≥5 push events in one hour)
- Not a useful discriminating feature without controlling for baseline activity level

**Active weeks — also confounded:**
- Many negatives are new accounts (all activity post-2024), inflating their post-window active_weeks
- Both-window negatives actually show *larger* Δactive_weeks (+22.6) than positives (+3.2)
- Directionally wrong — suggests account age is dominating this feature

**Repos touched — no signal:**
- Near-zero difference between groups in both-window analysis (2.6 vs 2.6)
- Drop from feature set

### What this tells us about the classifier

The message length delta is the most promising feature by far and survives the both-window filter. The velocity/frequency features are badly confounded by the negative sampling strategy — selecting by activity level in GH Archive biases toward prolific committers.

**Revised negative sampling strategy for full scrape:** do NOT use activity threshold as the selection criterion. Instead, sample randomly from GH Archive actors and verify they have commit history in both windows and zero AI markers. Accept lower activity levels. This will reduce the activity confound.

**Minimum viability threshold for full scrape:** require both pre_commit_count ≥ 10 and post_commit_count ≥ 10 for any account to be included in model training. Discard zero-pre accounts entirely.

### Technical issues encountered and fixed

1. **OOM on GH Archive download** — original script loaded full decompressed archive into memory (~500MB). Fixed: streaming gzip to disk, then iterating line-by-line via generator. Never loads full archive into memory.
2. **Empty cache from first run** — script was killed mid-run, leaving incomplete cache files. Fixed: clear cache before rerun, cache only written on success.
3. **first_marker_date blank** — Code Search API doesn't return repo `created_at`. Fixed: follow-up `/repos/{owner}/{repo}` call to populate date.
4. **Commits empty from events API** — events endpoint only covers 90 days. Fixed: switched to `/repos/{owner}/{repo}/commits` for full history.
5. **PR data all nulls** — PullRequestEvent payload parsing was wrong. Fixed: switched to `/repos/{owner}/{repo}/pulls` endpoint.
6. **HTTP 409 on empty repos** — some repos have no commits (unborn branch). Handled gracefully, continues to next repo.

---

---

## scrape_classifier_full.py — Status and Design (March 2026)

### Current version: v2.7

`scripts/scrape_classifier_full.py` is the full-scale production scraper. v2.7 incorporates lessons from multiple runs, two code reviews, and a DNS-failure incident.

**v1 → v2.0 improvements (first code review):**
- Negative sampling: random (no activity threshold), dynamic loop until 200 accepted, both-window filter enforced at scrape time with correct PRE_START lower bound
- Rate limiting: 1.0s delay (~3,600 req/hr), rate-limit-aware backoff reads `X-RateLimit-Reset` header, secondary rate limit floor 60s, MAX_RETRIES raised to 5
- Resume safety: incremental status file writes on every decision, tagged output files prevent test/full collisions, positive progress file skips completed accounts on restart
- New features: commit message structure (multiline, conventional prefix, test mentions via `\btest[s]?\b`, bullets), inter-commit burst patterns, test co-write rate via 20% file sample (denominator fixed to sampled commits only), PR body length
- Feature leakage guard: Claude markers written to separate labeling CSV, absent from raw data used for feature extraction
- Commit deduplication by SHA across repos (prevents double-counting forks)
- Multi-hour GH Archive: 6 hours across 3 days (vs 1 hour in v1) for better co-author recall
- Symmetric both-window filter: positives now also required to meet MIN_PRE_COMMITS + MIN_POST_COMMITS

**v2.0 → v2.1 improvements (second code review):**
- **Temporal split fix**: `first_marker_date` for Code Search positives is `repo.created_at`, which can predate CLAUDE.md addition by years. Fixed by adding `marker_confidence` field: GH Archive co-author positives tagged `high` (use actual push timestamp as post-window start); Code Search positives tagged `low` (fall back to global POST_START cutoff). `marker_confidence` propagates to features CSV for downstream stratification.
- **Stage 1c removed**: Contributors API discovery loop was an expensive no-op — it only checked repos already owned by known positives, so it could never surface new accounts. Removed cleanly; deferred to future iteration if a cross-GitHub approach is designed.
- **Repo sort**: changed from `pushed` ascending (surfaced dormant repos with thin histories) to `created` ascending (oldest repos first — more likely to have meaningful pre-period history while still having commit depth to pass the both-window filter).

**v2.5 → v2.6 improvements:**
- GH Archive hours expanded from 6 to 12 across 3 months (Nov 2024, Jan 2025, Mar 2025) to diversify the negative candidate pool and reduce January 2025 selection bias.
- MAX_NEGATIVES_TARGET and MAX_NEGATIVES_CANDIDATES raised to 500 and 2000 respectively.
- Negative candidate shuffle uses a locally-seeded `Random(42)` on sorted input for reproducible queue order regardless of execution path.
- Status file handle wrapped in try/finally for crash safety.

**v2.6 → v2.7 improvements (network resilience, March 29 2026):**
- **NetworkError exception class**: DNS/connection failures (`socket.gaierror`, `ConnectionError`, `URLError` wrapping socket errors, OS errors 101/110/111/113) are now distinguished from GitHub API errors. `gh_get()` raises `NetworkError` instead of returning `None` for transient outages.
- **Network-aware retry**: transient network errors use a 120s retry floor with up to 8 attempts (vs 60s/5 for rate limits). Gives DNS outages time to resolve before giving up.
- **Circuit breaker**: if 5 consecutive accounts fail with `NetworkError`, the scraper pauses for 5 minutes then resumes. Prevents burning through the entire candidate list during a prolonged outage.
- **Skip-not-reject on network error**: accounts that fail due to network issues are NOT written to the status file. They remain unprocessed and will be retried on the next run, instead of being permanently marked as rejected.
- Both stage3a (positives) and stage3b (negatives) are protected by the circuit breaker.

### TEST_RUN flag

Single flag at top of script controls scale:

```
TEST_RUN = True   →  20 pos, 20 neg, cache: classifier_cache_test/, files: test_*
TEST_RUN = False  →  500 pos, 500 neg, cache: classifier_cache_full/, files: full_*
```

### Output files (tagged by run type)

| File | Test | Full |
|------|------|------|
| Login lists | `test_positive_logins.csv` | `full_positive_logins.csv` |
| Neg candidates | `test_negative_candidates.csv` | `full_negative_candidates.csv` |
| Neg status | `test_negative_status.csv` | `full_negative_status.csv` |
| Raw data | `classifier_test_raw.json` | `classifier_full_raw.json` |
| Features | `classifier_test_features.csv` | `classifier_full_features.csv` |
| Claude markers | `test_claude_markers.csv` | `full_claude_markers.csv` |

### Rate limit reality

At 1.0s delay: ~3,600 req/hr. Per account: ~122 calls average (profile + repos + commits + PRs + file samples).
- Test run (40 accounts): ~1 hour
- Full run (400 accounts): ~14-20 hours

Script is resume-safe — kill and restart at any time.

---

## Test Run Results and Coverage Issue (March 2026)

Two test runs completed (v1 scraper + v2.0 scraper). Both showed the same structural problem:

**Coverage:** Only 4 of 20 positive accounts had ≥10 commits in both the pre and post windows. All 20 negatives passed. The both-window filter worked as intended — the issue is that the positive sample is structurally skewed.

**Root cause:** GitHub Code Search for `filename:CLAUDE.md` returns files that exist *now*. This systematically surfaces recent adopters whose accounts may have no meaningful pre-adoption commit history. CLAUDE.md accounts cluster heavily post-2023.

**Signal despite thin sample (both-window accounts only, n=4 pos / 20 neg):**

| Feature | AI users (Δ) | Controls (Δ) | Ratio |
|---------|-------------|--------------|-------|
| Mean message length | +53.8 chars | -3.4 chars | ~16x |
| Frac conventional commits | +0.26 | +0.02 | ~15x |
| Frac PR has body | +0.69 | +0.05 | ~15x |
| Test co-write rate | +0.19 | -0.01 | ~15x |

Signal is real — three independent features all point the same direction at 15-16x effect sizes. The coverage problem does not make the signal go away; it reduces the n available for classifier training.

**v2.1 mitigations:**
- Multi-hour GH Archive co-author scan increases high-confidence positives (these tend to be earlier adopters with more pre-window history)
- Per-account temporal split (high-confidence positives only) avoids artificially narrow pre-windows
- `marker_confidence` column in features CSV enables stratified analysis

---

## Immediate Next Steps (April 25, 2026)

### Completed since April 22

1. ~~Population scrape v3~~ **Done April 25** — 2,999 accounts, 53 countries, post-processing
   crashed once on missing pandas dep, fixed in `run_population_scrape_v3.sh` (`--with pandas`).
2. ~~build_panel_v2.py first run on v3 data~~ **Done** — see April 25 robustness section below.
3. ~~Robustness check battery~~ **Done** — `scripts/robustness_checks.py` ran cleanly.
   Results saved to `data/robustness_results.txt`.

### April 25 Robustness Findings — IMPORTANT

The C-W borderline negative (p=0.055 on commits per dev) survives nearly every spec
we threw at it: dropping `baseline_log_commits` (which is fully absorbed by entity FE
and contributes nothing — the placebo mitigation was a no-op), threshold IV
(`pct_above_0.5`), median classifier score, and a 2024-only OLS cross-section. The
negative direction is robust across specifications.

**But the headline finding is the dependent variable split:**

| DV | Coef | p | Interpretation |
|---|---|---|---|
| `log_commits_per_dev` | -5.14 to -7.66 | 0.05–0.25 | Negative across all specs |
| `log_prs_per_dev` | **+1.33** | **0.76** | Near-zero, slightly positive |
| `log_events_per_dev` | -7.59 | 0.10 | Negative (commits-dominated) |

PRs are the cleaner productivity proxy. They go in the *opposite* direction to commits
and sit on zero. This is consistent with AI tools shifting commit granularity (fewer,
larger commits with longer messages — which is also what the classifier was trained
to detect) rather than reducing productivity.

**Permutation placebo (R6, 2024 cross-section, 1000 perms):** observed coef at the
3.4th percentile of the null distribution. Two-sided p=0.073. Not exonerated, but not
a smoking gun either.

### Revised paper framing — to be implemented

The story is no longer "null result, panel too thin." It is now:

1. **No detectable effect on PRs per dev** (cleaner productivity metric, p=0.76)
2. **Negative association with commits per dev** that is robust across specs but is
   most plausibly explained as a commit-granularity artefact, not a productivity effect
3. **Account-level DiD shows pre/post adoption shifts** in commit message length and
   structure — which is *consistent* with the granularity story
4. The classifier is itself a publishable methodological contribution (AUC 0.940,
   Aider cross-tool generalisation 0.727)

### Paper revision plan

**Format and audience (decided April 25, 2026):**
- **Output**: single Jupyter notebook (`.ipynb`) paper. Code + narrative live together,
  regenerable from data. Convert to PDF for arXiv via Quarto/nbconvert later if needed.
- **Audience**: rigorous enough for an economics reader (panel methods, robustness,
  honest caveats, clustered SEs), accessible enough for CS readers (the classifier
  and behavioural-shift findings are first-class content, not buried in an appendix).
- **Scope**: ONE paper, not two. Classifier methodology and productivity analysis
  bound together — the classifier is the instrument that enables the analysis, and
  separating them weakens both.
- **Tone**: present findings cleanly but caveat hard. The granularity-shift hypothesis
  is consistent with the data, not proven by it. The country-level negative is robust
  across specs but most plausibly a measurement artefact, not a productivity effect.
  Do not oversell. The blog post (separate, later) is where exploratory speculation
  lives; the paper stays disciplined.
- **Octopus format**: deferred. Will be derived from the paper later, not the source.

**Structural ordering for the notebook paper:**
1. Problem & motivation (replicate AI productivity literature with an account-level IV)
2. The classifier as instrument (methods + validation, AUC 0.94, Aider 0.73)
3. Country-level panel regression (Phase 1 baseline, Phase 2 results)
4. Robustness battery (the 14-spec table)
5. The DV heterogeneity finding (commits negative, PRs flat) — present as observation,
   discuss granularity-shift as one plausible interpretation among others
6. Account-level DiD as supporting evidence (without overclaiming causation)
7. Limitations (classifier confound, panel thinness, IV time-invariance,
   commit-granularity ambiguity)
8. Conclusions: a careful null on PRs, a robust-but-ambiguous negative on commits,
   and a methodological contribution (the classifier)

**Implementation steps:**
1. **Choose paper output format**: ~~deciding now~~ **Jupyter notebook** (above).
2. **Update octopus/ chapters with v3 numbers**: ~~deferred~~ — paper first, octopus
   derived later.
3. **Fix `baseline_log_commits`** in `build_panel_v2.py` — it's a no-op (fully
   absorbed by entity FE) and confusing in methods writeup. Either remove or replace
   with a time-varying pre-period feature (slope, volatility) that isn't collinear
   with country FE.
4. **Drop Regression B from the paper** — it's been broken since v2.0 and adds nothing.
5. **Build the notebook**: `notebooks/paper.ipynb`, regenerable end-to-end from
   `data/` files. Each section computes its own numbers from source rather than
   hand-coding values.
6. **Run PR outcome extension before final paper tables**:
   - Scrape authored PRs for the classifier cohort with:
     `GITHUB_TOKEN=... uv run --with pandas --with statsmodels scripts/pr_outcome_metrics.py --scrape --analyse`
   - Smoke-test first with `--max-accounts 5 --max-prs-per-account 25`.
   - Outputs: `data/pr_outcome_cache/`, `data/account_pr_outcomes.csv`,
     `data/account_pr_did_results.txt`.
   - Main account-level PR outcomes: PRs opened/month, PRs merged/month, merge rate,
     closed-unmerged rate, time-to-merge, PR size, review comments, commits per PR.
   - Use these as accepted-output metrics to interpret the commit/PR divergence.
7. **Generate figures inline**: country-level scatter (adoption × productivity, both
   DVs side by side), DV heterogeneity coefficient plot, account-level pre/post
   density of commit message length, and PR outcome coefficient plot if scrape succeeds.
8. **Caveats discipline**: every quantitative claim ties back to a robustness row.
   No floating effect sizes without the spec they came from.

### Analytical improvements made (cumulative)

- **Weighted regression**: countries weighted by n_developers (reduces single-dev noise).
- **Minimum-N threshold**: country-years with <5 developers excluded.
- **Parallel trends test**: 2022→2023 productivity change regressed on 2024 adoption.
- **Country trim**: 16 low-value countries dropped from PANEL_COUNTRIES in v3.
- **Placebo test (account-level)**: pre-period classifier scores compared across quartiles.
- **Robustness battery (April 25)**: 14 specifications across IV variants, DV variants,
  with/without baseline control, and an honest 2024-only cross-section + permutation
  placebo. See `data/robustness_results.txt`.

---

*Last updated: April 25, 2026*
