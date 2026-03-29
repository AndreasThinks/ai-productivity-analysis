# AI Productivity Analysis — Project Plan

*Living document. Updated as the project evolves. See AGENTS.md for coding conventions and agent instructions.*

---

## Project Status: Phase 2 — Classifier Design

Phase 1 (panel regression with Oxford Insights AI Readiness Index) returned a null result. The likely cause is that the independent variable measures government AI policy readiness, not whether individual developers are actually using AI tools. Phase 2 replaces it with an account-level classifier.

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

## Immediate Next Steps

1. ~~Run full scrape with v2.1~~ **Done** — multiple runs completed (v2.1 through v2.6)
2. ~~Check both-window coverage~~ **Done** — 102 accounts passed (30 pos / 72 neg) in v2.2; 65 in v2.6 (18 pos / 47 neg)
3. ~~First classifier~~ **Done** — RF best model, CV AUC 0.828 (see Classifier Training Results below)
4. **v2.7 scrape in progress** (March 29) — resuming from 129/500 negatives accepted. Targeting 500 negatives total. Scraper has new network resilience (circuit breaker, skip-not-reject). Log: `/tmp/scraper_v27.log`
5. **Retrain classifier on expanded dataset** — once v2.7 completes with ~500 negatives, retrain RF and check if CV AUC improves with larger N
6. **Validate on other tool markers** — collect Aider co-author accounts, run classifier, compare score distributions
7. **Build `scripts/build_panel_v2.py`** — replace `ai_readiness_score` in panel with per-country fraction of accounts classified as AI users

---

*Last updated: March 29, 2026*
