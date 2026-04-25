# Methodology Decisions Log

*Living document. Every non-obvious analytical or engineering choice is recorded here with rationale, alternatives considered, and date of decision. This is for reproducibility, reviewer transparency, and future-us who will have forgotten why we did what we did.*

*Last updated: April 22, 2026*

---

## Table of Contents

1. [Country Trim: 54 → 38 Panel Countries](#decision-1-country-trim)
2. [Population Scraper v3: Rate-Limit Resilience](#decision-2-v3-scraper)
3. [Classifier Placebo Test](#decision-3-placebo-test)
4. [Panel Regression Improvements](#decision-4-panel-regression)
5. [Expansion Positive Integration](#decision-5-expansion-positives)

---

## Decision 1: Country Trim — Dropping 16 Low-Value Countries

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Implemented in `scrape_population_v3.py`

### Context

The original `PANEL_COUNTRIES` set contained 54 countries. The population scrape v1 (2,048 accounts scored) showed extreme imbalance:
- 5 countries at >=30 accounts (IN, US, DE, CN, BR)
- 20 countries at 10-29 accounts
- 12 countries at 3-4 accounts ("thin" tier)
- 9 countries at 1-2 accounts ("critical tail")

The critical tail countries were consuming API quota for marginal statistical value. Reaching 30 accounts per country in the tail would require ~565 additional scrapes.

### Decision

Drop 16 countries from the panel, retaining 38. Countries dropped fall into three tiers:

**Tier A — Critical tail (<=2 accounts, unusable IV):**
LK, SA, GR, IE, MY, RO, TW, ZA, UA

**Tier B — Thin + redundant (3-4 accounts, covered by neighbors):**
NZ (covered by AU), CO (covered by BR/MX), BE (covered by DE/FR/NL),
TH (covered by ID/VN/PH/SG), HU (covered by PL/CZ), NP (covered by IN/BD)

**Tier C — Borderline:**
AR — kept conditionally if scrape reaches 15+ before regression run. Currently at 3 accounts. May be re-added.

### Rationale

1. **Statistical:** A country fixed effect with <2 observations provides an intercept but zero within-country variation. It contributes nothing to the IV coefficient estimate.
2. **Efficiency:** Those ~565 scrapes can instead go to getting the 38 kept countries to 30 accounts each.
3. **Geographic diversity preserved:** Final 38 includes 2 African (NG, KE), 2 Latin American (BR, MX), 2 South Asian (IN, BD), and representation across all income levels.
4. **Minimum viable set:** ~25 countries is the floor for credible panel IV. 38 gives headroom for robustness checks.

### Alternatives considered

- **Keep all 54, accept thin data:** Rejected. Would produce noisy, unreliable per-country adoption rates and burn API quota.
- **Use only countries with >=15 accounts from the start:** Rejected. Too restrictive — would drop ~25 countries immediately, leaving insufficient geographic diversity.
- **Weight by n_accounts instead of dropping:** Partially adopted (see Decision 4), but dropping is still necessary for countries with essentially zero useful data.

### Impact

- `scrape_population_v3.py` `PANEL_COUNTRIES` updated to 38-country set.
- Existing v1/v2 scores from dropped countries retained in data files for robustness checks, but new scrapes will not target them.
- Expected 30% improvement in scoring throughput (no more wasting quota on low-value countries).

---

## Decision 2: Population Scraper v3 — Rate-Limit Resilience

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Running as systemd service

### Context

Population scrapers v1 and v2 shared a fatal flaw in `gh_get()`: when a 403 rate-limit response arrived, they computed sleep duration from `X-RateLimit-Reset` and stayed inside a `for attempt in range(5)` retry loop. A single blocked API call could chain up to ~5 hours of sleep. When the function finally returned `None` and moved to the next account, the next API call hit the same wall. Result: the process entered a sleep spiral and never recovered.

On April 22, v1 was observed at 22:48 sleeping 3605s (1 hour), then again at 23:48, etc. The process was technically alive but made zero progress for >3 hours.

### Decision

Build v3 with six layers of rate-limit discipline:

1. **Hard sleep cap at 300 seconds (5 min):** Every `time.sleep()` in the retry path capped via `MAX_SLEEP`. No more 3600s spirals.
2. **Distinguish quota exhaustion from abuse detection:** On 403, check `X-RateLimit-Remaining`. If >0, treat as abuse (shorter backoff, max 2 attempts). If 0, sleep until reset (still capped at 5 min).
3. **Proactive `/rate_limit` check:** Before every API call, if global tracker shows <=10 remaining requests, pause proactively.
4. **Global cooldown after 3 consecutive rate-limit hits:** Instead of burning retries per-account, the whole script takes a 5-minute breather and refreshes rate-limit state from GitHub's endpoint.
5. **Jitter on all delays:** `base * (0.75 + 0.5 * random())` so request cadence doesn't look robotic.
6. **Reduced retry greed:** `MAX_RETRIES` down from 5 to 3. `RATE_LIMIT_MAX_ATTEMPTS` capped at 2.

### Rationale

The root cause was not the rate limit itself (the token had 4,996/5,000 remaining), but the retry logic's failure mode. The fix addresses three distinct scenarios:
- **Abuse detection (403 with quota remaining):** GitHub sometimes blocks aggressive patterns even when quota exists. Shorter, capped backoff + jitter reduces false-positive triggers.
- **Primary rate limit (403 with quota exhausted):** Sleep until reset, but don't burn all retries on the same timestamp.
- **Network errors:** Distinguish DNS/connection failures (raise `NetworkError`) from API errors (return `None`), so transient outages don't permanently reject accounts.

### Alternatives considered

- **Patch v1/v2 in place:** Rejected. The retry logic is structurally flawed in both versions. A clean rewrite was safer.
- **Switch to GraphQL exclusively:** Partially adopted (v3 keeps GraphQL batch location pre-filter from v2), but REST is still needed for commit/PR scraping. GraphQL has its own rate limits.
- **Use multiple tokens / rotate PATs:** Rejected. The token itself was fine; the problem was the request pattern.

### Impact

- v3 running as `hermes-population-scraper.service` since April 22 23:27.
- Writes to `population_scores_v3.csv` (fresh file, doesn't collide with v1/v2).
- Target: 3,000 accounts across 38 countries.
- Cron check-in every 3 hours monitors progress and auto-restarts if stalled.

---

## Decision 3: Pre-Period Classifier Placebo Test

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Script written, awaiting execution

### Context

The classifier's top features (message length, conventional commits, multiline messages, bullet lists) are markers of "developer conscientiousness" — careful, structured commit message writing. This trait is correlated with AI tool use, but also with being an experienced, meticulous developer who was already that way before ChatGPT existed.

If the classifier is actually sorting by pre-existing conscientiousness rather than AI adoption, the country-level IV (`pct_ai_users`) becomes endogenous. It measures "fraction of meticulous developers," not "fraction of AI users." The panel regression then measures pre-existing quality → productivity, not AI adoption → productivity.

### Decision

Build `scripts/classifier_placebo_test.py` that:

1. Takes the trained classifier and training features.
2. Constructs pre-only feature vectors: sets every `post_*` = corresponding `pre_*`, every `delta_*` = 0.
3. Scores all accounts using only pre-period information.
4. Splits into quartiles by pre-only score.
5. Compares pre-existing metrics across Q1 (lowest) and Q4 (highest) via Welch's t-test.
6. Also compares pre-only scores between confirmed positives and negatives.

### Interpretation thresholds

- **3+ significant differences (p<0.05):** Classifier is substantially confounded. IV may be endogenous. Consider adding pre-period controls or reframing causal claims.
- **1-2 significant differences:** Partial confound. Flag as limitation, add robustness check.
- **0 significant differences:** Signal appears adoption-specific. Passes the most obvious smell test.

### Rationale

With observational data, a true placebo (randomly assigning AI adoption) is impossible. The pre-period test is the best available check: if accounts that the classifier would later label as "AI users" already looked different before AI tools existed, the classifier is capturing stable traits, not behavioral change.

This is a standard technique in diff-in-diff validity checking. It's weaker than an RCT but stronger than assuming the classifier is perfect because AUC=0.94.

### Alternatives considered

- **Compare to a "placebo year" (2021 vs 2022):** Rejected. We don't have 2021 data for most accounts.
- **Exclude writing-style features entirely:** Already done in ablation (0.940 → 0.909). Behavioural signals carry the model, but writing-style features still dominate importance.
- **Use only pre-period features for training:** Rejected. That would train a "conscientiousness detector" by design, defeating the purpose.

### Impact

- Script ready at `scripts/classifier_placebo_test.py`.
- **CRITICAL RESULT (April 22, 2026):** 6/8 pre-existing metrics show significant differences (p<0.05). Confirmed positives score 0.418 pre-only vs 0.033 for negatives (p<0.0001). The classifier captures pre-existing developer conscientiousness, not purely AI adoption. This must be addressed in the panel regression via pre-period controls or the causal claim must be reframed.
- See Decision 4 (panel regression) for mitigation strategy.

---

## Decision 4: Panel Regression Improvements

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Implemented in `scripts/build_panel_v2.py`

### Context

The original `build_panel_v2.py` had a broken IV: it assigned the global mean classifier score (0.138) to all countries in 2024 and 0.0 in 2022-2023. Zero cross-country variation. The regression was numerically singular (coefficient = 1e12, p = 0.998).

Even after fixing the IV aggregation, three structural problems remained:
1. **Thin panels:** median 2 developers per country-year means single outliers dominate.
2. **No weighting:** Estonia (1 dev) and Germany (25 devs) counted equally.
3. **Parallel trends untested:** No check whether high-adoption countries were already diverging pre-treatment.

### Decision

Four improvements to `build_panel_v2.py`:

1. **Load v1, v2, and v3 scores:** Deduplicate by login across all three scrapers. Maximize available data.
2. **Minimum-N threshold:** Drop country-years with <5 developers before regression. Eliminates single-developer noise.
3. **Weighted regression (Regression C-W):** Weight by `n_developers` so high-confidence observations count more.
4. **Parallel trends diagnostic (Regression D):** Regress 2022→2023 productivity change on 2024 adoption rate. If significant, DiD identification fails.

### Rationale

**Minimum-N threshold:** A country-year with 1 developer isn't measuring "productivity per developer," it's measuring "that one person's productivity." Dropping observations with <5 developers is standard practice in small-N panel work. It trades sample size for reliability.

**Weighted regression:** In the unweighted model, Estonia (1 dev, potentially an outlier) has the same influence as Germany (25 devs, more representative). Weighting by `n_developers` is the econometrically correct approach when the dependent variable is an estimated mean with heteroskedastic variance. Standard errors are already clustered by country; weighting adds a second layer of robustness.

**Parallel trends:** The TWFE design assumes high-adoption and low-adoption countries would have followed the same productivity trajectory if AI hadn't arrived. If countries with high 2024 adoption were already growing faster in 2022→2023, the coefficient is biased. Regression D tests this directly.

### Alternatives considered

- **Event-study design with interacted year dummies:** Deferred. With only 3 time periods (2022, 2023, 2024), the event-study is underpowered. If v3 produces quarterly adoption data, this becomes viable.
- **Two-way clustering (country + time):** Deferred. linearmodels supports `cov_type="twoway_clustered"` but requires balanced panel. Current panel is unbalanced.
- **Arellano-Bond GMM for dynamic panels:** Rejected. Overkill for 3 time periods. GMM needs T>=4 to be reliable.

### Post-placebo-test update (April 22, 2026)

The classifier placebo test revealed significant confounding: 6/8 pre-existing metrics differ between high and low pre-only score quartiles. Confirmed positives score 0.418 pre-only vs 0.033 for negatives (p<0.0001).

**Mitigation:** Added `baseline_log_commits` control to Regression C and C-W. This is the average pre-2024 productivity per country. It partials out the pre-existing productivity difference, so the coefficient on `pct_ai_users` captures the *additional* effect of AI adoption beyond baseline developer quality.

**Limitation:** This control only addresses country-level average pre-existing productivity, not account-level conscientiousness. If within-country variation in adoption is also correlated with within-country variation in developer quality, residual confounding may remain. The paper should flag this explicitly.

### Impact

- `build_panel_v2.py` now produces 4 regressions: A (Oxford baseline), B (broken time proxy, reference only), C (unweighted real IV), C-W (weighted real IV), D (parallel trends).
- Regression C-W is the primary result. Regression D is a validity check, not a causal estimate.

---

## Decision 5: Expansion Positive Integration — Merging 41 Additional Confirmed Positives

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Implemented — `data/classifier_expanded_features.csv` created, model retrained

### Context

The original classifier training set: 33 confirmed Claude Code positives + 202 negatives = 235 accounts. AUC 0.940, but 33 positives is thin for reviewer confidence.

An earlier expansion run (`scripts/scrape_expanded_positives.py`) discovered 300 high-confidence Claude Code users via GH Archive co-author search (2023-Q4). Of these:
- 41 passed the both-window filter (>=5 pre commits AND >=5 post commits)
- 259 were dropped (mostly recent accounts with 0 pre-period commits)

The 41 expansion positives have full feature vectors already extracted.

### Decision

Merge the 41 expansion positives into the training set, retrain the classifier, and save as a new model file. Final training set: 74 positives + 202 negatives = 276 accounts.

### Rationale

1. **Sample size:** 74 positives is more than double the original 33. Reduces variance in feature importance and AUC estimates.
2. **Ground truth quality:** All 74 positives are confirmed via explicit markers (co-author trailers), not pseudo-labels. No leakage risk.
3. **Temporal diversity:** The expansion positives were discovered via a different method (GH Archive co-author scan vs Code Search for CLAUDE.md), reducing selection bias toward any single discovery channel.
4. **Both-window filter preserved:** All 74 positives have meaningful pre/post history, supporting the diff-in-diff design.

**Why not use the 259 dropped accounts?**
They have no meaningful pre-period commits. Adding them would require either:
- Training a post-only classifier (loses the temporal change signal)
- Relaxing the both-window filter (introduces account-age confounds)
Both were rejected. The 259 are retained as an out-of-sample validation pool.

**Why not use all 300 as a post-only classifier?**
A post-only classifier would answer "does this account's 2024 behavior look like a confirmed AI user's 2024 behavior?" This is useful for prevalence estimation but weaker for causal claims. The diff-in-diff design (pre vs post change) is the core methodological contribution. Preserving it is worth the smaller sample.

### Implementation steps

1. Load `data/classifier_full_features.csv` (original 235 accounts)
2. Load `data/expansion_features.csv` (41 expansion positives)
3. Add `tool_type` column for consistency (all 'claude' in this merge)
4. Concatenate, verify no login duplicates
5. Retrain Random Forest with same hyperparameters (200 trees, max_depth=4)
6. Save new model bundle to `data/classifier_model_expanded.pkl`
7. Run 5-fold stratified CV, report AUC
8. Run placebo test on retrained model

### Impact

- New model file: `data/classifier_model_expanded.pkl`
- Training set: 276 accounts (74 pos, 202 neg)
- If AUC remains ~0.94, the model is stable and the expansion adds confidence.
- If AUC drops significantly, the expansion positives may differ systematically from the original positives (e.g., different tool usage patterns). This would be flagged for investigation.

### Open questions

- Should the 259 dropped expansion accounts be used as an out-of-sample validation set? Yes — score them with the retrained model and report mean score.
- Should the original `classifier_model.pkl` be preserved? Yes — kept for comparison and reproducibility.

---

## Decision 6: Model Persistence and Versioning Convention

**Date:** April 22, 2026
**Decision maker:** Avery (agent), Andreas (user) approved
**Status:** Implemented

### Decision

All model files are versioned with descriptive suffixes:
- `classifier_model.pkl` — original model (33 pos, 202 neg)
- `classifier_model_expanded.pkl` — expanded model (74 pos, 202 neg)

Each bundle contains: `model`, `imputer`, `feature_cols`, `model_name`, `training_date`, `n_positives`, `n_negatives`.

Scripts that load models accept an environment variable `CLASSIFIER_MODEL` defaulting to `classifier_model_expanded.pkl`.

### Rationale

Reviewer reproducibility and rollback safety. If the expanded model performs worse, we can revert to the original without retraining.

---

## Outstanding Decisions (To Be Made)

1. **Quarterly vs annual aggregation:** If v3 produces enough data, should we switch from country-year to country-quarter panel? More time variation but thinner per-quarter observations.
2. **Alternative IV:** Google Trends "ChatGPT" as a free, fast alternative to the classifier-based IV. Could be run in parallel as a robustness check.
3. **Negative set expansion:** The 202 negatives are from the original scrape. Should we add more random negatives from v3 to balance the class? Not critical (random forest handles imbalance well), but worth considering.
4. **Publication framing:** If the final regression is null even after weighted regression and expanded training, the paper becomes a carefully-framed null result. "We built a behavioral classifier for AI adoption, applied it across 38 countries, and found no detectable effect on commit productivity." This is a real contribution if the methods are sound.

---

*End of decisions log. Every change to methodology after this date should be appended as a new Decision entry with date, rationale, and impact.*
