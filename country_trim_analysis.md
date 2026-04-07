# Country Panel Trim Analysis
# AI Productivity Study — April 2026

Produced by: automated subagent review
Files reviewed: project_plan.md, analysis_results_march2026.md,
data_source_assessment.md, data/regression_results_v2.txt,
data/country_quarter_ai_adoption.csv, scripts/scrape_population.py,
data/population_scores.csv

---

## 1. What the Model is Trying to Measure

Unit of analysis: country × quarter (panel, quarterly windows Q4 2022
through Q4 2024).

Dependent variable: log(commits_per_dev + 1) — GitHub developer
productivity measured as commit events per located developer.

Independent variable (IV): pct_ai_users — the fraction of GitHub
accounts in a country that the behavioral classifier predicts are AI
coding tool users. This is the quantity the population scrape is
building toward.

Estimator: PanelOLS with entity (country) fixed effects and time
(quarter/year) fixed effects. Clustered standard errors by country.

The causal question: does higher AI tool adoption in a country's
developer population cause measurable increases in commit productivity,
after controlling for stable country-level differences and global
time trends?

CRITICAL CURRENT STATUS: The pct_ai_users column in the adoption CSV
is NOT a real per-country measure yet. It is a time-based proxy —
every country is assigned 0.0 for 2022 and 2023, and the identical
value 0.1376 (the mean classifier score across all 235 accounts) for
2024. There is zero cross-country variation in the IV. This is
confirmed by regression_results_v2.txt: Regression B returns a
coefficient of -1.5e13 with SE of 3.4e14 — numerically singular,
because the IV and the time FE are collinear. The model is broken
until the population scrape generates real per-country rates.

---

## 2. Data Sources Per Country and Coverage

The panel joins three data streams:

  A. GitHub productivity panel (github_panel_flat.csv): commits/PRs/
     creates per located developer, per country per quarter. Built
     from random GitHub Archive samples. Coverage is patchy — many
     country-quarters have only 1-2 located developers, making
     per-developer rates extremely noisy.

  B. AI adoption IV (pct_ai_users from population scrape): fraction
     of accounts classified as AI users. This is what the scraper
     is currently building. Target: 30 accounts per country.

  C. Oxford Insights AI Readiness Index: used in Phase 1 (now
     abandoned as IV). Covers 193 countries — not a bottleneck.
     Still useful as a control variable.

Key finding on data source coverage: Oxford Insights and World Bank
traffic data (the recommended replacement IV from data_source_
assessment.md) both cover 150+ countries. The DATA BOTTLENECK is
entirely on the GitHub scrape side — specifically, having enough
scored accounts per country to produce a reliable pct_ai_users
estimate, and enough located developers per country-quarter to
produce a reliable productivity estimate.

Countries missing from the adoption CSV (i.e., no panel observations
at all yet): CL has only 1 quarter. LK has 0 quarters in the panel
despite being in PANEL_COUNTRIES. NP, EG, AR, CO all appear but with
very sparse coverage.

Countries appearing in the adoption CSV but NOT in PANEL_COUNTRIES:
HK (Hong Kong), IR (Iran), PE (Peru) — these appear to be overflow
from location parsing. They should be excluded from the regression.

---

## 3. Current Scrape Progress vs Target (30 accounts per country)

DONE (>=30 accounts):
  IN: 66  US: 63  DE: 37  CN: 32  BR: 32

CLOSE (10-29 accounts, scrape will finish soon):
  FR: 25  GB: 22  KR: 21  PL: 20  BD: 18
  CA: 16  RU: 15  NL: 14  ID: 13  SE: 12
  JP: 12  IT: 11  CH: 11  PK: 10  NG: 10
  ES: 10  AU: 10

PARTIAL (5-9 accounts):
  VN: 8   SG: 8   PT: 7   AT: 7   TR: 6
  PH: 6   KE: 6   DK: 6   MX: 5

THIN (3-4 accounts, will take much longer):
  NP: 4   NO: 4   FI: 4   EG: 4   TH: 3
  NZ: 3   IL: 3   HU: 3   CZ: 3   CO: 3
  BE: 3   AR: 3

CRITICAL TAIL (1-2 accounts — essentially unusable):
  ZA: 2   UA: 2   TW: 2   RO: 2   MY: 2   IE: 2
  SA: 1   LK: 1   GR: 1

At the current scrape rate, reaching 30 accounts in the critical tail
countries (1-2 accounts each) would require scraping 28-29 more
accounts per country. With 9 countries at <=2 accounts, that is ~250
more scrapes just for countries that will contribute marginal
statistical value. The "THIN" tier (12 countries at 3-4 accounts)
needs another 26-27 each, adding ~315 more scrapes. These 21
countries require ~565 additional account scrapes to reach target
with little guarantee they improve model quality.

---

## 4. Which Countries Can Be Dropped and Why

### Criterion 1: Too few panel observations for country FE
A country fixed effect requires at least 2 time periods of variation
to be identified. Countries appearing in only 1 quarter of the
adoption CSV provide a country intercept but zero within-country
variation, contributing nothing to the IV coefficient estimate.

  CL (Chile): 1 quarter only. Drop.
  SA (Saudi Arabia): 1 quarter only AND only 1 account scored. Drop.

### Criterion 2: Population scrape so sparse the IV will be unusable
Below ~10-12 accounts, the estimated pct_ai_users is just noise.
With 1 account, the score is either 0 or 1 — not a population rate.
With 2-3 accounts, the variance is enormous.

  LK (Sri Lanka): 1 account, 0 quarters in CSV. Drop.
  SA (Saudi Arabia): 1 account (already flagged above). Drop.
  GR (Greece): 1 account. Drop.
  IE (Ireland): 2 accounts. Drop.
  MY (Malaysia): 2 accounts. Drop.
  RO (Romania): 2 accounts. Drop.
  TW (Taiwan): 2 accounts. Drop.
  ZA (South Africa): 2 accounts. Drop.
  UA (Ukraine): 2 accounts. Drop (see caveat below).

### Criterion 3: Geographic redundancy — little variation added
The panel model needs cross-country variation in adoption rates and
productivity. Countries that are economically and culturally similar
to better-covered neighbors add little marginal identification power.

  IE — redundant with GB. Both are English-speaking, high-income,
       Western European. GB has 22 accounts and good panel coverage.

  RO — redundant with PL, CZ. Eastern Europe well-covered by Poland
       (20 accounts). Czech Republic also in panel.

  TW — redundant with JP, KR, CN. East Asia well-covered. TW also
       has unique AI access constraints (similar to CN) that may
       invalidate the IV for a different reason.

  MY — redundant with SG, ID, PH, VN. SE Asia already has 4
       countries with better coverage.

  ZA — Africa partially covered by NG (10 accounts) and KE (6).
       ZA adds geographic diversity but at 2 accounts the IV is
       unusable anyway.

  NZ — redundant with AU. AU has 10 accounts and is the English-
       speaking Pacific country with the larger GitHub presence.

  BE — redundant with DE, FR, NL. Western Europe is densely covered.

  CO — redundant with BR, MX, AR. Latin America has better-covered
       representatives.

  TH — redundant with ID, VN, PH, SG, MY. SE Asia is already
       over-represented relative to account coverage.

  HU — redundant with PL, CZ. Eastern Europe well-covered.

### Criterion 4: IV validity concerns
The classifier is trained on GitHub public-repo behavioral signals,
primarily detecting Claude Code, Aider, and Copilot agent mode.
For some countries, these tools may be systematically less available
or used, making the measured pct_ai_users a biased proxy.

  CN (China): ChatGPT access limited; Baidu ERNIE/Tongyi dominate.
    The classifier will undercount AI adoption relative to reality.
    HOWEVER, CN already has 32 accounts (DONE), and it adds large-
    economy diversity. Keep, but flag the validity caveat in the
    paper.

  RU (Russia): Sanctions limit access to Western AI tools. Yandex
    and local alternatives dominate. Same validity caveat applies.
    RU has 15 accounts (nearly at target). Keep with caveat.

  SA (Saudi Arabia): Only 1 account AND Arabic-language AI tools
    not captured by classifier. Already flagged for drop.

### Criterion 5: Countries adding essential heterogeneity
These should be kept even if scrape is slow, because they represent
dimensions of variation not covered by any other country in the panel:

  NG (Nigeria): Only major Sub-Saharan Africa (non-East Africa)
    representative. Different income level, internet infrastructure.
    At 10 accounts — keep and continue.

  KE (Kenya): East Africa, fast-growing tech hub. At 6 accounts —
    continue to 30.

  BD (Bangladesh): South Asia developing economy. At 18 — close to
    done. Keep.

  EG (Egypt): North Africa / MENA. At 4 accounts — borderline.
    The only Arab-majority country after dropping SA. Keep but
    accept it may not reach 30 in time.

  AE (UAE): The only Gulf State in the panel. Not in population
    scores yet but appears in adoption CSV. Check scrape status.

  IL (Israel): Unique tech innovation profile. At 3 accounts —
    slow, but worth keeping for economic heterogeneity.

---

## 5. Recommended Country Set: 38 Countries (cut 16 from 54)

### DROP these 16 countries:

Tier A — Drop with confidence (critical tail + redundant):
  1.  LK  Sri Lanka       — 1 account, 0 panel quarters
  2.  SA  Saudi Arabia    — 1 account, 1 panel quarter, IV invalid
  3.  GR  Greece          — 1 account, EU-redundant
  4.  IE  Ireland         — 2 accounts, GB-redundant
  5.  MY  Malaysia        — 2 accounts, SE Asia-redundant
  6.  RO  Romania         — 2 accounts, E.Europe-redundant
  7.  TW  Taiwan          — 2 accounts, E.Asia-redundant
  8.  ZA  South Africa    — 2 accounts, Africa partially covered
  9.  UA  Ukraine         — 2 accounts, E.Europe covered by PL/CZ
  10. NZ  New Zealand     — 3 accounts, AU-redundant

Tier B — Drop (thin scrape + geographic redundancy):
  11. CO  Colombia        — 3 accounts, LatAm covered by BR/MX/AR
  12. BE  Belgium         — 3 accounts, W.Europe covered by DE/FR/NL
  13. TH  Thailand        — 3 accounts, SE Asia over-represented
  14. HU  Hungary         — 3 accounts, E.Europe covered by PL/CZ
  15. NP  Nepal           — 4 accounts, S.Asia covered by IN/BD/PK

Borderline drop:
  16. AR  Argentina       — 3 accounts. LatAm diversity is limited
        without it (BR=large economy, MX=North America-adjacent).
        Consider keeping if scrape can reach 15+ before the
        regression run. Drop if still at 3 by deadline.

### KEEP these 38 countries (or 39 if AR retained):

Large economies (essential):
  US, IN, DE, CN, BR, GB, FR, CA, AU, RU, JP, KR

Western Europe (substantial GitHub communities):
  NL, SE, CH, ES, IT, NO, DK, FI, PT, AT

Eastern Europe:
  PL, CZ  (two is sufficient)

South/Central Asia:
  BD, PK, EG (borderline), IL

Southeast Asia:
  SG, ID, PH, VN

East Asia (beyond top-tier):
  (JP, KR already listed above)

Africa:
  NG, KE

Latin America:
  BR (already above), MX, AR (if retained)

Middle East / Gulf:
  AE, TR (large economy, bridges Europe-Middle East)

---

## 6. Statistical Power Analysis: Is 38 Countries Enough?

With 38 countries × ~3 time points = ~114 observations.
Parameters: 38 (country FE) + 3 (time FE) + 1 (IV) = 42.
Residual df: ~72.

That is adequate for PanelOLS. The current 54-country model has
54 × ~2.6 obs = 142 observations but 54 FE = very thin residual df.
Moving to 38 well-covered countries with 3+ obs each INCREASES
effective statistical power despite the smaller N, because:
  - More observations per country = better within-country identification
  - The IV will have lower measurement error (scrape at 30 accounts)
  - Country FE are better-estimated with more within-country obs

The minimum viable set for publication credibility is approximately
25 countries spanning multiple continents and income levels. 38 is
comfortably above that floor.

---

## 7. Key Structural Problem to Fix First (Separate from Country Trim)

The country trim is a secondary issue. The PRIMARY problem is that
the current pct_ai_users column has zero cross-country variation:
every country gets 0.0 pre-2024 and the identical global mean (0.1376)
in 2024. This makes the IV collinear with the time fixed effect.

The regression cannot identify anything until country-specific
adoption rates are computed from the scored accounts. The population
scrape must reach at least 15-20 accounts per country before
running any regression.

Once the scrape generates real per-country rates, the country FE
model will have something to estimate. Until then, any regression
result is numerically meaningless (as seen in Regression B).

---

## 8. Concrete Scraper Action Plan

1. Immediately remove these 16 countries from PANEL_COUNTRIES in
   scrape_population.py. The scraper will stop investing time in
   accounts from these countries.

   Remove: LK, SA, GR, IE, MY, RO, TW, ZA, UA, NZ, CO, BE, TH,
           HU, NP, AR (or conditionally AR if still at 3 accounts)

2. DO NOT drop from existing population_scores.csv entries — those
   accounts already scored should be kept for robustness checks.
   Just stop scraping NEW accounts from those countries.

3. Redirect scraper throughput to the THIN tier countries that are
   worth keeping: IL (3), EG (4), NO (4), FI (4), TH... wait TH is
   dropped. Keep scraping toward: IL, EG, NO, FI, DK, MX, PT, AT,
   KE, TR, PH, SG, VN.

4. Set a soft deadline: if any kept country has not reached 15
   accounts by the time the top-20 countries all hit 30, exclude
   it from the regression (but keep it in the dataset for future
   robustness runs).

---

## 9. Selection Bias Assessment

Dropping 16 countries creates a sample that skews toward:
  - Larger economies (by design — more GitHub users, better IVs)
  - Higher-income countries (SE Asian and African thin-scrape
    countries are mostly middle income)
  - Countries with heavier English-language GitHub use (IE, NZ, ZA
    dropped; all English-speaking but redundant with others)

This is acceptable IF:
  a) The paper acknowledges the sample is not globally representative
  b) The final set retains at least 2 African countries (NG, KE) and
     2 Latin American (BR, MX) and 2 South Asian (IN, BD) for the
     broad development gradient
  c) A robustness check with the full set of well-covered countries
     is reported (e.g., subsetting to countries with >= 15 accounts)

The dropped countries are not systematically different in the
outcome we care about (AI-driven productivity change) in ways we
cannot handle. Their absence from the PANEL is primarily because
they lack the data density to contribute valid estimates, not
because of a pre-existing outcome difference.

---

## Summary Recommendation

Cut from 54 to 38 panel countries (drop 16).

The 9 critical-tail countries (<=2 accounts each) should ALL be
dropped — they cannot produce valid per-country IV estimates.
Additionally drop 7 thin-scrape countries (3-4 accounts) that are
geographically redundant: NZ, CO, BE, TH, HU, NP, AR.

This saves approximately 565 account scrapes that would have been
spent reaching target on low-value countries. Those scrapes can
instead go toward getting the 38 kept countries to 30 accounts
each — which is what actually matters for model quality.

The minimum viable set for a credible published IV panel regression
is 25 countries. 38 gives comfortable headroom for robustness
checks, dropped observations due to missing productivity data, and
reviewers asking for heterogeneous-effects subgroup analyses.

Fix the IV collinearity problem (zero cross-country variation) before
re-running any regression. The country trim alone will not fix the
model — the scrape output must produce genuinely different adoption
rates across countries, and the regression must not assign the global
mean to all countries in a year.
