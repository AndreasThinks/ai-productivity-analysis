# AI Adoption Data Source Assessment

Assessment of candidate data sources for the AI adoption index side of the panel study.
Target: country × year panel, 2021–2024 (or a subset thereof).

---

## Summary Table

| Source | Country Coverage | Year Range | Metric Type | Download? | Verdict |
|--------|-----------------|------------|-------------|-----------|---------|
| Stanford HAI Global Vibrancy Tool | 36 (composite), up to 66 (individual indicators) | 2017–2024 | Supply-side + demand-side composite (42 indicators) | No direct CSV — manual extraction or researcher contact | ✅ Strong |
| World Bank / Semrush (Liu & Wang 2025) | 153–218 | Nov 2022–2025 (monthly) | ChatGPT/GenAI traffic penetration (% of internet users) | Request replication data from authors; or buy Semrush API | ✅ Strong |
| Google Trends ("ChatGPT") | ~90 countries | Jan 2023 onwards | Search interest index (0–100, normalised within country) | Yes — free CSV from trends.google.com | ✅ Good, free |
| Ipsos AI Monitor surveys | 28–32 countries | 2021, 2023, 2024 (waves) | Attitude/sentiment ("AI changed daily life", "nervous about AI") | PDF free; microdata on request | ✅ Useful as supplement |
| Oxford Insights Govt AI Readiness Index | 193 countries | 2019–2024 (annual) | Government AI readiness composite | Yes — free download | ✅ Good for 2021 baseline |
| Stanford HAI AI Index (raw data) | ~60–100+ countries | 2017–2023 | AI investment, publications, patents (supply-side) | Yes — Google Drive spreadsheets | ✅ Good supplement |
| Microsoft AIEI AI Diffusion Report | 147 countries | H1/H2 2025 only | GenAI user share (% working-age population, telemetry) | No — PDF tables only | ❌ No historical data |
| ApX ML Engagement Index | 199 countries | 2025 snapshot only | Platform engagement on ApX courses | No — HTML table only | ❌ No historical data |
| BCG CCI 2023 | ~21 countries | 2023 only | % consumers using ChatGPT | No — proprietary | ❌ Single year |
| GitHub Copilot by country | N/A | N/A | Copilot adoption | Not released publicly | ❌ Not available |
| Eurobarometer AI surveys | EU-27 | 2019, 2025 | AI attitudes (not usage) | Yes — GESIS (SPSS/Stata) | ⚠️ EU only, attitudes only |
| Pew Research multi-country AI survey | 26 countries | 2025 only | AI awareness + attitudes | Yes — free data archive | ⚠️ 2025 only |

---

## Recommended Data Strategy

### Primary AI Adoption Variable

**World Bank / Semrush traffic data (Liu & Wang 2025)** is the strongest single source.
It directly measures ChatGPT/GenAI tool penetration as a share of internet users, at country level,
monthly, across 153+ economies. The study has been peer-reviewed in *World Development* (March 2026),
which means replication data should be available.

Action: Email Yan Liu and He Wang (World Bank) for the replication dataset, or contact via
the World Development journal submission system.

Paper: https://openknowledge.worldbank.org/handle/10986/... (WP 11231, June 2025)

Coverage caveat: starts November 2022. No 2021 data — ChatGPT did not exist.

### Pre-ChatGPT Baseline (2021–2022)

For the pre-treatment period, use:

1. **Oxford Insights Government AI Readiness Index** (free download, 193 countries, annual since 2019)
   — captures structural AI readiness: government, technology sector, data/infrastructure pillars.
   URL: https://oxfordinsights.com/ai-readiness/

2. **Stanford HAI AI Index** supply-side indicators (AI investment, publications, patents by country)
   — freely downloadable from Google Drive, 2017–2023.

These serve as the "pre-treatment AI propensity" variable that the DiD/FE model can use
to characterise how ready a country was to adopt AI when tools became available.

### Secondary / Robustness Variables

- **Google Trends for "ChatGPT"** — free, downloadable, available from Jan 2023. Good
  robustness check because it captures demand-side interest independent of access/payment.
  Note: normalised within-country (0–100), not directly comparable across countries.
  Use country fixed effects and relative changes, not cross-section levels.

- **Ipsos survey waves (2021, 2023, 2024)** — attitude variables ("AI changed my daily life",
  "excited about AI products") across 28–32 countries. Useful to control for cultural attitudes
  toward technology, which may confound the adoption–productivity relationship.

- **Stanford HAI Global Vibrancy Tool** — 42 indicators, 36 countries, 2017–2024, annual.
  Best as a robustness check for the supply-side view of AI development.
  Data access: manual extraction from interactive tool, or contact Stanford HAI research team.
  URL: https://hai.stanford.edu/ai-index/global-vibrancy-tool

---

## Panel Coverage Implications

The 2021–2024 target window has an inherent structural break: ChatGPT launched November 2022.
Three design options:

**Option A — 2022–2024 panel (recommended)**
Drop 2021, start from Q4 2022 or early 2023. Use World Bank traffic data as primary AI adoption
variable. Cleaner causal story: we're directly measuring GenAI tool penetration, not a proxy.
Downside: only ~2 years of treatment variation.

**Option B — 2021–2024 panel with two-variable adoption measure**
Use Oxford/HAI supply-side index for 2021–2022, then World Bank traffic data for 2022–2024.
Requires an assumption that pre-treatment AI readiness predicts post-treatment adoption speed
(plausible, can test empirically). Richer panel, but more modelling assumptions.

**Option C — Instrument the adoption variable**
Use pre-2022 AI readiness (Oxford Insights) as an instrument for post-2022 GenAI adoption
(World Bank traffic). IV approach deals cleanly with the data discontinuity and also addresses
reverse causality (more productive developers might adopt AI faster). Strongest causal claim
but requires IV assumptions to hold.

---

## Key Gaps and Risks

1. **World Bank data access**: The Semrush-sourced dataset is proprietary. If the authors
   don't release the replication data, we'd need to reconstruct it via Semrush API (paid) or
   find an alternative. Google Trends is the free fallback.

2. **GitHub location hit rate (~25%)**: Our productivity-side data has significant missingness.
   Need to check whether located users are representative (location field correlates with
   profile completeness, which may correlate with activity level — potential selection bias).

3. **Country overlap**: World Bank covers 153+ countries, but our GitHub sample will likely
   produce credible estimates for only 20–40 countries with enough located users. The intersection
   with AI adoption data that has 36–66 countries is workable but narrow.

4. **Causal interpretation**: Even with FE and DiD designs, reverse causality is possible
   (highly productive developer communities may drive higher AI adoption). The IV approach
   in Option C would be the strongest response to reviewers.

---

## Immediate Next Steps

1. Email World Bank authors for Liu & Wang (2025) replication dataset
2. Download Oxford Insights AI Readiness Index (2021–2024) — free, immediate
3. Download Stanford HAI AI Index raw data from Google Drive — free, immediate
4. Pull Google Trends data for "ChatGPT" for top 40 countries, 2022–2024 — free, immediate
5. Scale up GitHub scraping pipeline to quarterly samples, 2021–2024, ~500 users per window
6. Join datasets and run exploratory FE regression on initial panel
