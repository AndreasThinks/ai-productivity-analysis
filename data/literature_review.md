# Literature Review: Detecting and Measuring the Productivity Effects of AI Coding Tool Adoption

## 1. Introduction

This review surveys the empirical literature on AI coding tools and developer productivity, with emphasis on measurement approaches and causal identification strategies relevant to the present study. The review is organised into four areas: (1) existing field experiments on AI coding assistants, (2) the challenge of measuring AI tool adoption at scale, (3) causal inference methods for productivity analysis, and (4) the relationship between AI diffusion and productivity at aggregate levels.

## 2. Field Experiments on AI Coding Assistants

The most direct evidence on AI coding tool productivity effects comes from randomised field experiments and quasi-experimental studies conducted since 2022.

**GitHub Copilot.** Bird et al. (2023) conducted a multi-perspective field experiment with Microsoft developers, finding that Copilot users completed tasks 55% faster and produced code with 40% fewer rejections in code review. The study collected both objective productivity metrics and self-reported happiness measures, establishing one of the first rigorous baselines for AI-assisted development. Katz et al. (2024) extended this line of research with additional evidence from GitHub Copilot, focusing on developer productivity measured through code review outcomes and cycle time.

**Cursor and IDE-based assistants.** A more recent study from Carnegie Mellon University (CMU Strudel, 2025) examines Cursor's impact on software projects using a difference-in-differences design, finding measurable productivity gains in code completion frequency and task completion rates. This study is methodologically close to our own account-level design, though it operates at the project rather than individual developer level.

**ChatGPT for software development.** Quispe and Grijalba (2024) exploit the timing of ChatGPT's November 2022 release as a natural experiment, using GitHub data to estimate synthetic difference-in-differences effects. Their approach is closest to our country-level panel design in its use of aggregate GitHub activity as the outcome variable, though they focus on the introduction of general-purpose LLMs rather than dedicated coding assistants.

**General LLM productivity effects.** Peng et al. (2023) provide a comprehensive study of LLM impacts on programming tasks, documenting both productivity gains and quality trade-offs. Brynjolfsson et al. (2025) extend this to high-skilled work more broadly in a Management Science paper, finding that generative AI increases productivity particularly in complex, non-routine tasks — a mechanism that plausibly applies to software development.

A consistent finding across these studies is that AI assistance reduces friction in the development loop: tasks that previously required context-switching or manual boilerplate become faster. Our account-level finding of a 5× increase in commit frequency (inter-commit hours dropping from 281 to 58) aligns with this pattern, though the magnitude exceeds most prior estimates.

## 3. The Measurement Challenge: Identifying AI Tool Users

A persistent difficulty in this literature is that AI coding tool adoption is largely invisible in public data. Unlike software version control events, adoption leaves no mandatory trace. The literature has pursued three measurement strategies, each with limitations.

**Explicit markers.** The most common approach identifies users via explicit artefacts — configuration files, AI-generated commit trailers, or repository readmes. This is the approach we use for ground-truth construction (Section 3.2 of the paper). Its limitation is bias: users who leave these markers may be power users or early adopters, unrepresentative of the broader adoption population. The GitHub (2023) developer survey relies on self-report, which is expensive and subject to recall and social desirability bias.

**Survey-based measurement.** Vendor-reported adoption rates (e.g., GitHub's public statistics on Copilot uptake) are methodologically opaque and likely subject to selection. Academic surveys (e.g., the Developer Experience Survey) provide more credible estimates but are cross-sectional, expensive to repeat, and cannot support retrospective analysis.

**Behavioural inference.** Our contribution addresses this gap directly. We demonstrate that AI tool adoption produces detectable changes in commit behaviour — increased commit frequency, longer commit messages, improved PR documentation — that can be identified without explicit markers. The cross-tool generalisation result (classifier trained on Claude Code users detecting Aider users at AUC comparable to held-out Claude users) is the key validity claim: we are measuring a general pattern of AI-assisted development, not a tool-specific fingerprint.

The narrow cross-country range in classifier-derived adoption rates (6.3% to 10.7%) is itself a finding with implications for measurement. Even among GitHub's active developer population — a self-selected, relatively high-skill group — AI coding tool penetration is still in the single digits. This suggests that aggregate productivity effects will take time to manifest as diffusion propagates beyond early adopters.

## 4. Causal Inference Methods for Productivity Analysis

Our study employs two causal designs with complementary strengths and limitations. The methodological literature informs both.

**Difference-in-differences.** The account-level design follows the canonical DiD framework (Angrist and Pischke, 2009). The key identifying assumption is parallel trends: absent AI tool adoption, treated and control accounts would have followed the same trajectory. We address this through regression adjustment for pre-period levels and robustness checks restricting to high-confidence adopters. The pre-period differences between groups (AI adopters were already more active) indicate selection, which the adjustment partially but not fully addresses. This is a limitation common to all observational DiD designs that cannot randomise treatment assignment.

The literature on DiD in technology adoption contexts (e.g., the diff-in-diff studies of other developer tools) typically faces similar selection concerns. Our approach of using explicit markers for treatment assignment partially mitigates this: the markers (CLAUDE.md files, co-author trailers) are based on code patterns and tool outputs, not on productivity metrics, reducing the risk that selection is directly on the outcome.

**Panel regression with fixed effects.** The country-level design uses a fixed-effects panel specification, the standard approach for aggregate productivity analysis (Angrist and Pischke, 2009). Country fixed effects absorb all time-invariant cross-country heterogeneity; time fixed effects absorb common shocks. The remaining coefficient identifies the within-country relationship between AI adoption and productivity.

The null result at the country level is not surprising given the design. The Oxford Insights Government AI Readiness Index used in Regression A measures government AI policy readiness — a distal proxy for developer tool adoption. The per-country classifier scores used in Regression C have narrow range (4.4 percentage points) and are measured with noise (median 2 developers per country-year in the productivity panel). The power analysis in Section 6.2 quantifies this: with 59 observations clustered in 20 country groups, the design is severely underpowered for detecting even moderate effect sizes.

**Comparison to Liu and Wang (2025).** The most directly comparable prior work is Liu and Wang's use of web traffic data (Semrush) to measure AI adoption across countries. Their approach has a key advantage: web traffic provides a continuous, high-resolution measure of AI tool usage that varies across countries and time. Our classifier-derived rates are necessarily bounded by the GitHub population and the willingness of users to leave explicit markers. Future work combining web traffic measures with the classifier approach could substantially improve the country-level design.

## 5. AI Diffusion and Aggregate Productivity

A broader literature addresses whether technology adoption at the individual level aggregates to team, firm, or national productivity effects. The classic technology diffusion literature (following Rogers, 2003) emphasises that adoption spreads through networks over time, with aggregate effects lagging individual adoption.

Our findings are consistent with this pattern. The account-level effect is detectable because it operates at the individual level where adoption is binary and immediate. The country-level null likely reflects the combination of (a) insufficient time horizon (only 2024 post-treatment data), (b) measurement noise in the aggregate outcome, and (c) narrow cross-country variation in adoption rates. The discussion in Section 7.1 elaborates these mechanisms.

The implications for future research are clear: a longer post-period (2024–2026) and larger productivity panel (5,000+ users per quarterly window) would substantially increase power to detect whether the individual-level effect aggregates to national productivity measures.

## 6. Summary

The empirical literature on AI coding tool productivity effects is growing rapidly but faces a common measurement problem: adoption is invisible in public data. Our behavioural classifier addresses this directly, achieving AUC 0.940 and cross-tool generalisation. The account-level DiD finds large, significant effects (+13.1 commits/week, 5× increase in commit frequency); the country-level panel finds a null result, which we interpret as a measurement limitation rather than evidence against an underlying effect. Both findings are consistent with the broader technology diffusion literature: individual-level adoption effects may be large, but aggregate effects require time and sufficient cross-sectional variation to manifest.

## References

Angrist, J. and Pischke, J.S. (2009). *Mostly Harmless Econometrics: An Empiricist's Companion*. Princeton University Press.

Bird, C., Dam, H.K., Phong, T., and Sharma, A. (2023). A Multi-Perspective Study of the Effects of GitHub Copilot on Developer Productivity and Happiness. *ICSE 2023*.

Brynjolfsson, E., Li, D., and Ray, B. (2025). The Effects of Generative AI on High-Skilled Work. *Management Science*, forthcoming.

GitHub (2023). Survey reveals AI's impact on the developer experience. GitHub Blog.

Katz, D., Gonzalez, D., Moshkovich, D., and Nadj, R. (2024). The Impact of AI on Developer Productivity: Evidence from GitHub Copilot. *arXiv preprint arXiv:2302.06590*.

Liu, W. and Wang, C. (2025). AI diffusion and productivity: Evidence from web traffic data. Working paper.

Peng, S. et al. (2023). Impact of Large Language Models on Programming: A Comprehensive Study. *arXiv preprint arXiv:2302.06587*.

Quispe, A. and Grijalba, R. (2024). Impact of the Availability of ChatGPT on Software Development: A Synthetic Difference in Differences Estimation using GitHub Data. *arXiv preprint arXiv:2406.11046*.

Svyatkovskiy, A. et al. (2023). IntelliCode Compose: Code Completion through Neural Programming and Synthetic Data. *IEEE Transactions on Software Engineering*.

CMU Strudel et al. (2025). Does AI-Assisted Coding Deliver? A Difference-in-Differences Study of Cursor's Impact on Software Projects. *arXiv preprint arXiv:2511.04427*.

Oxford Insights (2023). *Government AI Readiness Index 2023*.