# Octopus Publication: Interpretation

**Title:** Reconciling individual-level behavioural change with a country-level null: what the commit signature of AI coding tool adoption means

**Keywords:** AI coding tools, developer behaviour, null result, individual effects, aggregate effects, measurement, causal inference, limitations

**Links to:** Analysis — "Account-level difference-in-differences and country-level panel regression analysis"

---

## 1. Reconciling the Individual and Country-Level Results

The account-level and country-level results appear to contradict each other. The account-level difference-in-differences finds large, highly significant changes in commit behaviour for AI adopters: 13 additional commits per active week, a five-fold increase in commit frequency, and substantially improved pull request documentation, all significant at p < 0.001 and robust to high-confidence restriction and outlier winsorisation.

These estimates are likely upper bounds, as the treated sample consists of early, intensive adopters and the classifier selection mechanism is partly based on the same behavioural features used as DiD outcomes.

The country-level panel regression, by contrast, finds no detectable effect (coef = −6.06, SE = 7.96, p = 0.451), with the observed coefficient falling well within the range of what random country assignment would produce.

We argue these results are consistent under three mechanisms that likely operate jointly.

**Measurement noise in the outcome variable.** The country-level commit activity panel is constructed from hourly GH Archive samples, yielding a median of 2 located developers per country-year observation. With this degree of sparsity in the dependent variable, an effect would need to be implausibly large to be statistically detectable. The individual-level result (13 additional commits per active week) is a large within-person change; scaled to a country-year average over two developers drawn from a noisy sample, the variance dominates any signal.

**Narrow cross-country variation in adoption rates.** Per-country adoption rates range from 6.3% (Italy) to 10.7% (Netherlands) — a spread of 4.4 percentage points. With such compressed cross-sectional variation in the treatment intensity, the panel regression has very limited power to identify a slope, even absent noise in the dependent variable. The placebo test confirms this: 26.9% of random country assignments produce a coefficient at least as extreme as our observed value, indicating that the panel is not informative about whether there is a country-level effect.

**Insufficient post-period.** The post-period covers a single year (2024). Country-level effects of technology adoption typically require several years to manifest, as diffusion propagates through teams, organisations, and the broader developer population beyond early adopters. The account-level effect is detectable in 2024 precisely because it operates at the individual level, where adoption is binary and immediate.

---

## 2. What the Commit Signature Means

The dominant account-level findings — reduced inter-commit hours and increased commits per active week — describe a consistent pattern: AI adopters commit more frequently within their active coding sessions, not necessarily more often across calendar time (active weeks actually decline post-adoption). This is consistent with AI assistance lowering the marginal cost of incremental commits: generating a commit message, resolving a test failure, or refactoring a function becomes fast enough that developers commit at a finer granularity.

The increased PR body rate (+32 percentage points) and improved message documentation are also consistent with AI assistance making documentation faster to produce.

**An important caveat.** These changes in commit behaviour are not equivalent to changes in developer output or productivity. A developer who makes 20 small commits instead of 5 large ones is not necessarily four times more productive. The commit signature this paper detects reflects changes in *how developers work* — their rhythm, cadence, and documentation habits — not a direct measure of features shipped, issues resolved, or code quality. Future work that links commit behaviour changes to output-based measures (PR merge rate, issue closure, deployment frequency) would be needed to establish that the behavioural changes correspond to genuine output gains.

---

## 3. The Classifier as a Measurement Contribution

A result independent of the commit activity question is the demonstration that AI coding tool adoption can be detected at scale from public commit behaviour at AUC 0.940, generalising across tools. This has practical implications for measurement.

Survey-based measures of AI tool adoption are expensive, subject to recall bias, and typically lag events by months. Adoption rates reported by vendors are methodologically opaque. The classifier developed here offers a complementary approach: a scalable, non-survey measure derivable from public data, applicable retroactively to any period for which GitHub Archive data is available, and validated to generalise across tools rather than detecting a single vendor's stylistic footprint.

The ablation result — AUC 0.909 using only commit timing and frequency features, without any inspection of message content — is particularly relevant for privacy considerations. A deployment using only commit timestamps and counts, with no analysis of commit content, would be technically feasible and minimally intrusive.

---

## 4. Limitations

**Selection.** The treated accounts are identified via explicit public AI markers — CLAUDE.md files, co-author trailers. These are almost certainly not representative of the full population of AI tool users. They are likely power users, early adopters, and developers who intentionally configured their tooling to leave traces. The effect sizes are plausibly upper bounds on the average treatment effect for the broader AI-using developer population.

**Parallel trends.** AI adopters show significantly higher pre-period activity on several dimensions, indicating that they were on a steeper trajectory before adoption. The regression adjustment controls for pre-period levels but cannot fully eliminate this selection. The treatment effect estimates should be interpreted as upper bounds, not unbiased causal estimates.

**Classifier–DiD circularity.** The classifier is trained using behavioural features — changes in commits per active week and inter-commit hours — that also serve as primary DiD outcome variables. Accounts classified as AI adopters are partly those with the largest shifts in these same metrics, creating a mechanical upward bias in the treatment effect estimates. The DiD adds information about the magnitude of change conditional on the classifier's selection, and the cross-tool generalisation confirms the behavioural signature is real, but the effect sizes should be interpreted as upper bounds rather than unbiased causal estimates of the ATT.

**Temporal confound.** The post-period (2024) also corresponds to a period of rapid improvement in AI tool capabilities. Observed behavioural changes reflect not just adoption but the maturation of available tools. Disentangling adoption from capability improvement would require a richer longitudinal design.

**Classifier-DiD relationship.** The classifier is trained to discriminate AI adopters from controls using the same behavioural features (commit cadence, message structure) that serve as outcomes in the DiD. Accounts classified as AI users are those with the largest behavioural shifts. The DiD adds information about the direction and magnitude of change conditional on baseline, but the two analyses are not fully independent. Future work should ideally test against outcome measures not included in the classifier feature set.

**Panel thinness.** The country-level panel is built from hourly GH Archive samples, yielding a median of 2 developers per country-year. Scaling the panel scrape to 5,000+ users per quarterly window would substantially reduce measurement error in the dependent variable and improve power.

---

## 5. Directions for Future Work

The most productive extension would address the country-level measurement problems directly: a larger commit activity panel (5,000+ users per quarterly window) and a longer post-period (2024–2026) would substantially increase power and allow the country-level design to test whether individual-level commit behaviour changes aggregate to national-level patterns. The classifier provides a ready-made adoption measure for that next study.

Future work might also: (1) apply the classifier to proprietary firm-level data where true adoption is known, to validate precision estimates; (2) link commit behaviour changes to output-based measures (PR merge rate, issue closure velocity) to test whether the behavioural signature corresponds to genuine output changes; and (3) extend the cross-tool validation to Copilot, Cursor, and other tools with distinct stylistic signatures.

---

*Working paper: https://andreasthinks.me/papers/ai-productivity*
*Code and data: https://github.com/AndreasThinks/ai-productivity-analysis*
