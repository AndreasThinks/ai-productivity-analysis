# Peer Review: "Detecting and Measuring the Productivity Effects of AI Coding Tool Adoption"

**Manuscript**: Working Paper, April 2026
**Author**: Andreas Varotsis
**Reviewer**: Anonymous (Senior Academic Reviewer)
**Date**: 7 April 2026

---

## Summary

This paper makes two contributions: (1) a behavioural classifier that identifies AI coding tool users from public GitHub commit metadata (AUC 0.940), validated across tools (Claude Code and Aider); and (2) two causal designs — an account-level difference-in-differences (N=235) and a country-level panel regression (20 countries, 2022–2024) — estimating the productivity effects of AI tool adoption. The account-level DiD finds large, statistically significant effects (13.1 additional commits per active week, 275-hour reduction in inter-commit time). The country-level regression finds a null result, attributed to measurement noise, narrow cross-country variation, and an insufficient time horizon. The paper is well-written and methodologically self-aware, with appropriate caveats about its limitations.

---

## Overall Assessment

This is a promising working paper with a genuinely novel methodological contribution (the classifier) and an honest, transparent treatment of its own limitations. However, several major issues — particularly around the causal interpretation of the account-level results, the extremely small treated sample (N=33), and the conflation of commit frequency with "productivity" — need to be addressed before the paper is ready for a peer-reviewed venue. The country-level analysis, by the author's own admission, is severely underpowered and contributes relatively little beyond demonstrating the null. The paper would benefit from a clearer separation of its two contributions (classifier vs. productivity estimation) and a more cautious framing of what the account-level results actually show.

**Verdict: Major Revisions**

---

## Major Issues

### 1. Commit frequency is not productivity

The paper repeatedly equates changes in commit behaviour — commits per active week, inter-commit hours, PR documentation — with "developer productivity." This is the paper's central interpretive weakness. More frequent commits may reflect finer-grained version control habits induced by AI tools (as the author notes in Section 7.3), not higher output. A developer who commits the same total code in 20 small commits instead of 5 large ones is not 4x more productive. The abstract states the paper studies "the effect of AI coding tool adoption on developer productivity," but what it actually measures is *commit behaviour change*. The title should reflect this, or the paper needs to provide evidence that commit frequency proxies for genuine output (e.g., lines of code, features shipped, issues closed). At minimum, the abstract and introduction need to sharply distinguish between "productivity" and "commit cadence."

**Fix**: Either (a) reframe the paper as measuring behavioural change rather than productivity, or (b) add output-based measures (LOC, issues closed, PR merge rate) to establish that commit frequency changes correspond to actual output changes.

### 2. Treated sample is extremely small (N=33) and non-representative

The entire account-level causal analysis rests on 33 confirmed AI adopters (25 high-confidence). These are developers who left explicit public markers of AI tool use — CLAUDE.md files, co-author trailers. The paper acknowledges these are "almost certainly not a representative sample" and likely "power users, early adopters." This is correct, but the implications are underplayed. The treatment effect estimates (13.1 commits/week, 275-hour inter-commit reduction) are based on extreme adopters and are likely severe upper bounds. More importantly, with N=33 treated units, the estimates are highly sensitive to outliers. The paper does not report any outlier analysis, winsorisation, or trimmed estimates for the account-level DiD.

**Fix**: (a) Report median effects alongside means; (b) conduct and report outlier sensitivity (e.g., winsorise at 5th/95th percentiles, drop the top 3 treated accounts and re-estimate); (c) be much more explicit in the abstract and conclusion that these are effects *for extreme early adopters*, not the general developer population.

### 3. Severe selection bias in the account-level DiD

The paper acknowledges that AI adopters "show significantly higher pre-period activity on several dimensions" (Section 6.2) and that "the regression adjustment controls for pre-period levels but cannot eliminate this selection." This is too mild a statement for the severity of the problem. The DiD specification:

$$\Delta Y_i = \alpha + \beta \cdot \text{Treatment}_i + \gamma \cdot Y^{\text{pre}}_i + \varepsilon_i$$

controls for the *level* of pre-period outcomes but not for pre-period *trends*. Developers who are already on a steeper upward trajectory (more active, more engaged) are precisely the ones likely to adopt AI tools early. The parallel trends assumption is not just "untestable" — it is actively implausible given the documented pre-period differences. The regression adjustment mitigates but does not solve the selection problem.

**Fix**: (a) If pre-period data has enough temporal resolution, show pre-treatment trends for treated vs. control (even 2–3 pre-period time points would help); (b) consider propensity score matching or coarsened exact matching on pre-period characteristics; (c) at minimum, frame $\beta$ as an upper bound on the ATT rather than a point estimate.

### 4. The classifier is trained on the same data used for the DiD

The classifier is trained to distinguish AI adopters from non-adopters using pre/post behavioural features. The account-level DiD then estimates treatment effects on the *same behavioural features* (commits per active week, inter-commit hours) for the same accounts. This creates a mechanical relationship: accounts classified as AI users are *by construction* those with the largest behavioural shifts, so finding large treatment effects in the DiD is partly tautological. The paper needs to clearly address this circularity.

**Fix**: The DiD should ideally use outcome measures that are *not* classifier features (e.g., issues, PRs merged, release frequency). If using the same features, the paper must explicitly discuss the mechanical linkage and argue why the DiD adds information beyond the classifier's discriminative ability.

### 5. "We" authorship voice for a single-author paper

The paper uses "we" throughout but has a single listed author. This is common in economics but worth flagging — if the paper is single-authored, "I" is more appropriate for many venues. More importantly, if AI tools (Claude) were used extensively in writing the paper (as the git history suggests), this should be disclosed in a note or acknowledgements section. Transparency about AI assistance in writing a paper *about AI coding tools* is particularly important for credibility.

**Fix**: Add an acknowledgements section disclosing any AI assistance in writing the paper, data collection, or analysis.

---

## Minor Issues

1. **Missing citation**: `@anthropic` is referenced in the paper text but has no corresponding entry in `references.bib`. This will produce a broken citation in any rendering pipeline.

2. **Unused bib entries**: `bird2023copilot`, `katz2024impact`, and `svyatkovskiy2023intellicode` are defined in `references.bib` but never cited in the paper. Bird et al. (2023) and Katz et al. (2024) are highly relevant and should arguably be cited in the literature review — omitting them while citing less central work is a gap.

3. **Incomplete reference list in the paper**: The References section at the end of the paper (Cell 27) lists only 3 references (Angrist & Pischke, Liu & Wang, Oxford Insights) — the rest are presumably handled by the BibTeX rendering pipeline, but since this is a Jupyter notebook, not a LaTeX document, the `@citation` syntax will render as literal text in most viewers. The paper needs either a complete reference list or a functioning citation rendering pipeline.

4. **Section numbering inconsistency**: The robustness checks for the country-level analysis appear under Section 6.5, but the power analysis and heterogeneity analysis are numbered 6.6 and 6.7. The robustness checks in `missing_sections.md` were originally numbered as Section 7.4 (under Discussion). The current numbering places all of these under Results (Section 6), which makes the Results section disproportionately long. Consider moving power analysis and heterogeneity to the Discussion section.

5. **Regression B is pointless**: The paper includes a "time proxy" regression (B) that produces a numerically degenerate result (coef ~ 10^12) due to perfect collinearity, then notes this was "expected." Including a specification you know will fail adds nothing and confuses readers. Either drop it or explain upfront why it is included as a pedagogical foil.

6. **Inconsistent N**: The abstract says "235 accounts (33 confirmed adopters, 202 controls)" but the heterogeneity section references "859 classified accounts." The relationship between these samples is unclear. Are the 859 the population scoring sample? This needs clarification.

7. **p=0.065 is not "statistically indistinguishable"**: Section 4.5 states that Aider users' scores are "statistically indistinguishable from the Claude training positives (p = 0.065)." At conventional significance levels (0.05), p=0.065 is marginally significant and does *not* support a claim of indistinguishability. Rephrase to "not significantly different at the 5% level" or similar.

8. **The placebo test interpretation is confused**: Section 6.5 reports that the 95th percentile of placebo coefficients is 0.12, while the actual coefficient is -4.91, and concludes this is "well outside the placebo range." But -4.91 is negative and the 95th percentile is the upper tail of the placebo distribution. The relevant comparison is the 5th percentile (or the two-tailed interval). As written, this test actually suggests the observed negative coefficient is anomalous — which undermines rather than supports the paper's interpretation. This needs careful re-examination.

9. **"Median of 2 located developers per country-year"**: This is buried in the Discussion (Section 7.1) and is devastating for the country-level analysis. If the median country-year cell has *two* developers, the country-level productivity estimates are essentially noise. This should be disclosed prominently in the Data section, not the Discussion.

10. **No table of summary statistics**: The paper references "Table 4" for pre-period differences but the notebook shows a code cell (Cell 6) that presumably generates summary statistics — but no rendered table is visible in the markdown. The paper needs a clearly labelled Table 1 with descriptive statistics for the full sample, treated, and control groups.

11. **No ethics/IRB discussion**: The paper scrapes individual GitHub account data (commit histories, locations, activity patterns) to classify developers' tool usage without consent. While GitHub data is public, the ethical implications of building surveillance-like classifiers on developer behaviour deserve at least a brief discussion, especially given the privacy angle mentioned in Section 7.2.

---

## Section-by-Section Notes

### Abstract
Well-structured and honest about the null result. The AUC 0.940 is prominently featured, which is appropriate. However, the claim that the paper studies "productivity effects" is overclaimed — see Major Issue 1. The abstract should note the small treated sample size more prominently.

### 1. Introduction
Strong motivation of the measurement problem. The three challenges (invisibility, selection, unit of analysis) are well-articulated. The roadmap paragraph is clear. The introduction would benefit from a clearer statement of what "productivity" means in this paper (commit-based behavioural proxies, not output measures).

### 2. Literature Review
This is the strongest prose section of the paper. The three-stream organisation (experiments, observational, measurement) is effective. The discussion of Demirer et al. (2025), METR (2025), and Quispe & Grijalba (2024) is detailed and accurate. The positioning of the paper's contributions against the literature is compelling. However:
- Bird et al. (2023) — the Microsoft Copilot field study — is defined in the bib but not cited. This is a major omission for a paper on AI coding tool productivity.
- The "Gaps and Contributions" subsection (2.4) effectively sets up the paper's value-add.

### 3. Data
Clear description of the three data sources. The ground-truth labelling methodology is transparent and the distinction between high/low confidence markers is appropriate. However:
- The "median of 2 developers per country-year" should be stated here, not in Section 7.
- No discussion of potential biases in GitHub Archive sampling (e.g., bot accounts, CI/CD commits, mirrored repositories).
- The 21% location parse rate is noted only in the Limitations section — it belongs here.

### 4. Behavioural Classifier
The strongest methodological section. The design rationale is clear, the feature leakage concern is addressed head-on, the ablation is informative, and the cross-tool validation is genuinely novel. The AUC of 0.909 with activity-only features is a strong result.

Concerns:
- Feature importance (inter-commit hours 0.130, message length 0.120) is reported for Random Forest but no SHAP values or partial dependence plots are shown. For a classifier that is itself a contribution, more interpretability analysis would strengthen the paper.
- No discussion of calibration. AUC is a discrimination metric; for the classifier to be used as a measurement tool (as proposed), calibration matters. Are the predicted probabilities well-calibrated?
- The 43 features for 235 observations (43/235 = 0.18 features-per-observation ratio) raises overfitting concerns, despite the cross-validation. Regularisation details are not discussed.

### 5. Causal Designs
The account-level DiD specification is clearly presented. The country-level panel design is standard. However:
- The term "IV" is used loosely. In Regression C, `pct_ai_users` is not an instrumental variable in the econometric sense — it is a regressor. The paper calls it an "instrument" and "IV" in several places. This terminology is misleading and should be corrected.
- The identifying assumption discussion is too brief for the account-level DiD. See Major Issue 3.

### 6. Results
The account-level results are striking but see Major Issues 1-4 for interpretation concerns. The country-level null is well-characterised. The robustness checks (placebo, leave-one-out, threshold sensitivity) are appropriate additions. The power analysis honestly acknowledges the sample limitations. The heterogeneity analysis appropriately hedges its descriptive findings.

### 7. Discussion
Section 7.1 (reconciling account vs. country results) is well-argued and honest. Section 7.2 (classifier as measurement tool) is the paper's most compelling contribution claim. Section 7.3 (behavioural signature interpretation) is thoughtful. Section 7.4 (limitations) is admirably honest — the selection bias, temporal confound, and panel thinness are all acknowledged.

### 8. Conclusion
Appropriately scoped. The two-contribution framing is effective. The suggestion for future work (5,000+ users per window, 2024-2026 panel) is concrete and actionable. Does not overclaim beyond the evidence.

### 9. Code and Data Availability
This section is present and reasonably detailed. Minor issue: the repository URL uses "andreasclaw" but the git user is "AndreasThinks" — verify this is correct. The reproduction instructions are minimal but functional.

---

## Methodological Assessment

### Classifier methodology
- **Strengths**: Clean separation of label-defining features from classifier features; cross-tool validation is a genuinely strong design element; ablation addresses the writing-style confound convincingly.
- **Concerns**: (a) Label leakage risk — the co-author trailer is used both to define the label and as the timestamp for the pre/post split; if any temporal features are computed relative to this timestamp, there is indirect leakage. (b) The 33/202 class imbalance is not addressed — was any oversampling, undersampling, or cost-sensitive learning used? (c) Cross-validation with 33 positives means some CV folds may have very few positive examples, inflating variance in AUC estimates (the ±0.054 SD suggests this).

### Account-level DiD
- **Design**: Standard first-differences with regression adjustment. Appropriate for the setting but see Major Issues 3 and 4.
- **Parallel trends**: Actively implausible given documented pre-period differences. The paper should either provide pre-trend evidence or reframe as a descriptive comparison.
- **Standard errors**: HC3 is appropriate for heteroskedasticity but does not address potential clustering (e.g., if multiple accounts are from the same organisation).

### Country-level panel regression
- **Specification**: PanelOLS with two-way fixed effects is standard. Clustered SEs at the country level are appropriate.
- **Interpretation**: The null result is honestly interpreted. The power analysis is a welcome addition.
- **Concern**: With 20 countries and 3 time periods (59 observations), the number of country fixed effects (20) consumes a large fraction of degrees of freedom. The effective sample size for identifying the coefficient is very small.
- **Concern**: The adoption variable is zero for all countries before 2024, which means the coefficient is identified entirely from cross-sectional variation in 2024 (not from within-country temporal variation). This makes it a cross-sectional regression dressed up as a panel, and the country fixed effects are doing nothing useful for identification.

### Robustness checks
The placebo tests, leave-one-out, and threshold sensitivity are appropriate. However:
- No event-study plot for the account-level DiD (showing treatment effects by time relative to adoption) — this is standard in DiD papers and its absence is notable.
- No Bonferroni or FDR correction for the multiple outcome tests in the account-level DiD (7 outcomes tested).

---

## Literature Review Assessment

### Accuracy
The literature review accurately represents the cited papers. The discussion of Demirer et al. (2025) correctly distinguishes it from Brynjolfsson et al. (2025) — a correction the author explicitly made (per the bib comments). The METR (2025) finding of 19% *slower* completion is correctly reported and contextualised.

### Missing papers
- **Bird et al. (2023)** — the Microsoft Copilot survey study — is in the bib but not cited. Given it is one of the earliest and most widely cited studies of Copilot's effects, this is a notable omission.
- **Katz et al. (2024)** — also in the bib, also not cited. Worth including or explicitly justifying exclusion.
- **Ziegler et al. (2022)** — "Productivity Assessment of Neural Code Completion" (GitHub internal study) — not cited and relevant.
- **Kalliamvakou (2022)** — GitHub's research blog on Copilot productivity — relevant for the survey evidence stream.
- No citation of the **GitClear** annual reports (2023, 2024) on code churn and AI-generated code quality, despite the root-level `literature_review.md` mentioning them.

### Positioning
The paper's contribution positioning in Section 2.4 is compelling: the measurement gap (no non-survey, non-telemetry detection), the cross-tool generalisation gap, and the aggregation gap are all genuine. This is the paper's strongest selling point.

---

## Writing Quality

### Overall
The prose is above average for a working paper in this field. Sentences are generally clear, the structure is logical, and the author avoids both excessive hedging and overclaiming (with exceptions noted above). The literature review section is particularly well-written.

### Specific issues
- The word "productivity" is used loosely throughout. Define it precisely in the introduction and use it consistently.
- "Inter-commit hours" is defined implicitly but never formally. Add a one-line definition in Section 4.2 or 3.
- The transition from Section 6.4 (country-level results) to Section 6.5 (robustness) is abrupt. A bridging sentence would help.
- Several sections use "we discuss this in Section X" — avoid forward references where possible; they interrupt the reader's flow.
- The phrase "behavioural classifier" is used throughout but a reader unfamiliar with the term may wonder what "behavioural" adds. Consider "commit-behaviour classifier" or similar.

### Consistency
- British spelling is used throughout (generalises, behavioural, organisation) — consistent and appropriate.
- The paper alternates between "AI coding tools," "AI coding assistants," and "AI tools" — pick one and use it throughout.

---

## Publication Readiness Checklist

| Item | Status | Notes |
|------|--------|-------|
| Abstract accurately reflects findings | **Needs Work** | Overclaims "productivity"; should note N=33 treated |
| Introduction motivates the research question | **Ready** | Clear, well-structured |
| Literature review is comprehensive and accurate | **Needs Work** | Missing Bird et al., Katz et al.; GitClear reports |
| Data section is reproducible and transparent | **Needs Work** | Panel thinness not disclosed here; no bot filtering discussion |
| Methodology is clearly described and defensible | **Needs Work** | "IV" terminology misused; classifier-DiD circularity not addressed |
| Results are correctly interpreted | **Needs Work** | Commits ≠ productivity; placebo test interpretation questionable |
| Limitations are honestly disclosed | **Ready** | Admirably thorough |
| Conclusion doesn't overclaim | **Ready** | Appropriately scoped |
| References are complete and correctly formatted | **Blocking** | @anthropic missing from bib; reference list incomplete; @citation syntax won't render |
| Figures and tables are clear and labelled | **Needs Work** | No summary statistics table visible; figures referenced but rendering uncertain |
| Section numbering and structure is consistent | **Needs Work** | Results section too long; consider restructuring |

---

## Final Recommendation

**Major Revisions**

The paper has a genuinely novel contribution in the behavioural classifier, which is well-validated and could stand alone as a measurement paper. The account-level productivity analysis, however, suffers from (a) conflation of commit frequency with productivity, (b) a very small and non-representative treated sample, (c) a mechanical relationship between the classifier features and the DiD outcomes, and (d) implausible parallel trends. These issues do not require new data collection — they require reframing, additional robustness checks (outlier sensitivity, matched comparisons, multiple testing correction), and more honest language about what the estimates measure. The country-level analysis adds little in its current form but is worth retaining as an honest null. The reference rendering pipeline needs to be fixed before any distribution. With these revisions, the paper would be suitable for a computational social science or empirical software engineering venue.
