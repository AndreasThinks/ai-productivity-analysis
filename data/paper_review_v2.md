# Follow-Up Review: "Detecting AI Coding Tool Adoption and Its Behavioural Effects on Developer Commit Activity"

**Manuscript**: Working Paper v2, April 2026
**Author**: Andreas Varotsis
**Reviewer**: Anonymous (Senior Academic Reviewer)
**Date**: 7 April 2026
**Review type**: Revision assessment (following Major Revisions verdict)

---

## Response to Revisions

### Issue 1: Commit frequency ≠ productivity — paper overclaimed "productivity"

**Fully addressed.** The title has been changed from "Productivity Effects of AI Coding Tool Adoption" to "Behavioural Effects on Developer Commit Activity." The abstract now consistently refers to "behavioural changes" and "commit behaviour" rather than "productivity." Section 7.4 includes an explicit limitation paragraph: "Commit behaviour is not productivity. This paper measures changes in commit frequency, timing, and documentation patterns — not direct output measures such as features shipped, issues resolved, or code quality." This is exactly the reframing requested. The language is now appropriately scoped throughout.

### Issue 2: N=33 treated sample — outlier sensitivity needed

**Partially addressed.** The revision adds a winsorised DiD (Table 4b, Cell 23) clipping outcomes and pre-period controls at the 5th/95th percentiles — a direct response to the request. The high-confidence subsample robustness check (N=25) was already present and is retained. However, two of the three requested fixes are missing: (a) median effects alongside means are not reported, and (b) no "drop the top K treated accounts" sensitivity is shown. The abstract and conclusion still present the 13.1 commits/week effect without qualifying it as an upper bound for extreme early adopters. The conclusion states "roughly 13 additional commits per active week" without caveat.

### Issue 3: Severe selection bias — parallel trends implausible

**Partially addressed.** A new "Parallel Trends Assumption" subsection appears in Section 6.5, which is welcome. However, this subsection primarily discusses the country-level design (Regression A's pre-treatment trends test, p=0.34), not the account-level DiD where the concern is most acute. The account-level parallel trends problem is acknowledged in Section 7.4 ("AI adopters show significantly higher pre-period activity... The regression adjustment controls for pre-period levels but cannot fully eliminate selection bias"), which is largely unchanged from v1. No propensity score matching or pre-trend visualisation has been added. The treatment effect is still presented as a point estimate rather than framed as an upper bound on the ATT. The revision acknowledges the problem more explicitly but does not materially strengthen the identification.

### Issue 4: Classifier trained on same features used as DiD outcomes — circularity

**Not addressed.** The revision does not discuss the mechanical relationship between classifier features and DiD outcomes. The classifier is trained to discriminate AI adopters using, among other things, changes in commits per active week and inter-commit hours — the same variables that appear as primary DiD outcomes. Accounts classified as treated are partly those with the largest behavioural shifts, so finding large DiD coefficients is partly tautological. This concern is not mentioned in the limitations section or anywhere else in the revised manuscript. This remains a significant interpretive weakness.

### Issue 5: "We" voice / AI assistance not disclosed

**Fully addressed.** An Acknowledgements section (Cell 30) now states: "The author thanks the open-source developer community... This paper was written with the assistance of Claude (Anthropic), which was used for literature review drafting, code review, and editing. All analytical decisions, interpretations, and errors are the author's own." This is transparent and appropriate. The "we" convention is retained throughout, which is acceptable for a working paper in the economics tradition, especially now that AI assistance is disclosed.

### Issue 6: Missing citations (Bird et al., Katz et al.); broken @anthropic reference

**Fully addressed.** Bird et al. (2023) is now cited in Section 2.1 with a substantive discussion of their survey findings (2,000+ Microsoft developers, 88% reported increased productivity). Katz et al. (2024) is also cited in Section 2.1 and included in the Gaps and Contributions subsection (2.4). Both appear in the References section (Cell 31). The broken `@anthropic` reference has been replaced with inline text ("Anthropic's Claude Code") in the introduction. The reference list now includes 5 entries rendered in full, with the remaining citations handled through the Quarto/BibTeX pipeline via the raw YAML frontmatter (Cell 1).

### Issue 7: "IV" terminology misused

**Fully addressed.** The term "instrumental variable" / "IV" no longer appears in the revised manuscript. Section 5.2 now describes Regression C simply as "Per-country classifier scores from population sample (primary)" without invoking IV terminology. The description of the adoption variable as a regressor is now accurate.

### Issue 8: Placebo test interpretation confused

**Fully addressed.** The placebo test has been completely rewritten with a proper simulation (1,000 permutations of country-level adoption rates). The revised version reports the full placebo distribution (mean −0.13, SD 5.37, 90% interval [−8.76, 8.70]), correctly notes the observed coefficient (−6.06) falls within this interval, and computes a two-tailed permutation p-value (0.269). The interpretation is now correct: "This result confirms the null: the observed negative coefficient is fully consistent with sampling variability." The previous error — comparing a negative coefficient to the upper tail — is eliminated.

### Issue 9: Median-2-developers stat buried in Discussion

**Fully addressed.** Section 3.4 ("Summary Statistics") now explicitly states: "For the country-level commit activity panel, the median number of located developers per country-year observation is 2 — a level of thinness that substantially limits the power of the country-level regression (discussed further in Section 6.6)." This is prominently placed in the Data section where it belongs, with a forward reference to the power analysis. Section 7.1 retains the discussion of its implications, which is appropriate.

### Issue 10: No FDR correction for multiple outcome tests

**Fully addressed.** Cell 16 now implements Benjamini-Hochberg FDR correction across the 7 simultaneous outcome tests, using `statsmodels.stats.multitest.multipletests`. Table 4 reports both raw p-values and FDR-corrected q-values. The narrative (Cell 18) correctly notes: "The two primary outcomes (commits per active week, inter-commit hours) and active weeks remain significant after FDR correction; secondary outcomes should be interpreted with appropriate caution given the multiple comparisons." This is the right approach and the right interpretation.

---

## Remaining Concerns

### R1. Classifier–DiD circularity still unaddressed (Issue 4)

This is the most significant unresolved problem. The classifier selects treated accounts partly based on the magnitude of their behavioural shifts in commits per active week and inter-commit hours — the same variables that serve as primary DiD outcomes. The DiD therefore partly recovers the classifier's own discriminative signal. The paper needs either: (a) an explicit paragraph in the limitations explaining why the DiD adds information beyond the classifier's discrimination, or (b) DiD outcomes that are not classifier features (issues closed, PR merge latency, release frequency), or at minimum (c) a demonstration that the DiD coefficients are meaningfully different from what a mechanical back-calculation from the classifier's feature importances would predict.

### R2. Treatment effects not framed as upper bounds

The abstract reports "AI adopters increase commits per active week by 13.1 (p < 0.001)" and the conclusion says "roughly 13 additional commits per active week" — both without qualification. Given that (a) treated accounts are extreme early adopters identified via public markers, (b) selection bias is acknowledged but unresolved, and (c) the classifier-DiD circularity inflates the estimates mechanically, these effects should be explicitly framed as upper bounds. A single sentence in the abstract ("likely an upper bound given the selection of early, intensive adopters") would suffice.

### R3. Winsorisation results not reported in the narrative

Cell 23 computes winsorised DiD estimates and Cell 24 provides a brief narrative, but the actual winsorised coefficients are not discussed — only whether significance changed. The reader needs to see whether the point estimates attenuate substantially (e.g., does 13.1 become 8.0 or 12.5?). If winsorisation barely moves the coefficients, say so explicitly — that is the strongest possible robustness result. If it halves them, that is also important.

### R4. Cross-tool validation language (p = 0.065)

Cell 11 still states Aider users' scores are "statistically indistinguishable from the Claude training positives (p = 0.065)." This language was flagged in the original review (Minor Issue 7) and has not been changed. At α = 0.05, p = 0.065 does not support a claim of indistinguishability. Rephrase to "not significantly different at the 5% level" or "marginally different."

### R5. The 859 vs 235 sample relationship remains unclear

The heterogeneity analysis (Section 6.7) references "859 classified accounts" with "approximately 280 accounts in each tercile," while the DiD uses 235. The population scoring sample is described as 887 in Section 3.1. The relationship between these overlapping samples (235 labelled, 887 scored, 859 classified) needs a clear sentence, ideally in Section 3 or a data flow diagram. A reader encountering "859" for the first time in Section 6.7 will be confused.

### R6. Regression B still included

The original review (Minor Issue 5) flagged that Regression B (time proxy) produces a degenerate result due to perfect collinearity, and suggested dropping it or explaining upfront why it is included. The revision retains it with the same framing: "included for reference... as expected." This is minor but still adds noise to the results section without pedagogical value.

---

## New Strengths

1. **The reframing works.** The title change and consistent use of "commit behaviour" / "behavioural effects" throughout the paper is a genuine improvement. The paper now makes a defensible claim rather than an overclaimed one. The limitations paragraph on "commit behaviour is not productivity" (Section 7.4) is one of the most honest methodological caveats I have seen in a working paper in this space.

2. **FDR correction is well-implemented.** The Benjamini-Hochberg correction with both raw and corrected values in the same table is best practice. The narrative correctly distinguishes primary outcomes that survive correction from secondary ones that do not.

3. **The placebo test is now a real contribution.** The 1,000-permutation test with a proper two-tailed interpretation is methodologically sound and honestly interpreted. It strengthens the null result by showing the observed coefficient is indistinguishable from noise — a much more informative statement than the original confused comparison.

4. **Bird et al. and Katz et al. integration.** These are not just cited but substantively discussed, with Bird et al.'s survey findings and Katz et al.'s adoption curves placed in the appropriate literature stream. The Gaps and Contributions section (2.4) now reads as comprehensive.

5. **Panel thinness disclosed early.** Moving the median-2-developers statistic to Section 3.4 is a small change that substantially improves the paper's honesty. A reader now knows, before encountering any country-level results, that the panel is extremely thin.

6. **Acknowledgements section.** The AI assistance disclosure is appropriate in both content and tone. For a paper about AI coding tools, this transparency is particularly important.

---

## Publication Readiness

**Verdict: Yes, with minor fixes.**

The following must be done before posting as a preprint:

1. **Add a paragraph on classifier–DiD circularity** to Section 7.4 (Limitations). Explain that the classifier is trained on features that overlap with the DiD outcomes, that this creates a mechanical upward bias in the treatment effect estimates, and that the DiD should be interpreted as evidence of the *magnitude* of behavioural change conditional on the classifier having already identified these accounts as adopters — not as an independent causal estimate. (~100 words.)

2. **Frame the account-level effects as upper bounds** in the abstract and conclusion. One sentence each: "These estimates likely represent an upper bound, as the treated sample consists of early, intensive adopters identified via public markers."

3. **Report the winsorised point estimates** in the Cell 24 narrative (not just significance retention). State whether the coefficients attenuate by <10%, 10–30%, or >30%.

4. **Fix the "statistically indistinguishable" language** in Cell 11. Replace with "not significantly different at the 5% level (p = 0.065)."

5. **Add one sentence** in Section 3 or 6.7 clarifying the relationship between the 235 labelled, 887 scored, and 859 classified samples.

---

## Bottom Line

The revision has addressed the majority of the original concerns substantively and in good faith. The reframing from "productivity" to "commit behaviour" resolves the paper's central interpretive weakness; the FDR correction, winsorisation, and rewritten placebo test strengthen the empirical analysis; and the literature review is now comprehensive. The one significant gap — the classifier–DiD circularity — is a real methodological issue but is addressable with a paragraph of honest discussion rather than new analysis, because the paper's primary contribution is the classifier, not the DiD. The account-level effects should be framed as upper bounds, which costs the paper nothing in terms of contribution but substantially improves its credibility. With the five fixes listed above — all achievable in a single editing pass — the paper is ready for preprint distribution. The classifier methodology, cross-tool validation, and honest null result at the country level make genuine contributions to the empirical AI-and-software-development literature.
