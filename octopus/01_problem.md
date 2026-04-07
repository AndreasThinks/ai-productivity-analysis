# Octopus Publication: Research Problem

**Title:** How can AI coding tool adoption be measured at scale from public developer behaviour, and does adoption affect commit activity?

**Keywords:** AI coding tools, GitHub Copilot, Claude Code, developer behaviour, measurement, software engineering, adoption, commit activity

---

## The Problem

The rapid diffusion of AI coding assistants since late 2022 — including GitHub Copilot, Anthropic's Claude Code, and Aider — has generated widespread interest in their effects on software developer behaviour and output. Understanding these effects at scale is important for researchers, policymakers, and organisations evaluating AI investment. However, the problem poses a fundamental measurement challenge with two components.

**Component 1: Adoption is largely invisible in public data.** Most AI coding tool usage leaves no trace in public repositories or commit histories. Developers who use AI assistance rarely announce it. The ground truth — who is using which tools, and when they started — is typically locked inside proprietary vendor telemetry or requires expensive survey collection. This makes it difficult to estimate effects on any population larger than those participating in controlled experiments.

**Component 2: Aggregate effects are unknown.** The existing literature consists almost entirely of firm-level or laboratory studies with narrow samples. We do not know whether individual-level productivity changes — if they exist — aggregate to detectable effects at the level of countries or regions, or whether national AI adoption rates predict national-level changes in developer commit activity. This is a distinct question from the individual-level one, and answering it requires both a measure of adoption at country scale and a measure of productivity at country scale.

These two components define an open research problem: can AI coding tool adoption be detected and measured from publicly observable developer behaviour, without relying on self-reported adoption, proprietary telemetry, or tool-specific artefacts? And if so, what do the resulting adoption measures reveal about the effects of AI tools on developer commit activity at both the individual and country level?

---

*This problem statement links to the Methods publication, which describes the classifier and causal designs used to address it.*
