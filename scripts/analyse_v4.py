#!/usr/bin/env python
"""
v4 analysis: features, event-time DiD and classifier from the v4 caches.

Reads the private caches written by scripts/rescrape_v4.py and writes only
pseudonymised outputs to data/:

  data/v4_account_features.csv   one row per account, pre/post/delta features
  data/v4_did_results.csv        DiD estimates, all commits and human-only
  data/v4_classifier_results.csv cross-validated AUCs and ablations
  data/v4_results.md             plain-language summary of the above

Every tempo outcome is computed twice: on all commits, and on commits with no
agent marker ("human-only"). If an effect is there for all commits but not for
human-only ones, it is agent auto-commits, not a change in how people work.

  uv run scripts/analyse_v4.py
"""

from __future__ import annotations

import json
import warnings
import re
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests

try:
    from scripts.paths import DATA_DIR, PRIVATE_DIR
    from scripts.pseudonymise import load_salt, pseudonym
    from scripts.rescrape_v4 import CACHE_DIR, WINDOW_DAYS, detect_agent, event_windows, parse_ts, strip_agent_markers
except ImportError:
    from paths import DATA_DIR, PRIVATE_DIR
    from pseudonymise import load_salt, pseudonym
    from rescrape_v4 import CACHE_DIR, WINDOW_DAYS, detect_agent, event_windows, parse_ts, strip_agent_markers

WEEKS = WINDOW_DAYS / 7
CONVENTIONAL_RE = re.compile(r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?!?:", re.IGNORECASE)
TEST_RE = re.compile(r"\btests?\b", re.IGNORECASE)

ACTIVITY = ["commits_per_week", "active_weeks", "active_days", "commits_per_active_day",
            "median_gap_hours", "frac_burst", "repos_touched"]
STYLE = ["mean_msg_len", "frac_multiline", "frac_conventional", "frac_mentions_test", "frac_bullets"]
PR = ["prs_opened", "prs_merged", "prs_merged_external"]


def window_features(commits: Iterable[dict[str, Any]]) -> dict[str, float]:
    """Activity and message-style features for one window of commits.

    Windows are the same length for every account, so counts are comparable
    without per-active-week normalisation; commits_per_week divides by the
    calendar length of the window.
    """
    rows = []
    for c in commits:
        ts = parse_ts(c.get("authored_at") or c.get("committed_at"))
        if ts is None or c.get("n_parents", 1) > 1:  # merge commits carry no authored work
            continue
        rows.append((ts, strip_agent_markers(c.get("message", "")), c.get("repo")))
    rows.sort(key=lambda r: r[0])
    n = len(rows)
    if n == 0:
        return {k: 0.0 for k in ACTIVITY + STYLE} | {"n_commits": 0.0, "median_gap_hours": np.nan}
    gaps = [(b[0] - a[0]).total_seconds() / 3600 for a, b in zip(rows, rows[1:])]
    days = {r[0].date() for r in rows}
    weeks = {r[0].isocalendar()[:2] for r in rows}
    msgs = [r[1] for r in rows]
    return {
        "n_commits": float(n),
        "commits_per_week": n / WEEKS,
        "active_weeks": float(len(weeks)),
        "active_days": float(len(days)),
        "commits_per_active_day": n / len(days),
        "median_gap_hours": float(median(gaps)) if gaps else np.nan,
        "frac_burst": float(np.mean([g <= 1.0 for g in gaps])) if gaps else 0.0,
        "repos_touched": float(len({r[2] for r in rows})),
        "mean_msg_len": float(np.mean([len(m) for m in msgs])),
        "frac_multiline": float(np.mean(["\n" in m for m in msgs])),
        "frac_conventional": float(np.mean([bool(CONVENTIONAL_RE.match(m)) for m in msgs])),
        "frac_mentions_test": float(np.mean([bool(TEST_RE.search(m)) for m in msgs])),
        "frac_bullets": float(np.mean([("\n- " in m or "\n* " in m) for m in msgs])),
    }


def account_row(cache: dict[str, Any], salt: str) -> dict[str, Any]:
    w = event_windows(parse_ts(cache["event_date"]))
    split = {"pre": [], "post": []}
    for c in cache["commits"]:
        side = w.label(parse_ts(c.get("authored_at") or c.get("committed_at")))
        if side:
            split[side].append(c)
    row: dict[str, Any] = {
        "account_id": pseudonym(cache["login"], salt),
        "group": cache["group"],
        "tool": cache["tool"],
        "treated": int(cache["group"] == "adopter"),
        "event_month": w.event.strftime("%Y-%m"),
        "commits_truncated": bool(cache.get("commits_truncated")),
    }
    for side, commits in split.items():
        agent_flags = [detect_agent(c) for c in commits]
        row[f"{side}_agent_share"] = float(np.mean([a is not None for a in agent_flags])) if commits else 0.0
        for k, v in window_features(commits).items():
            row[f"{side}_{k}"] = v
        human = [c for c, a in zip(commits, agent_flags) if a is None]
        for k, v in window_features(human).items():
            row[f"{side}_human_{k}"] = v
        for k, v in cache[f"prs_{side}"].items():
            row[f"{side}_{k}"] = float(v)
    for stem in ACTIVITY + STYLE + PR + ["n_commits"] + [f"human_{s}" for s in ACTIVITY + ["n_commits"]]:
        row[f"delta_{stem}"] = row[f"post_{stem}"] - row[f"pre_{stem}"]
    return row


def build_features(cache_dir: Path = CACHE_DIR, salt: str | None = None, min_pre_commits: int = 5) -> pd.DataFrame:
    salt = salt or load_salt()
    rows = [account_row(json.loads(p.read_text()), salt) for p in sorted(cache_dir.glob("*.json"))]
    df = pd.DataFrame(rows)
    # Same activity floor for both groups, on the pre-period only, so the
    # filter cannot condition on the outcome.
    return df[df["pre_n_commits"] >= min_pre_commits].reset_index(drop=True)


def did(df: pd.DataFrame, stems: list[str], label: str) -> pd.DataFrame:
    """ΔY = α + β·treated + γ·Y_pre, HC3 errors, BH q-values within the family."""
    out = []
    for stem in stems:
        d = df[[f"delta_{stem}", f"pre_{stem}", "treated"]].dropna()
        d.columns = ["y", "pre", "treated"]
        if d["treated"].sum() < 5 or d["y"].std() == 0 or d["pre"].std() == 0:
            continue
        m = smf.ols("y ~ treated + pre", data=d).fit(cov_type="HC3")
        out.append({"sample": label, "outcome": stem, "n": len(d), "n_treated": int(d["treated"].sum()),
                    "coef": m.params["treated"], "se": m.bse["treated"], "p": m.pvalues["treated"]})
    res = pd.DataFrame(out)
    if len(res):
        res["q_bh"] = multipletests(res["p"], method="fdr_bh")[1]
    return res


def classifier_results(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-validated AUC for Claude adopters vs controls, with ablations.

    The Aider adopters are never trained on; their mean score is the
    cross-tool check.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    sets = {
        "all features": [f"{s}_{f}" for s in ("pre", "post", "delta") for f in ACTIVITY + STYLE],
        "activity only": [f"{s}_{f}" for s in ("pre", "post", "delta") for f in ACTIVITY],
        "style only": [f"{s}_{f}" for s in ("pre", "post", "delta") for f in STYLE],
        "human-only activity": [f"{s}_human_{f}" for s in ("pre", "post", "delta") for f in ACTIVITY],
        "pre-period only (placebo)": [f"pre_{f}" for f in ACTIVITY + STYLE],
    }
    train = df[(df["group"] == "control") | (df["tool"] == "claude")]
    aider = df[(df["group"] == "adopter") & (df["tool"] == "aider")]
    y = train["treated"].values
    rows = []
    for name, cols in sets.items():
        model = make_pipeline(SimpleImputer(strategy="median"),
                              RandomForestClassifier(n_estimators=300, max_depth=5, random_state=42))
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        prob = cross_val_predict(model, train[cols], y, cv=cv, method="predict_proba")[:, 1]
        model.fit(train[cols], y)
        rows.append({
            "feature_set": name, "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()),
            "cv_auc": roc_auc_score(y, prob),
            "aider_mean_score": float(model.predict_proba(aider[cols])[:, 1].mean()) if len(aider) else np.nan,
            "control_mean_score": float(prob[y == 0].mean()),
        })
    return pd.DataFrame(rows)


def write_summary(feats: pd.DataFrame, did_res: pd.DataFrame, clf: pd.DataFrame, path: Path) -> None:
    def table(d: pd.DataFrame) -> str:
        return d.to_markdown(index=False, floatfmt=".3f") if len(d) else "_none_"

    lines = [
        "# v4 results: equal event-time windows",
        "",
        f"Accounts after the pre-period activity floor: {len(feats)} "
        f"({int(feats.treated.sum())} adopters, {int((1 - feats.treated).sum())} controls). "
        f"Window: {WINDOW_DAYS} days either side of each account's date.",
        "",
        f"Agent share of post-window commits, adopters: {feats.loc[feats.treated == 1, 'post_agent_share'].mean():.2f}; "
        f"controls: {feats.loc[feats.treated == 0, 'post_agent_share'].mean():.2f}.",
        "",
        "## Difference-in-differences",
        "",
        table(did_res),
        "",
        "## Classifier",
        "",
        table(clf),
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    feats = build_features()
    feats.to_csv(DATA_DIR / "v4_account_features.csv", index=False)
    claude = feats[(feats.group == "control") | (feats.tool == "claude")]
    aider = feats[(feats.group == "control") | (feats.tool == "aider")]
    did_res = pd.concat([
        did(claude, ACTIVITY + STYLE + PR, "claude: all commits"),
        did(claude, [f"human_{s}" for s in ACTIVITY], "claude: human-only commits"),
        did(aider, ACTIVITY + STYLE + PR, "aider: all commits"),
        did(aider, [f"human_{s}" for s in ACTIVITY], "aider: human-only commits"),
    ], ignore_index=True)
    did_res.to_csv(DATA_DIR / "v4_did_results.csv", index=False)
    clf = classifier_results(feats)
    clf.to_csv(DATA_DIR / "v4_classifier_results.csv", index=False)
    write_summary(feats, did_res, clf, DATA_DIR / "v4_results.md")
    print((DATA_DIR / "v4_results.md").read_text())


if __name__ == "__main__":
    main()
