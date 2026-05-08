#!/usr/bin/env python
"""
Scrape and analyse account-level pull-request outcomes for the AI productivity
project.

This extends the classifier/account-level DiD with accepted-output metrics:
opened PRs, merged PRs, merge rate, time to merge, PR size, and review burden.
It uses GitHub issue search (type:pr author:<login>) so authored PRs are found
across repositories, not only repos owned by the account.

Run scrape + analysis:
  GITHUB_TOKEN=... uv run --with pandas --with statsmodels \
    scripts/pr_outcome_metrics.py --scrape --analyse

Run analysis only from existing PR cache:
  uv run --with pandas --with statsmodels scripts/pr_outcome_metrics.py --analyse
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any, Iterable

import pandas as pd
import statsmodels.formula.api as smf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
FEATURES_PATH = DATA_DIR / "classifier_full_features.csv"
CACHE_DIR = DATA_DIR / "pr_outcome_cache"
OUTCOMES_PATH = DATA_DIR / "account_pr_outcomes.csv"
DID_RESULTS_PATH = DATA_DIR / "account_pr_did_results.txt"
STATUS_PATH = DATA_DIR / "pr_outcome_status.csv"

PRE_START = datetime(2022, 1, 1, tzinfo=timezone.utc)
POST_START = datetime(2024, 1, 1, tzinfo=timezone.utc)
PRE_MONTHS = 24.0
DEFAULT_MAX_PRS_PER_ACCOUNT = 300
REQUEST_TIMEOUT = 30
API_DELAY = 1.0

# Metrics for which build_did_row creates delta_* columns. Keep these focused on
# outcomes, not raw identifiers.
PR_METRIC_STEMS = [
    "prs_opened",
    "prs_merged",
    "prs_closed_unmerged",
    "opened_prs_per_month",
    "merged_prs_per_month",
    "merge_rate",
    "median_hours_to_merge",
    "mean_hours_to_merge",
    "mean_additions",
    "mean_deletions",
    "mean_lines_changed",
    "mean_changed_files",
    "mean_comments",
    "mean_review_comments",
    "mean_commits_per_pr",
]

DID_METRICS = [
    "prs_opened",
    "prs_merged",
    "opened_prs_per_month",
    "merged_prs_per_month",
    "merge_rate",
    "median_hours_to_merge",
    "mean_lines_changed",
    "mean_review_comments",
    "mean_commits_per_pr",
    "prs_closed_unmerged",
]


def parse_ts(value: str | None) -> datetime | None:
    """Parse GitHub ISO timestamps as timezone-aware UTC datetimes."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def month_span(start: datetime, end: datetime) -> float:
    """Approximate calendar months between two datetimes, bounded above zero."""
    months = (end.year - start.year) * 12 + (end.month - start.month)
    day_fraction = (end.day - start.day) / 30.4375
    second_fraction = (
        (end.hour - start.hour) * 3600
        + (end.minute - start.minute) * 60
        + (end.second - start.second)
    ) / (30.4375 * 24 * 60 * 60)
    return max(months + day_fraction + second_fraction, 1 / 30.4375)


def window_for_timestamp(value: str | None) -> str | None:
    """Return 'pre', 'post', or None for a PR created_at timestamp."""
    ts = parse_ts(value)
    if ts is None:
        return None
    if PRE_START <= ts < POST_START:
        return "pre"
    if ts >= POST_START:
        return "post"
    return None


def _safe_number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        if isinstance(value, float) and math.isnan(value):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def _median(values: Iterable[float]) -> float:
    vals = list(values)
    return float(median(vals)) if vals else 0.0


def _summarize_window(prs: list[dict[str, Any]], months: float) -> dict[str, float]:
    opened = len(prs)
    merged = [pr for pr in prs if pr.get("merged_at")]
    closed_unmerged = [
        pr for pr in prs
        if pr.get("closed_at") and not pr.get("merged_at")
    ]

    merge_hours = []
    for pr in merged:
        created = parse_ts(pr.get("created_at"))
        merged_at = parse_ts(pr.get("merged_at"))
        if created and merged_at and merged_at >= created:
            merge_hours.append((merged_at - created).total_seconds() / 3600.0)

    additions = [_safe_number(pr.get("additions")) for pr in prs]
    deletions = [_safe_number(pr.get("deletions")) for pr in prs]
    lines_changed = [a + d for a, d in zip(additions, deletions)]

    return {
        "prs_opened": float(opened),
        "prs_merged": float(len(merged)),
        "prs_closed_unmerged": float(len(closed_unmerged)),
        "opened_prs_per_month": opened / months if months else 0.0,
        "merged_prs_per_month": len(merged) / months if months else 0.0,
        "merge_rate": len(merged) / opened if opened else 0.0,
        "median_hours_to_merge": _median(merge_hours),
        "mean_hours_to_merge": _mean(merge_hours),
        "mean_additions": _mean(additions),
        "mean_deletions": _mean(deletions),
        "mean_lines_changed": _mean(lines_changed),
        "mean_changed_files": _mean(_safe_number(pr.get("changed_files")) for pr in prs),
        "mean_comments": _mean(_safe_number(pr.get("comments")) for pr in prs),
        "mean_review_comments": _mean(_safe_number(pr.get("review_comments")) for pr in prs),
        "mean_commits_per_pr": _mean(_safe_number(pr.get("commits")) for pr in prs),
    }


def summarize_pr_outcomes(
    prs: list[dict[str, Any]],
    *,
    post_window_end: datetime | None = None,
) -> dict[str, float]:
    """Compute pre/post PR outcome metrics from normalised PR dictionaries."""
    post_window_end = post_window_end or datetime.now(timezone.utc)
    post_months = month_span(POST_START, post_window_end)

    buckets = {"pre": [], "post": []}
    for pr in prs:
        window = window_for_timestamp(pr.get("created_at"))
        if window in buckets:
            buckets[window].append(pr)

    out: dict[str, float] = {}
    for prefix, months in [("pre", PRE_MONTHS), ("post", post_months)]:
        summary = _summarize_window(buckets[prefix], months)
        out.update({f"{prefix}_{key}": round(value, 6) for key, value in summary.items()})
    return out


def build_did_row(feature_row: dict[str, Any] | pd.Series, metrics: dict[str, float]) -> dict[str, Any]:
    """Merge classifier metadata with PR metrics and delta columns."""
    source = dict(feature_row)
    row: dict[str, Any] = {
        "login": source.get("login"),
        "label": int(source.get("label", 0)),
        "discovery_method": source.get("discovery_method", ""),
        "marker_confidence": source.get("marker_confidence", ""),
    }
    row.update(metrics)

    for stem in PR_METRIC_STEMS:
        pre = _safe_number(row.get(f"pre_{stem}"))
        post = _safe_number(row.get(f"post_{stem}"))
        row[f"delta_{stem}"] = round(post - pre, 6)
    return row


def benjamini_hochberg(p_values: Iterable[float]) -> list[float]:
    """Return Benjamini-Hochberg adjusted q-values in original p-value order."""
    indexed = []
    for i, value in enumerate(p_values):
        try:
            p_value = float(value)
        except (TypeError, ValueError):
            p_value = float("nan")
        indexed.append((i, p_value))

    finite = [(i, p) for i, p in indexed if not math.isnan(p)]
    m = len(finite)
    q_values = [float("nan")] * len(indexed)
    if not finite:
        return q_values

    ranked = sorted(finite, key=lambda item: item[1])
    adjusted = []
    for rank, (original_i, p_value) in enumerate(ranked, start=1):
        adjusted.append((original_i, min(p_value * m / rank, 1.0)))

    running_min = 1.0
    for original_i, q_value in reversed(adjusted):
        running_min = min(running_min, q_value)
        q_values[original_i] = round(running_min, 12)
    return q_values


def compute_did_results(df: pd.DataFrame, metrics: list[str] | None = None) -> pd.DataFrame:
    """Run account-level DiD regressions: delta_metric ~ label + pre_metric."""
    metrics = metrics or DID_METRICS
    rows = []
    for metric in metrics:
        pre_col = f"pre_{metric}"
        delta_col = f"delta_{metric}"
        if pre_col not in df.columns or delta_col not in df.columns:
            continue
        subset = df[["label", pre_col, delta_col]].dropna().copy()
        if subset["label"].nunique() < 2 or len(subset) < 4:
            continue
        model = smf.ols(f"{delta_col} ~ label + {pre_col}", data=subset).fit(cov_type="HC3")
        rows.append({
            "metric": metric,
            "n": int(model.nobs),
            "treated_n": int(subset["label"].sum()),
            "control_n": int((subset["label"] == 0).sum()),
            "treatment_coef": float(model.params.get("label", float("nan"))),
            "se": float(model.bse.get("label", float("nan"))),
            "p_value": float(model.pvalues.get("label", float("nan"))),
            "pre_control_mean": float(subset.loc[subset["label"] == 0, pre_col].mean()),
            "pre_treated_mean": float(subset.loc[subset["label"] == 1, pre_col].mean()),
        })
    results = pd.DataFrame(rows)
    if not results.empty:
        results["bh_q_value"] = benjamini_hochberg(results["p_value"].tolist())
    return results


def build_coverage_diagnostics(
    outcomes: pd.DataFrame,
    *,
    max_prs_per_account: int = DEFAULT_MAX_PRS_PER_ACCOUNT,
) -> dict[str, int]:
    """Summarise PR outcome coverage and important caveat counts."""
    if outcomes.empty:
        return {
            "accounts": 0,
            "treated": 0,
            "controls": 0,
            "zero_pr_accounts": 0,
            "zero_pr_treated": 0,
            "zero_pr_controls": 0,
            "capped_accounts": 0,
            "capped_treated": 0,
            "capped_controls": 0,
            "pr_active_accounts": 0,
            "pr_active_treated": 0,
            "pr_active_controls": 0,
            "both_window_pr_active_accounts": 0,
            "both_window_pr_active_treated": 0,
            "both_window_pr_active_controls": 0,
        }

    labels = outcomes["label"].fillna(0).astype(int)
    if "n_prs" in outcomes:
        n_prs = outcomes["n_prs"]
    else:
        n_prs = outcomes.get("pre_prs_opened", 0) + outcomes.get("post_prs_opened", 0)
    zero = n_prs == 0
    capped = n_prs >= max_prs_per_account
    pr_active = n_prs > 0
    both_window = (outcomes.get("pre_prs_opened", 0) > 0) & (outcomes.get("post_prs_opened", 0) > 0)

    def count(mask: pd.Series) -> int:
        return int(mask.sum())

    def treated_count(mask: pd.Series) -> int:
        return int((mask & (labels == 1)).sum())

    def control_count(mask: pd.Series) -> int:
        return int((mask & (labels == 0)).sum())

    return {
        "accounts": int(len(outcomes)),
        "treated": int((labels == 1).sum()),
        "controls": int((labels == 0).sum()),
        "zero_pr_accounts": count(zero),
        "zero_pr_treated": treated_count(zero),
        "zero_pr_controls": control_count(zero),
        "capped_accounts": count(capped),
        "capped_treated": treated_count(capped),
        "capped_controls": control_count(capped),
        "pr_active_accounts": count(pr_active),
        "pr_active_treated": treated_count(pr_active),
        "pr_active_controls": control_count(pr_active),
        "both_window_pr_active_accounts": count(both_window),
        "both_window_pr_active_treated": treated_count(both_window),
        "both_window_pr_active_controls": control_count(both_window),
    }


def build_sensitivity_results(
    outcomes: pd.DataFrame,
    *,
    max_prs_per_account: int = DEFAULT_MAX_PRS_PER_ACCOUNT,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Run focused robustness specifications for the strongest PR outcome metrics."""
    metrics = metrics or ["prs_opened", "prs_merged", "merge_rate", "median_hours_to_merge"]
    if outcomes.empty:
        return pd.DataFrame()
    if "n_prs" in outcomes:
        n_prs = outcomes["n_prs"]
    else:
        n_prs = outcomes.get("pre_prs_opened", 0) + outcomes.get("post_prs_opened", 0)
    marker_confidence = outcomes.get("marker_confidence", pd.Series("", index=outcomes.index)).fillna("")
    specs = {
        "main": outcomes,
        "uncapped_only": outcomes[n_prs < max_prs_per_account],
        "nonzero_prs_only": outcomes[n_prs > 0],
        "both_prepost_activity": outcomes[(outcomes.get("pre_prs_opened", 0) > 0) & (outcomes.get("post_prs_opened", 0) > 0)],
        "drop_zero_pre": outcomes[outcomes.get("pre_prs_opened", 0) > 0],
        "high_conf_treated_only": outcomes[(outcomes["label"] == 0) | ((outcomes["label"] == 1) & (marker_confidence == "high"))],
    }
    rows = []
    for spec_name, spec_df in specs.items():
        if spec_df.empty or spec_df["label"].nunique() < 2:
            continue
        results = compute_did_results(spec_df, metrics)
        for result in results.to_dict("records"):
            result.update({
                "spec": spec_name,
                "capped_n": int((spec_df["n_prs"] >= max_prs_per_account).sum()) if "n_prs" in spec_df else 0,
            })
            rows.append(result)
    return pd.DataFrame(rows)


class GitHubClient:
    def __init__(self, token: str | None, *, delay: float = API_DELAY):
        self.token = token
        self.delay = delay

    def get(self, url: str) -> Any:
        headers = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "ai-productivity-pr-outcome-scraper",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        req = urllib.request.Request(url, headers=headers)
        for attempt in range(5):
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    time.sleep(self.delay)
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    reset = e.headers.get("X-RateLimit-Reset")
                    if reset and reset.isdigit():
                        wait = max(int(reset) - int(time.time()) + 5, 30)
                    else:
                        wait = 60 * (attempt + 1)
                    print(f"Rate limited on {url}; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if e.code == 404:
                    return None
                raise
            except urllib.error.URLError:
                time.sleep(10 * (attempt + 1))
        return None


def normalise_pr(search_item: dict[str, Any], detail: dict[str, Any] | None) -> dict[str, Any]:
    """Flatten GitHub search + PR detail payloads to the fields used here."""
    detail = detail or {}
    return {
        "html_url": search_item.get("html_url"),
        "repository_url": search_item.get("repository_url"),
        "number": search_item.get("number"),
        "title": search_item.get("title", "")[:200],
        "created_at": detail.get("created_at") or search_item.get("created_at"),
        "updated_at": detail.get("updated_at") or search_item.get("updated_at"),
        "closed_at": detail.get("closed_at") or search_item.get("closed_at"),
        "merged_at": detail.get("merged_at"),
        "state": detail.get("state") or search_item.get("state"),
        "additions": detail.get("additions"),
        "deletions": detail.get("deletions"),
        "changed_files": detail.get("changed_files"),
        "comments": detail.get("comments") or search_item.get("comments"),
        "review_comments": detail.get("review_comments"),
        "commits": detail.get("commits"),
        "author_login": (search_item.get("user") or {}).get("login"),
    }


def scrape_prs_for_login(
    login: str,
    client: GitHubClient,
    *,
    max_prs: int = DEFAULT_MAX_PRS_PER_ACCOUNT,
) -> list[dict[str, Any]]:
    """Fetch authored PRs for a login across all repositories via issue search."""
    prs: list[dict[str, Any]] = []
    page = 1
    while len(prs) < max_prs and page <= 10:
        query = f"type:pr author:{login} created:>={PRE_START.date().isoformat()}"
        url = (
            "https://api.github.com/search/issues?"
            + urllib.parse.urlencode({
                "q": query,
                "sort": "created",
                "order": "asc",
                "per_page": 100,
                "page": page,
            })
        )
        result = client.get(url)
        items = (result or {}).get("items", []) if isinstance(result, dict) else []
        if not items:
            break
        for item in items:
            if len(prs) >= max_prs:
                break
            detail_url = (item.get("pull_request") or {}).get("url")
            detail = client.get(detail_url) if detail_url else None
            prs.append(normalise_pr(item, detail))
        if len(items) < 100:
            break
        page += 1
    return prs


def load_feature_accounts(path: Path = FEATURES_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    """Write JSON atomically so interrupted scrapes do not leave corrupt caches."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2))
    os.replace(tmp_path, path)


def record_status(
    status_path: Path,
    login: str,
    status: str,
    n_prs: int,
    error: str = "",
) -> None:
    """Append one scrape status row and flush it immediately."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not status_path.exists()
    with open(status_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "login", "status", "n_prs", "error"])
        writer.writerow([datetime.now(timezone.utc).isoformat(), login, status, n_prs, error])
        f.flush()
        os.fsync(f.fileno())


def select_accounts_for_scrape(accounts: pd.DataFrame, max_accounts: int | None) -> pd.DataFrame:
    """Select accounts for scraping, balancing treated/control in limited smoke runs."""
    if not max_accounts or max_accounts >= len(accounts) or "label" not in accounts.columns:
        return accounts.head(max_accounts) if max_accounts else accounts

    positives = accounts[accounts["label"] == 1]
    controls = accounts[accounts["label"] == 0]
    if positives.empty or controls.empty:
        return accounts.head(max_accounts)

    n_pos = max_accounts // 2 + (max_accounts % 2)
    n_control = max_accounts - n_pos
    selected = pd.concat([positives.head(n_pos), controls.head(n_control)], ignore_index=True)
    return selected


def scrape_accounts(
    accounts: pd.DataFrame,
    *,
    token: str | None,
    max_accounts: int | None = None,
    max_prs_per_account: int = DEFAULT_MAX_PRS_PER_ACCOUNT,
    cache_dir: Path = CACHE_DIR,
    status_path: Path = STATUS_PATH,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    client = GitHubClient(token)
    selected = select_accounts_for_scrape(accounts, max_accounts)
    for i, row in enumerate(selected.itertuples(index=False), start=1):
        login = row.login
        cache_file = cache_dir / f"{login}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text())
                n_prs = len(cached.get("prs", []))
            except (json.JSONDecodeError, OSError):
                n_prs = 0
            print(f"[{i}/{len(selected)}] {login}: cached ({n_prs} PRs)", flush=True)
            record_status(status_path, login, "cached", n_prs)
            continue
        print(f"[{i}/{len(selected)}] {login}: scraping PRs", flush=True)
        try:
            prs = scrape_prs_for_login(login, client, max_prs=max_prs_per_account)
            atomic_write_json(cache_file, {"login": login, "prs": prs})
            record_status(status_path, login, "done", len(prs))
            print(f"[{i}/{len(selected)}] {login}: done ({len(prs)} PRs)", flush=True)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            record_status(status_path, login, "error", 0, message[:500])
            print(f"[{i}/{len(selected)}] {login}: error: {message}", flush=True)
            continue


def build_outcome_dataset(accounts: pd.DataFrame, *, cache_dir: Path = CACHE_DIR) -> pd.DataFrame:
    rows = []
    for _, feature_row in accounts.iterrows():
        login = feature_row["login"]
        cache_file = cache_dir / f"{login}.json"
        if not cache_file.exists():
            continue
        cached = json.loads(cache_file.read_text())
        metrics = summarize_pr_outcomes(cached.get("prs", []))
        row = build_did_row(feature_row, metrics)
        row["n_prs"] = len(cached.get("prs", []))
        rows.append(row)
    return pd.DataFrame(rows)


def write_did_report(
    outcomes: pd.DataFrame,
    results: pd.DataFrame,
    *,
    path: Path = DID_RESULTS_PATH,
    sensitivity: pd.DataFrame | None = None,
    diagnostics: dict[str, int] | None = None,
) -> None:
    with open(path, "w") as f:
        f.write("Account-level PR outcome DiD\n")
        f.write("============================\n\n")
        diagnostics = diagnostics or build_coverage_diagnostics(outcomes)
        f.write("Coverage diagnostics\n")
        f.write("--------------------\n")
        for key, value in diagnostics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Interpretation warning: PR-volume outcomes are the primary accepted-output check. "
                "Merge-rate and time-to-merge estimates are secondary because zero-PR windows are encoded as zeros and can distort rate/latency comparisons.\n\n")
        if results.empty:
            f.write("No estimable regressions. Need both treated and control accounts.\n")
        else:
            f.write("Main estimates\n")
            f.write("--------------\n")
            f.write(results.to_string(index=False))
            f.write("\n\n")
        if sensitivity is not None and not sensitivity.empty:
            f.write("Sensitivity estimates\n")
            f.write("---------------------\n")
            f.write(sensitivity.to_string(index=False))
            f.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scrape", action="store_true", help="Fetch PRs from GitHub into data/pr_outcome_cache")
    parser.add_argument("--analyse", action="store_true", help="Build account_pr_outcomes.csv and DiD report")
    parser.add_argument("--max-accounts", type=int, default=None, help="Limit accounts for a smoke run")
    parser.add_argument("--max-prs-per-account", type=int, default=DEFAULT_MAX_PRS_PER_ACCOUNT)
    parser.add_argument("--features-path", type=Path, default=FEATURES_PATH)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    parser.add_argument("--outcomes-path", type=Path, default=OUTCOMES_PATH)
    parser.add_argument("--did-results-path", type=Path, default=DID_RESULTS_PATH)
    parser.add_argument("--status-path", type=Path, default=STATUS_PATH)
    args = parser.parse_args()

    if not args.scrape and not args.analyse:
        parser.error("Choose --scrape, --analyse, or both")

    accounts = load_feature_accounts(args.features_path)
    if args.scrape:
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            raise SystemExit("GITHUB_TOKEN is required for --scrape")
        scrape_accounts(
            accounts,
            token=token,
            max_accounts=args.max_accounts,
            max_prs_per_account=args.max_prs_per_account,
            cache_dir=args.cache_dir,
            status_path=args.status_path,
        )

    if args.analyse:
        outcomes = build_outcome_dataset(accounts, cache_dir=args.cache_dir)
        args.outcomes_path.parent.mkdir(parents=True, exist_ok=True)
        outcomes.to_csv(args.outcomes_path, index=False, quoting=csv.QUOTE_MINIMAL)
        results = compute_did_results(outcomes)
        diagnostics = build_coverage_diagnostics(outcomes, max_prs_per_account=args.max_prs_per_account)
        sensitivity = build_sensitivity_results(outcomes, max_prs_per_account=args.max_prs_per_account)
        args.did_results_path.parent.mkdir(parents=True, exist_ok=True)
        write_did_report(
            outcomes,
            results,
            path=args.did_results_path,
            sensitivity=sensitivity,
            diagnostics=diagnostics,
        )
        print(f"Wrote {args.outcomes_path} ({len(outcomes)} accounts)")
        print(f"Wrote {args.did_results_path}")


if __name__ == "__main__":
    main()
