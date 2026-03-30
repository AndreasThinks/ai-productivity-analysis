#!/usr/bin/env python3
"""
Aider Validation Script

Validates the Claude Code classifier against confirmed Aider users.

Strategy:
  1. Find confirmed Aider accounts via GitHub Commit Search API
     (searching for noreply@aider.chat co-author trailers)
  2. Scrape behavioural features for those accounts using the same
     pipeline as the main classifier scraper
  3. Load the trained RF classifier pkl
  4. Score Aider accounts + compare score distributions against:
       - Training positives (Claude Code users)
       - Training negatives (controls)
  5. Report: does the model score Aider users above the baseline?

Interpretation:
  - Aider >> negatives: model detects general AI-assisted dev behaviour ✓
  - Aider ~ negatives: model is Claude-specific, not general ✗
  - Aider ~ Claude positives: model fully generalises across tools

Aider's commit trailer format (reliable, default-on):
  Co-authored-by: aider (model-name) <noreply@aider.chat>
  Co-authored-by: aider (model-name) <aider@aider.chat>
"""

import os
import json
import csv
import time
import re
import socket
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

# ---------------------------------------------------------------------------
# Config — mirrors scrape_classifier_full.py conventions
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

PROJECT_ROOT = Path("/home/andreasclaw/projects/ai_productivity_analysis")
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = DATA_DIR / "aider_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH    = DATA_DIR / "classifier_model.pkl"
FEATURES_PATH = DATA_DIR / "classifier_full_features.csv"  # training data for comparison
OUTPUT_PATH   = DATA_DIR / "aider_validation_results.csv"
REPORT_PATH   = DATA_DIR / "aider_validation_report.txt"

MAX_AIDER_ACCOUNTS = 80   # enough for a meaningful distribution comparison
API_DELAY          = 1.0
REQUEST_TIMEOUT    = 15
MAX_RETRIES        = 5
SECONDARY_RATE_LIMIT_FLOOR = 60
NETWORK_RETRY_FLOOR        = 120
NETWORK_MAX_RETRIES        = 8
CONSECUTIVE_NETWORK_FAIL_LIMIT = 5
CIRCUIT_BREAKER_PAUSE      = 300

# Temporal windows — same as classifier training
PRE_START   = datetime(2022, 1, 1)
PRE_CUTOFF  = datetime(2024, 1, 1)
POST_START  = datetime(2024, 1, 1)

AIDER_LAUNCH    = datetime(2023, 6, 1)   # Aider first public release
DATE_SANITY_MAX = datetime(2027, 1, 1)

MIN_PRE_COMMITS  = 10
MIN_POST_COMMITS = 10

MAX_FILE_SAMPLE_PER_ACCOUNT = 40
FILE_SAMPLE_RATE  = 0.20
FILE_SAMPLE_DELAY = 1.0

# Strip Aider co-author trailers from messages before feature extraction
# to avoid leaking the Aider label into writing-style features.
AIDER_COAUTHOR_STRIP_RE = re.compile(
    r"Co-authored-by:\s*aider[\s\w.()\-]*<?(?:noreply@aider\.chat|aider@aider\.chat)>?",
    re.IGNORECASE,
)

# Aider commit search queries (date-chunked to bypass 1000-result cap)
_AIDER_COMMIT_SEARCH_QUERIES = [
    ("noreply%40aider.chat+committer-date%3A2026-01-01..2026-12-31",
     "aider email (2026+)"),
    ("noreply%40aider.chat+committer-date%3A2025-07-01..2025-12-31",
     "aider email (2025-H2)"),
    ("noreply%40aider.chat+committer-date%3A2025-01-01..2025-06-30",
     "aider email (2025-H1)"),
    ("noreply%40aider.chat+committer-date%3A2024-01-01..2024-12-31",
     "aider email (2024)"),
    ("noreply%40aider.chat+committer-date%3A2023-06-01..2023-12-31",
     "aider email (2023-H2)"),
    # Second email variant used by some Aider versions
    ("aider%40aider.chat+committer-date%3A2025-01-01..2026-12-31",
     "aider@aider.chat (2025-2026)"),
    ("aider%40aider.chat+committer-date%3A2023-06-01..2024-12-31",
     "aider@aider.chat (2023-2024)"),
]


# ---------------------------------------------------------------------------
# HTTP helpers (copied from scrape_classifier_full.py)
# ---------------------------------------------------------------------------

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "aider-validator/1.0",
    }


class NetworkError(Exception):
    pass


def _is_network_error(exc):
    if isinstance(exc, (socket.gaierror, socket.timeout, ConnectionError,
                        ConnectionResetError, ConnectionRefusedError, BrokenPipeError)):
        return True
    if isinstance(exc, urllib.error.URLError) and not isinstance(exc, urllib.error.HTTPError):
        return True
    if isinstance(exc, OSError) and exc.errno in (101, 110, 111, 113):
        return True
    return False


def gh_get(url, extra_headers=None):
    headers = _gh_headers()
    if extra_headers:
        headers.update(extra_headers)

    network_attempts = 0
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            if e.code == 403 and "secondary" in body.lower():
                wait = SECONDARY_RATE_LIMIT_FLOOR * (attempt + 1)
                print(f"    Secondary rate limit, sleeping {wait}s...")
                time.sleep(wait)
            elif e.code == 403:
                reset_ts = e.headers.get("X-RateLimit-Reset")
                if reset_ts:
                    wait = max(5, int(reset_ts) - int(time.time()) + 5)
                else:
                    wait = SECONDARY_RATE_LIMIT_FLOOR
                print(f"    Rate limit (403), sleeping {wait}s...")
                time.sleep(wait)
            elif e.code in (404, 409, 451):
                return None
            elif e.code >= 500:
                time.sleep(API_DELAY * (2 ** attempt))
            else:
                print(f"    HTTP {e.code} for {url}: {body[:120]}")
                return None
        except Exception as exc:
            if _is_network_error(exc):
                network_attempts += 1
                wait = NETWORK_RETRY_FLOOR * network_attempts
                print(f"    Network error ({exc}), sleeping {wait}s... "
                      f"(attempt {network_attempts}/{NETWORK_MAX_RETRIES})")
                if network_attempts >= NETWORK_MAX_RETRIES:
                    raise NetworkError(f"Network failed after {NETWORK_MAX_RETRIES} attempts") from exc
                time.sleep(wait)
            else:
                print(f"    Unexpected error for {url}: {exc}")
                return None
    return None


def _sleep():
    time.sleep(API_DELAY)


# ---------------------------------------------------------------------------
# Stage 1 — find Aider accounts via Commit Search API
# ---------------------------------------------------------------------------

def find_aider_accounts():
    """Search GitHub commit history for Aider co-author trailers."""
    print("\n=== STAGE 1: Finding Aider accounts via Commit Search ===")
    accounts = {}

    for query_str, query_label in _AIDER_COMMIT_SEARCH_QUERIES:
        if len(accounts) >= MAX_AIDER_ACCOUNTS:
            print(f"  Cap reached ({MAX_AIDER_ACCOUNTS}), stopping search")
            break

        print(f"\n  Query: {query_label}")
        page = 1

        while len(accounts) < MAX_AIDER_ACCOUNTS and page <= 10:
            url = (
                "https://api.github.com/search/commits"
                f"?q={query_str}&per_page=100&page={page}"
            )
            result = gh_get(url, extra_headers={
                "Accept": "application/vnd.github.cloak-preview+json",
            })
            _sleep()
            if result is None:
                print(f"    Request failed on page {page}, skipping query")
                break

            items = result.get("items", [])
            if not items:
                print(f"    No more results at page {page}")
                break

            new_this_page = 0
            for item in items:
                if len(accounts) >= MAX_AIDER_ACCOUNTS:
                    break
                author = item.get("author") or {}
                login = author.get("login", "")
                if not login or author.get("type") != "User" or login in accounts:
                    continue

                commit_date = (
                    item.get("commit", {}).get("committer", {}).get("date", "")
                    or item.get("commit", {}).get("author", {}).get("date", "")
                )
                try:
                    cdt = datetime.fromisoformat(
                        commit_date.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if not (AIDER_LAUNCH <= cdt <= DATE_SANITY_MAX):
                        continue
                except (ValueError, AttributeError):
                    continue

                accounts[login] = {
                    "login": login,
                    "discovery_method": "commit_search_aider",
                    "first_marker_date": commit_date,
                    "marker_type": f"aider_coauthor: {query_label}",
                    "marker_confidence": "high",
                }
                new_this_page += 1
                print(f"    {login}  ({commit_date[:10]})")

            total = result.get("total_count", "?")
            print(f"    Page {page}: {len(items)} results (total={total}), "
                  f"+{new_this_page} new, {len(accounts)} total")
            if len(items) < 100:
                break
            page += 1

    print(f"\nAider accounts found: {len(accounts)}")
    return accounts


# ---------------------------------------------------------------------------
# Stage 2 — scrape behavioural features (shared logic from main scraper)
# ---------------------------------------------------------------------------

def _scrape_commits_for_repo(owner, repo_name, account_login, max_commits=200):
    commits = []
    page = 1
    while len(commits) < max_commits:
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            f"?author={account_login}&per_page=100&page={page}"
        )
        data = gh_get(url)
        _sleep()
        if not data or not isinstance(data, list):
            break
        for item in data:
            c = item.get("commit", {})
            commits.append({
                "sha":        item.get("sha", ""),
                "message":    c.get("message", ""),
                "created_at": (c.get("committer") or c.get("author") or {}).get("date", ""),
                "repo":       repo_name,
                "file_sampled": False,
                "has_test_file": None,
                "has_impl_file": None,
            })
        if len(data) < 100:
            break
        page += 1
    return commits


def _scrape_prs_for_repo(owner, repo_name, account_login, max_prs=100):
    prs = []
    page = 1
    while len(prs) < max_prs:
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
            f"?state=all&per_page=100&page={page}"
        )
        data = gh_get(url)
        _sleep()
        if not data or not isinstance(data, list):
            break
        for pr in data:
            user = pr.get("user") or {}
            if user.get("login") != account_login:
                continue
            body = pr.get("body") or ""
            prs.append({
                "created_at":  pr.get("created_at", ""),
                "body_length": len(body),
            })
        if len(data) < 100:
            break
        page += 1
    return prs


def _sample_commit_files(owner, repo_name, commits):
    """Sample a fraction of commits and check for test + impl files."""
    import math
    n = len(commits)
    sample_size = min(MAX_FILE_SAMPLE_PER_ACCOUNT,
                      max(1, math.ceil(n * FILE_SAMPLE_RATE)))
    indices = sorted(
        __import__("random").sample(range(n), min(sample_size, n))
    )
    test_re = re.compile(
        r"(test|spec|_test\.|\.test\.|\.spec\.)", re.IGNORECASE
    )
    impl_re = re.compile(
        r"\.(py|js|ts|rb|go|java|cs|cpp|c|rs|php|swift|kt)$", re.IGNORECASE
    )
    for idx in indices:
        c = commits[idx]
        sha = c.get("sha", "")
        if not sha:
            continue
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
        data = gh_get(url)
        time.sleep(FILE_SAMPLE_DELAY)
        if not data:
            continue
        files = data.get("files", [])
        filenames = [f.get("filename", "") for f in files]
        c["file_sampled"]   = True
        c["has_test_file"]  = any(test_re.search(fn) for fn in filenames)
        c["has_impl_file"]  = any(impl_re.search(fn) for fn in filenames)
    return commits


def scrape_account(login):
    """Scrape commits and PRs for an account. Same logic as main scraper."""
    cache_path = CACHE_DIR / f"{login}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if not cached.get("error"):
            print(f"  {login}: cache hit")
            return cached

    # Get user profile
    profile = gh_get(f"https://api.github.com/users/{login}")
    _sleep()
    if not profile:
        return {"error": "profile fetch failed", "commits": [], "prs": []}

    # Get repos sorted by created date (oldest first for pre-period coverage)
    repos_data = gh_get(
        f"https://api.github.com/users/{login}/repos"
        f"?type=owner&sort=created&direction=asc&per_page=100"
    )
    _sleep()
    if not repos_data or not isinstance(repos_data, list):
        return {"error": "repos fetch failed", "commits": [], "prs": []}

    repos = [r["name"] for r in repos_data if not r.get("fork", False)][:20]

    all_commits = []
    all_prs     = []

    for repo_name in repos:
        commits = _scrape_commits_for_repo(login, repo_name, login)
        if commits:
            commits = _sample_commit_files(login, repo_name, commits)
            all_commits.extend(commits)
        prs = _scrape_prs_for_repo(login, repo_name, login)
        all_prs.extend(prs)

    result = {
        "login":   login,
        "commits": all_commits,
        "prs":     all_prs,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    print(f"  {login}: {len(all_commits)} commits, {len(all_prs)} PRs")
    return result


# ---------------------------------------------------------------------------
# Stage 3 — feature extraction (shared logic from main scraper)
# ---------------------------------------------------------------------------

def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _count_commits_in_window(commits, after, before=None):
    count = 0
    for c in commits:
        dt = _parse_dt(c.get("created_at"))
        if dt is None:
            continue
        if dt < after:
            continue
        if before and dt >= before:
            continue
        count += 1
    return count


def _deduplicate_commits(commits):
    seen, unique = set(), []
    for c in commits:
        sha = c.get("sha", "")
        if sha and sha in seen:
            continue
        seen.add(sha)
        unique.append(c)
    return unique


def _window_commit_features(commits, after, before=None, strip_re=None):
    window = [
        c for c in commits
        if _parse_dt(c.get("created_at")) is not None
        and _parse_dt(c.get("created_at")) >= after
        and (not before or _parse_dt(c.get("created_at")) < before)
    ]
    if not window:
        return {
            "commit_count": 0, "mean_message_length": 0.0, "active_weeks": 0,
            "repos_touched": 0, "mean_commits_per_active_week": 0.0,
            "frac_multiline": 0.0, "frac_conventional": 0.0,
            "frac_mentions_test": 0.0, "frac_has_bullets": 0.0,
            "mean_inter_commit_hours": 0.0, "frac_burst_commits": 0.0,
            "sampled_test_cowrite_rate": 0.0, "file_sample_count": 0,
            "mean_pr_body_length": 0.0, "frac_pr_has_body": 0.0,
        }

    def _strip(msg):
        if strip_re:
            msg = strip_re.sub("", msg)
        return msg.strip()

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2]
                        for c in window if _parse_dt(c["created_at"])})
    repos = len({c.get("repo", "") for c in window if c.get("repo")})
    cleaned = [_strip(c.get("message", "")) for c in window]
    msg_lengths = [len(m) for m in cleaned]

    conventional_re = re.compile(
        r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?:",
        re.IGNORECASE,
    )
    test_re = re.compile(r"\btest[s]?\b", re.IGNORECASE)

    multiline_count    = sum(1 for m in cleaned if "\n" in m)
    conventional_count = sum(1 for m in cleaned if conventional_re.match(m))
    test_count         = sum(1 for m in cleaned if test_re.search(m))
    bullets_count      = sum(1 for m in cleaned if "- " in m or "* " in m)

    sorted_w = sorted(window, key=lambda c: _parse_dt(c.get("created_at")) or datetime.min)
    inter_hours = []
    for i in range(1, len(sorted_w)):
        dt1 = _parse_dt(sorted_w[i - 1].get("created_at"))
        dt2 = _parse_dt(sorted_w[i].get("created_at"))
        if dt1 and dt2:
            inter_hours.append((dt2 - dt1).total_seconds() / 3600.0)

    mean_inter = sum(inter_hours) / len(inter_hours) if inter_hours else 0.0
    frac_burst = (sum(1 for h in inter_hours if h <= 2.0) / len(inter_hours)
                  if inter_hours else 0.0)

    sampled = [c for c in window if c.get("file_sampled")]
    if sampled:
        with_impl = sum(1 for c in sampled if c.get("has_impl_file"))
        with_both = sum(1 for c in sampled if c.get("has_impl_file") and c.get("has_test_file"))
        cowrite   = with_both / with_impl if with_impl > 0 else 0.0
    else:
        cowrite = 0.0

    return {
        "commit_count":                  len(window),
        "mean_message_length":           round(sum(msg_lengths) / len(msg_lengths), 2),
        "active_weeks":                  active_weeks,
        "repos_touched":                 repos,
        "mean_commits_per_active_week":  round(len(window) / max(active_weeks, 1), 2),
        "frac_multiline":                round(multiline_count / len(window), 3),
        "frac_conventional":             round(conventional_count / len(window), 3),
        "frac_mentions_test":            round(test_count / len(window), 3),
        "frac_has_bullets":              round(bullets_count / len(window), 3),
        "mean_inter_commit_hours":       round(mean_inter, 2),
        "frac_burst_commits":            round(frac_burst, 3),
        "sampled_test_cowrite_rate":     round(cowrite, 3),
        "file_sample_count":             len(sampled),
        "mean_pr_body_length":           0.0,
        "frac_pr_has_body":              0.0,
    }


def _window_pr_features(prs, after, before=None):
    window = [
        pr for pr in prs
        if _parse_dt(pr.get("created_at")) is not None
        and _parse_dt(pr.get("created_at")) >= after
        and (not before or _parse_dt(pr.get("created_at")) < before)
    ]
    if not window:
        return {"mean_pr_body_length": 0.0, "frac_pr_has_body": 0.0}
    body_lengths = [pr.get("body_length", 0) for pr in window]
    return {
        "mean_pr_body_length": round(sum(body_lengths) / len(body_lengths), 2),
        "frac_pr_has_body":    round(sum(1 for bl in body_lengths if bl > 50) / len(window), 3),
    }


def extract_features(accounts_dict, all_data):
    """Extract pre/post/delta features for Aider accounts."""
    print("\n=== STAGE 3: Feature extraction ===")
    rows = []
    skipped = 0

    for login, meta in accounts_dict.items():
        data = all_data.get(login, {})
        if data.get("error"):
            print(f"  {login}: skipped ({data['error']})")
            skipped += 1
            continue

        commits = _deduplicate_commits(data.get("commits", []))
        prs     = data.get("prs", [])

        # Use per-account post_start from the actual Aider commit timestamp
        marker_dt = _parse_dt(meta.get("first_marker_date", ""))
        if marker_dt and AIDER_LAUNCH <= marker_dt <= DATE_SANITY_MAX:
            account_post_start = marker_dt
            account_pre_cutoff = marker_dt
        else:
            account_post_start = POST_START
            account_pre_cutoff = PRE_CUTOFF

        pre_count  = _count_commits_in_window(commits, after=PRE_START, before=account_pre_cutoff)
        post_count = _count_commits_in_window(commits, after=account_post_start)

        if pre_count < MIN_PRE_COMMITS or post_count < MIN_POST_COMMITS:
            print(f"  {login}: SKIPPED both-window filter ({pre_count} pre, {post_count} post)")
            skipped += 1
            continue

        # Strip Aider trailers before feature extraction (prevent label leakage)
        pre_cf  = _window_commit_features(commits, PRE_START, account_pre_cutoff,
                                          strip_re=AIDER_COAUTHOR_STRIP_RE)
        post_cf = _window_commit_features(commits, account_post_start,
                                          strip_re=AIDER_COAUTHOR_STRIP_RE)
        pre_pf  = _window_pr_features(prs, PRE_START, account_pre_cutoff)
        post_pf = _window_pr_features(prs, account_post_start)

        pre_cf.update(pre_pf)
        post_cf.update(post_pf)

        row = {
            "login":              login,
            "group":              "aider",
            "discovery_method":   meta.get("discovery_method", ""),
            "first_marker_date":  meta.get("first_marker_date", ""),
        }
        for k, v in pre_cf.items():
            row[f"pre_{k}"] = v
        for k, v in post_cf.items():
            row[f"post_{k}"] = v
        for k in pre_cf:
            row[f"delta_{k}"] = round(post_cf[k] - pre_cf[k], 3)

        rows.append(row)
        print(f"  {login}: pre={pre_count} commits, post={post_count} commits, "
              f"Δmsg_len={row['delta_mean_message_length']:+.1f}")

    print(f"\n  Features extracted: {len(rows)}  ({skipped} skipped)")
    return rows


# ---------------------------------------------------------------------------
# Stage 4 — score with classifier + compare distributions
# ---------------------------------------------------------------------------

def score_and_compare(aider_rows):
    """Load pkl, score Aider accounts, compare against training distributions."""
    print("\n=== STAGE 4: Scoring and distribution comparison ===")

    # Load model
    pkg = joblib.load(MODEL_PATH)
    model        = pkg["model"]
    feature_cols = pkg["feature_cols"]
    print(f"  Model loaded: {pkg['model_name']}")
    print(f"  Feature columns expected: {len(feature_cols)}")

    # Load training data for comparison
    train_df = pd.read_csv(FEATURES_PATH)

    drop = {"login", "label", "marker_confidence", "has_claude_markers",
            "pre_commit_count", "post_commit_count", "discovery_method"}
    drop = {c for c in drop if c in train_df.columns}
    drop |= {c for c in train_df.columns if "marker" in c.lower()}

    y_train     = train_df["label"].values
    X_train_raw = train_df.drop(columns=list(drop))[feature_cols]
    imp = SimpleImputer(strategy="median")
    X_train = imp.fit_transform(X_train_raw)

    train_probs  = model.predict_proba(X_train)[:, 1]
    claude_probs = train_probs[y_train == 1]
    neg_probs    = train_probs[y_train == 0]

    # Score Aider accounts
    if not aider_rows:
        print("  No Aider rows to score.")
        return None

    aider_df  = pd.DataFrame(aider_rows)
    # Align to training feature columns — fill any missing cols with 0
    for col in feature_cols:
        if col not in aider_df.columns:
            aider_df[col] = 0.0
    X_aider_raw = aider_df[feature_cols]
    X_aider = imp.transform(X_aider_raw)   # use training imputer
    aider_probs = model.predict_proba(X_aider)[:, 1]

    # Add scores to rows
    for i, row in enumerate(aider_rows):
        row["classifier_prob"] = round(float(aider_probs[i]), 4)

    # Percentile rank in training distribution
    def _pct_rank(score, reference):
        return round(100.0 * (reference < score).mean(), 1)

    # Summary stats
    def _stats(arr):
        return {
            "n":      len(arr),
            "mean":   round(float(arr.mean()), 3),
            "median": round(float(np.median(arr)), 3),
            "std":    round(float(arr.std()), 3),
            "p25":    round(float(np.percentile(arr, 25)), 3),
            "p75":    round(float(np.percentile(arr, 75)), 3),
            "pct_above_0.5": round(100.0 * (arr >= 0.5).mean(), 1),
        }

    cs = _stats(claude_probs)
    ns = _stats(neg_probs)
    as_ = _stats(aider_probs)

    # Mann-Whitney U test: Aider vs negatives
    from scipy import stats as scipy_stats
    u_stat, p_aider_vs_neg   = scipy_stats.mannwhitneyu(aider_probs, neg_probs,
                                                         alternative="greater")
    u_stat2, p_aider_vs_cla  = scipy_stats.mannwhitneyu(aider_probs, claude_probs,
                                                         alternative="less")

    # Save scored rows
    aider_df["classifier_prob"] = aider_probs
    aider_df.to_csv(OUTPUT_PATH, index=False)

    # Build report
    lines = []
    lines.append("=" * 65)
    lines.append("AIDER VALIDATION REPORT")
    lines.append("=" * 65)
    lines.append(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"  Model: {pkg['model_name']}")
    lines.append(f"  Aider accounts scored: {len(aider_rows)}")
    lines.append("")
    lines.append("  Score distributions (classifier probability of AI use):")
    lines.append(f"  {'Group':<20}  {'N':>4}  {'Mean':>6}  {'Median':>7}  "
                 f"{'Std':>5}  {'P25':>5}  {'P75':>5}  {'>0.5':>5}")
    lines.append("  " + "-" * 63)
    for label, s in [("Claude (train +)", cs), ("Negatives (train)", ns), ("Aider", as_)]:
        lines.append(
            f"  {label:<20}  {s['n']:>4}  {s['mean']:>6.3f}  {s['median']:>7.3f}  "
            f"{s['std']:>5.3f}  {s['p25']:>5.3f}  {s['p75']:>5.3f}  {s['pct_above_0.5']:>4.1f}%"
        )
    lines.append("")
    lines.append(f"  Mann-Whitney: Aider > Negatives  p={p_aider_vs_neg:.4f}")
    lines.append(f"  Mann-Whitney: Aider < Claude     p={p_aider_vs_cla:.4f}")
    lines.append("")

    # Interpretation
    aider_mean = as_["mean"]
    neg_mean   = ns["mean"]
    cla_mean   = cs["mean"]
    gap_vs_neg = aider_mean - neg_mean
    gap_vs_cla = cla_mean - aider_mean

    lines.append("  Interpretation:")
    if p_aider_vs_neg < 0.05:
        lines.append(f"  ✓ Aider scores significantly above negatives (p={p_aider_vs_neg:.4f})")
        lines.append(f"    Gap vs negatives: +{gap_vs_neg:.3f}")
        if aider_mean >= 0.5:
            lines.append("  ✓ Majority of Aider accounts score above 0.5 threshold")
        if gap_vs_cla < 0.15:
            lines.append("  ✓ Gap vs Claude positives is small — model generalises well")
        elif gap_vs_cla < 0.30:
            lines.append(f"  ~ Gap vs Claude positives is moderate ({gap_vs_cla:.3f}) — "
                         "partial generalisation")
        else:
            lines.append(f"  ~ Gap vs Claude positives is large ({gap_vs_cla:.3f}) — "
                         "tool-specific signal present")
    else:
        lines.append(f"  ✗ Aider does NOT score significantly above negatives (p={p_aider_vs_neg:.4f})")
        lines.append("    The classifier may be Claude-specific, not general.")

    lines.append("")
    lines.append(f"  Files written:")
    lines.append(f"    {OUTPUT_PATH}")
    lines.append(f"    {REPORT_PATH}")
    lines.append("=" * 65)

    report_text = "\n".join(lines)
    print("\n" + report_text)
    with open(REPORT_PATH, "w") as f:
        f.write(report_text + "\n")

    return {
        "claude": cs, "negatives": ns, "aider": as_,
        "p_aider_vs_neg": p_aider_vs_neg,
        "p_aider_vs_cla": p_aider_vs_cla,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 65)
    print("AIDER VALIDATION PIPELINE")
    print("=" * 65)

    # Stage 1 — find Aider accounts
    aider_accounts = find_aider_accounts()
    if not aider_accounts:
        print("No Aider accounts found. Check API token and query.")
        return

    # Save account list
    accounts_path = DATA_DIR / "aider_accounts.csv"
    with open(accounts_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["login", "discovery_method",
                                           "first_marker_date", "marker_type",
                                           "marker_confidence"])
        w.writeheader()
        w.writerows(aider_accounts.values())
    print(f"  Saved {len(aider_accounts)} accounts → {accounts_path}")

    # Stage 2 — scrape
    print("\n=== STAGE 2: Scraping account data ===")
    all_data = {}
    consecutive_failures = 0
    for login in list(aider_accounts.keys()):
        try:
            all_data[login] = scrape_account(login)
            consecutive_failures = 0
        except NetworkError:
            consecutive_failures += 1
            print(f"  {login}: NetworkError (skip-not-reject)")
            if consecutive_failures >= CONSECUTIVE_NETWORK_FAIL_LIMIT:
                print(f"  Circuit breaker tripped — pausing {CIRCUIT_BREAKER_PAUSE}s")
                time.sleep(CIRCUIT_BREAKER_PAUSE)
                consecutive_failures = 0

    # Stage 3 — features
    aider_rows = extract_features(aider_accounts, all_data)

    # Stage 4 — score and compare
    score_and_compare(aider_rows)


if __name__ == "__main__":
    main()
