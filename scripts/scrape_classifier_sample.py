#!/usr/bin/env python3
"""
Subsample test scraper for Claude Code user classifier — v2.1

This is the small-scale test version of classifier_scraper_v2.py.  It uses
the same logic, fixes, and feature set but with reduced caps for quick
iteration (50 positives, 50 negatives, 30 accounts scraped per group).

All fixes from the full scraper are ported here:
- Broadened co-author regex (matches Claude Code, Claude Opus 4.6, etc.)
- Commit author filtering (?author={login}) to prevent cross-contamination
- Multi-hour GH Archive sampling (3 hours for subsample vs 6 for full)
- marker_confidence field (high for co-author, low for Code Search)
- Per-account temporal split for high-confidence positives only
- Symmetric both-window filter (applied to positives AND negatives)
- Feature leakage guard (has_claude_markers in separate CSV, not raw data)
- File sampling with explicit file_sampled flag
- sampled_test_cowrite_rate computed only over file-sampled commits
- Commit deduplication by SHA across repos
- Repo sort by created ascending (oldest first)
- Improved rate-limit handling (5 retries, 60s secondary floor)
- Commit message structure features (multiline, conventional, bullets, test)
- Inter-commit burst features (mean_inter_commit_hours, frac_burst_commits)
- PR body length features (mean_pr_body_length, frac_pr_has_body)
"""

import os
import json
import csv
import time
import gzip
import random
import re
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

PROJECT_ROOT = Path("/home/andreasclaw/projects/ai_productivity_analysis")
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "classifier_cache_subsample"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

_RUN_TAG = "subsample"

API_DELAY = 0.35
REQUEST_TIMEOUT = 15

# [FIX: rate limits] Increased retries and added secondary rate-limit floor
MAX_RETRIES = 5
BACKOFF_BASE = 2
SECONDARY_RATE_LIMIT_FLOOR = 60

# [FIX: multi-hour GH Archive] 3 hours for subsample (full uses 6)
GH_ARCHIVE_HOURS = [
    ("2025-01-13", 9),   # Monday morning UTC
    ("2025-01-15", 3),   # Wednesday early morning UTC
    ("2025-01-17", 14),  # Friday afternoon UTC
]

# Temporal split
PRE_CUTOFF  = datetime(2024, 1, 1)
POST_START  = datetime(2024, 1, 1)  # global fallback for negatives + low-confidence
PRE_START   = datetime(2022, 1, 1)

# Both-window threshold (applied symmetrically to positives AND negatives)
MIN_PRE_COMMITS  = 10
MIN_POST_COMMITS = 10

# Subsample caps
MAX_POSITIVES       = 50
MAX_NEGATIVES       = 50
SCRAPE_CAP_POSITIVE = 30
SCRAPE_CAP_NEGATIVE = 30
MAX_REPOS_PER_ACCT  = 5

# File sampling
MAX_FILE_SAMPLE_PER_ACCOUNT = 20   # lower than full run for speed
FILE_SAMPLE_RATE = 0.20
FILE_SAMPLE_DELAY = 0.35

# Progress reporting
PROGRESS_INTERVAL = 10

random.seed(42)


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "classifier-scraper-subsample-v2/1.0",
    }


def gh_get(url):
    """GET a GitHub API URL with retry + rate-limit-aware backoff.

    [FIX: rate limits] Secondary rate limits now sleep for at least
    SECONDARY_RATE_LIMIT_FLOOR (60s).  MAX_RETRIES is 5.
    """
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=_gh_headers())
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}

        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                remaining = e.headers.get("X-RateLimit-Remaining", "1")
                reset_ts  = e.headers.get("X-RateLimit-Reset",     "0")
                try:
                    remaining = int(remaining)
                    reset_ts  = int(reset_ts)
                except ValueError:
                    remaining, reset_ts = 1, 0

                if remaining == 0 and reset_ts > 0:
                    wait = max(reset_ts - int(time.time()) + 5, 5)
                    print(f"    Rate limit exhausted. Sleeping {wait}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    wait = max(SECONDARY_RATE_LIMIT_FLOOR, (BACKOFF_BASE ** attempt) * 2)
                    print(f"    Secondary rate limit. Waiting {wait}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)

            elif e.code == 404:
                return None
            elif e.code == 409:
                return None
            else:
                print(f"    HTTP {e.code} on {url}")
                return None

        except Exception as e:
            wait = (BACKOFF_BASE ** attempt)
            print(f"    request error ({e}), waiting {wait}s")
            time.sleep(wait)

    print(f"    Failed after {MAX_RETRIES} retries: {url}")
    return None


def _sleep():
    time.sleep(API_DELAY)


# ---------------------------------------------------------------------------
# GH Archive helpers — multi-hour
# ---------------------------------------------------------------------------

def _gh_archive_cache_path(date_str, hour):
    return DATA_DIR / f"gharchive_{date_str}-{hour}.jsonl"


def ensure_gh_archive():
    """Download and cache all configured GH Archive hours to disk."""
    all_ok = True
    for date_str, hour in GH_ARCHIVE_HOURS:
        cache_path = _gh_archive_cache_path(date_str, hour)
        if cache_path.exists():
            print(f"GH Archive cache exists: {cache_path}")
            continue

        url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
        print(f"Streaming GH Archive: {url}")
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "classifier-scraper-subsample-v2/1.0"}
            )
            with urllib.request.urlopen(req, timeout=180) as resp:
                with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                    with open(cache_path, "w", encoding="utf-8") as out:
                        count = 0
                        for line in gz:
                            line = line.rstrip("\n")
                            if line:
                                out.write(line + "\n")
                                count += 1
            print(f"Streamed {count} events → {cache_path}")
        except Exception as e:
            print(f"Failed to download GH Archive {date_str}-{hour}: {e}")
            all_ok = False
    return all_ok


def iter_gh_archive():
    """Yield parsed events from ALL cached archive hours."""
    for date_str, hour in GH_ARCHIVE_HOURS:
        cache_path = _gh_archive_cache_path(date_str, hour)
        if not cache_path.exists():
            continue
        with open(cache_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


# ---------------------------------------------------------------------------
# Stage 1a — Code Search for CLAUDE.md
#
# [FIX: marker_confidence] Tagged "low" because repo.created_at is NOT a
# reliable proxy for when CLAUDE.md was committed.  Feature extraction falls
# back to global POST_START for these accounts.
# ---------------------------------------------------------------------------

def stage1a_code_search():
    """Find accounts with CLAUDE.md via GitHub Code Search API."""
    print("\n=== STAGE 1a: Code Search for CLAUDE.md ===")
    positives = {}
    page = 1

    while len(positives) < MAX_POSITIVES and page <= 5:
        url = f"https://api.github.com/search/code?q=filename:CLAUDE.md&per_page=100&page={page}"
        print(f"  page {page}...")
        result = gh_get(url)
        _sleep()

        if not result or "items" not in result:
            print("  no more results")
            break

        for item in result["items"]:
            if len(positives) >= MAX_POSITIVES:
                break
            owner = item.get("repository", {}).get("owner", {})
            login = owner.get("login", "")
            if owner.get("type") != "User" or not login or login in positives:
                continue

            repo_full    = item.get("repository", {}).get("full_name", "")
            repo_created = item.get("repository", {}).get("created_at", "")

            positives[login] = {
                "login": login,
                "discovery_method": "code_search",
                "first_marker_date": repo_created,
                "marker_type": "CLAUDE.md",
                "marker_confidence": "low",
            }
            print(f"    {login}  ({repo_full}, created {repo_created[:10] or '?'})")

        page += 1

    print(f"Code search: {len(positives)} unique user accounts")
    return positives


# ---------------------------------------------------------------------------
# Stage 1b — GH Archive co-author scan
#
# [FIX: regex] Broadened to match all known Claude Code co-author formats:
#   Claude <noreply@...>, Claude Code <noreply@...>,
#   Claude Opus 4.6 <noreply@...>, bare email without angle brackets.
# [FIX: marker_confidence] Tagged "high" — actual push event timestamp.
# ---------------------------------------------------------------------------

CLAUDE_COAUTHOR_RE = re.compile(
    r"Co-[Aa]uthored-[Bb]y:\s*Claude[\s\w.]*<?noreply@anthropic\.com>?",
    re.IGNORECASE,
)


def stage1b_gh_archive():
    """Find accounts with Co-Authored-By: Claude across all archive hours."""
    print("\n=== STAGE 1b: GH Archive Co-Authored-By scan ===")
    positives = {}

    for event in iter_gh_archive():
        if event.get("type") != "PushEvent":
            continue
        login = event.get("actor", {}).get("login", "")
        if not login or login in positives:
            continue
        for commit in event.get("payload", {}).get("commits", []):
            if CLAUDE_COAUTHOR_RE.search(commit.get("message", "")):
                positives[login] = {
                    "login": login,
                    "discovery_method": "gh_archive_coauthor",
                    "first_marker_date": event.get("created_at", ""),
                    "marker_type": "Co-Authored-By: Claude",
                    "marker_confidence": "high",
                }
                print(f"    {login}")
                break

    print(f"GH Archive co-author scan: {len(positives)} accounts")
    return positives


# ---------------------------------------------------------------------------
# Stage 2 — Negative candidates
# ---------------------------------------------------------------------------

def stage2_negatives(positive_logins):
    """Sample active developers from GH Archive who are NOT in the positive set."""
    print("\n=== STAGE 2: Negative candidate discovery ===")
    actor_events = defaultdict(int)

    for event in iter_gh_archive():
        if event.get("type") == "PushEvent":
            login = event.get("actor", {}).get("login", "")
            if login:
                actor_events[login] += 1

    candidates = [
        login for login, count in actor_events.items()
        if count >= 5 and login not in positive_logins
    ]
    print(f"  {len(candidates)} active candidates (≥5 push events, not in positive set)")

    sampled = random.sample(candidates, min(MAX_NEGATIVES, len(candidates)))
    negatives = {
        login: {
            "login": login,
            "discovery_method": "gh_archive_sample",
            "first_marker_date": None,
            "marker_type": None,
            "marker_confidence": None,
        }
        for login in sampled
    }
    print(f"  sampled {len(negatives)} negatives")
    return negatives


# ---------------------------------------------------------------------------
# Stage 3 — Per-account deep scrape with file sampling
# ---------------------------------------------------------------------------

def _scrape_commits_for_repo(owner, repo_name, account_login, max_commits=200):
    """Fetch up to max_commits commits for one repo via /commits API.

    [FIX: author filtering] Uses ?author={account_login} to only return
    commits authored by the account we're scraping.  Without this, commits
    from PR contributors and co-maintainers get mixed into the account's
    feature set, diluting the signal with other people's coding patterns.
    """
    commits = []
    page = 1
    while len(commits) < max_commits and page <= 2:
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            f"?author={account_login}&per_page=100&page={page}"
        )
        result = gh_get(url)
        _sleep()
        if not result or not isinstance(result, list):
            break
        for c in result:
            commit_obj = c.get("commit", {})
            author_date = commit_obj.get("author", {}).get("date", "")
            stats = c.get("stats", {})
            commits.append({
                "sha": c.get("sha", "")[:12],
                "message": commit_obj.get("message", "")[:500],
                "created_at": author_date,
                "repo": f"{owner}/{repo_name}",
                "additions": stats.get("additions"),
                "deletions": stats.get("deletions"),
                "has_test_file": None,
                "has_impl_file": None,
                "file_sampled": False,
            })
        if len(result) < 100:
            break
        page += 1
    return commits


def _scrape_prs_for_repo(owner, repo_name, max_prs=50):
    """Fetch up to max_prs closed/merged PRs for one repo."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
        f"?state=closed&per_page={max_prs}&sort=updated&direction=desc"
    )
    result = gh_get(url)
    _sleep()
    if not result or not isinstance(result, list):
        return []
    prs = []
    for pr in result:
        prs.append({
            "title": (pr.get("title") or "")[:100],
            "body_length": len(pr.get("body") or ""),
            "created_at": pr.get("created_at"),
            "merged_at": pr.get("merged_at"),
            "state": pr.get("state"),
        })
    return prs


def _sample_commit_files(owner, repo_name, commits):
    """For ~20% of commits, fetch file changes.

    [FIX: file_sampled flag] Each sampled commit gets file_sampled=True so
    feature extraction computes test_cowrite_rate only over sampled commits.
    """
    if not commits:
        return

    sample_indices = random.sample(
        range(len(commits)),
        min(MAX_FILE_SAMPLE_PER_ACCOUNT, max(1, int(len(commits) * FILE_SAMPLE_RATE)))
    )

    impl_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"}
    test_keywords = {"test", "spec"}

    for idx in sample_indices:
        commit = commits[idx]
        sha = commit["sha"]

        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
        detail = gh_get(url)
        time.sleep(FILE_SAMPLE_DELAY)

        if not detail or "files" not in detail:
            continue

        commit["file_sampled"] = True
        files = detail.get("files", [])
        has_test = False
        has_impl = False

        for file_obj in files:
            filename = file_obj.get("filename", "").lower()
            if any(kw in filename for kw in test_keywords):
                has_test = True
            if any(filename.endswith(ext) for ext in impl_extensions):
                has_impl = True

        commit["has_test_file"] = has_test
        commit["has_impl_file"] = has_impl


def scrape_account(login):
    """Deep scrape one account. Returns from cache if available.

    [FIX: leakage guard] has_claude_markers written to separate CSV, not raw data.
    [FIX: repo sort] Sorted by created ascending — oldest repos first.
    """
    cache_file = CACHE_DIR / f"{login}.json"
    if cache_file.exists():
        print(f"  {login}: cached")
        with open(cache_file) as f:
            return json.load(f)

    print(f"  {login}: scraping...", end="", flush=True)
    data = {"login": login, "profile": None, "repos": [], "commits": [], "prs": [], "error": None}

    # Profile
    profile = gh_get(f"https://api.github.com/users/{login}")
    _sleep()
    if profile is None:
        data["error"] = "profile fetch failed"
        cache_file.write_text(json.dumps(data, indent=2))
        print(" FAILED")
        return data

    data["profile"] = {
        "login":        profile.get("login"),
        "created_at":   profile.get("created_at"),
        "location":     profile.get("location"),
        "public_repos": profile.get("public_repos"),
    }

    # [FIX: repo sort] Sort by created ascending — oldest repos first for
    # better pre-period coverage without the dormant-repo problem.
    repos_raw = gh_get(
        f"https://api.github.com/users/{login}/repos?sort=created&direction=asc&per_page=30"
    )
    _sleep()
    if not repos_raw or not isinstance(repos_raw, list):
        repos_raw = []

    repos_to_scrape = repos_raw[:MAX_REPOS_PER_ACCT]

    # [FIX: leakage guard] Marker detection separated from raw data
    claude_marker_repos = []

    for repo in repos_to_scrape:
        repo_name  = repo.get("name", "")
        owner_name = repo.get("owner", {}).get("login", login)

        contents = gh_get(
            f"https://api.github.com/repos/{owner_name}/{repo_name}/contents/"
        )
        _sleep()

        has_claude_marker = False
        if contents and isinstance(contents, list):
            for item in contents:
                if item.get("name", "").lower() in ("claude.md", ".claude", ".hermes", "agents.md"):
                    has_claude_marker = True
                    break

        if has_claude_marker:
            claude_marker_repos.append(f"{owner_name}/{repo_name}")

        # [FIX: leakage guard] No has_claude_markers in repo data
        data["repos"].append({
            "name":       repo_name,
            "created_at": repo.get("created_at"),
            "language":   repo.get("language"),
            "size":       repo.get("size"),
        })

        # [FIX: author filtering] Only fetch this account's commits
        commits = _scrape_commits_for_repo(owner_name, repo_name, login)
        _sample_commit_files(owner_name, repo_name, commits)
        data["commits"].extend(commits)

        prs = _scrape_prs_for_repo(owner_name, repo_name)
        data["prs"].extend(prs)

    # Write marker info to separate labeling file
    if claude_marker_repos:
        marker_path = DATA_DIR / f"{_RUN_TAG}_claude_markers.csv"
        write_header = not marker_path.exists()
        with open(marker_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["login", "repo", "marker_found"])
            for repo_full in claude_marker_repos:
                w.writerow([login, repo_full, True])

    print(f" {len(data['commits'])} commits, {len(data['prs'])} PRs")
    cache_file.write_text(json.dumps(data, indent=2))
    return data


def stage3_scrape_all(positives, negatives):
    """Deep scrape capped positive + negative sets."""
    print("\n=== STAGE 3: Per-account deep scrape ===")
    all_data = {}

    pos_logins = list(positives)[:SCRAPE_CAP_POSITIVE]
    neg_logins = list(negatives)[:SCRAPE_CAP_NEGATIVE]

    print(f"Positives to scrape: {len(pos_logins)}")
    for i, login in enumerate(pos_logins):
        all_data[login] = scrape_account(login)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"  Progress: {i+1}/{len(pos_logins)} positives scraped")

    print(f"Negatives to scrape: {len(neg_logins)}")
    for i, login in enumerate(neg_logins):
        all_data[login] = scrape_account(login)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"  Progress: {i+1}/{len(neg_logins)} negatives scraped")

    raw_path = DATA_DIR / "classifier_subsample_raw.json"
    raw_path.write_text(json.dumps(all_data, indent=2))
    print(f"Raw data saved → {raw_path}")
    return all_data


# ---------------------------------------------------------------------------
# Stage 4 — Feature extraction
# ---------------------------------------------------------------------------

def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
        return None


def _count_commits_in_window(commits, after, before=None):
    """Count commits in [after, before)."""
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
    """[FIX: dedup] Remove duplicate commits (same SHA from forked repos)."""
    seen = set()
    unique = []
    for c in commits:
        sha = c.get("sha", "")
        if sha and sha in seen:
            continue
        seen.add(sha)
        unique.append(c)
    return unique


def _window_commit_features(commits, after, before=None):
    """Compute commit features for commits in [after, before).

    Includes all features from the full scraper: message structure,
    inter-commit burst timing, and file-sample-aware test cowrite rate.
    """
    window = []
    for c in commits:
        dt = _parse_dt(c.get("created_at"))
        if dt is None:
            continue
        if dt < after:
            continue
        if before and dt >= before:
            continue
        window.append(c)

    if not window:
        return {
            "commit_count":                   0,
            "mean_message_length":            0.0,
            "active_weeks":                   0,
            "repos_touched":                  0,
            "mean_commits_per_active_week":   0.0,
            "frac_multiline":                 0.0,
            "frac_conventional":              0.0,
            "frac_mentions_test":             0.0,
            "frac_has_bullets":               0.0,
            "mean_inter_commit_hours":        0.0,
            "frac_burst_commits":             0.0,
            "sampled_test_cowrite_rate":      0.0,
            "file_sample_count":              0,
            "mean_pr_body_length":            0.0,
            "frac_pr_has_body":               0.0,
        }

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2] for c in window
                        if _parse_dt(c["created_at"])})
    repos        = len({c.get("repo", "") for c in window if c.get("repo")})
    msg_lengths  = [len(c.get("message", "")) for c in window]

    # Commit message structure features
    multiline_count = sum(1 for c in window if "\n" in c.get("message", ""))

    conventional_re = re.compile(
        r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?:", re.IGNORECASE
    )
    conventional_count = sum(1 for c in window if conventional_re.match(c.get("message", "")))

    _test_re = re.compile(r"\btest[s]?\b", re.IGNORECASE)
    test_count = sum(1 for c in window if _test_re.search(c.get("message", "")))

    bullets_count = sum(
        1 for c in window
        if "- " in c.get("message", "") or "* " in c.get("message", "")
    )

    # Inter-commit burst features
    sorted_commits = sorted(window, key=lambda c: _parse_dt(c.get("created_at")) or datetime.min)
    inter_commit_hours = []
    for i in range(1, len(sorted_commits)):
        dt1 = _parse_dt(sorted_commits[i-1].get("created_at"))
        dt2 = _parse_dt(sorted_commits[i].get("created_at"))
        if dt1 and dt2:
            hours = (dt2 - dt1).total_seconds() / 3600.0
            inter_commit_hours.append(hours)

    mean_inter_hours = sum(inter_commit_hours) / len(inter_commit_hours) if inter_commit_hours else 0.0
    burst_count = sum(1 for h in inter_commit_hours if h <= 2.0)
    frac_burst = burst_count / len(inter_commit_hours) if inter_commit_hours else 0.0

    # [FIX: file sampling] Compute only over commits with file_sampled=True
    sampled_commits = [c for c in window if c.get("file_sampled")]
    file_sample_count = len(sampled_commits)
    if sampled_commits:
        sampled_with_impl = sum(1 for c in sampled_commits if c.get("has_impl_file"))
        sampled_with_both = sum(
            1 for c in sampled_commits
            if c.get("has_impl_file") and c.get("has_test_file")
        )
        sampled_test_cowrite = (
            sampled_with_both / sampled_with_impl if sampled_with_impl > 0 else 0.0
        )
    else:
        sampled_test_cowrite = 0.0

    return {
        "commit_count":                   len(window),
        "mean_message_length":            round(sum(msg_lengths) / len(msg_lengths), 2),
        "active_weeks":                   active_weeks,
        "repos_touched":                  repos,
        "mean_commits_per_active_week":   round(len(window) / max(active_weeks, 1), 2),
        "frac_multiline":                 round(multiline_count / len(window), 3),
        "frac_conventional":              round(conventional_count / len(window), 3),
        "frac_mentions_test":             round(test_count / len(window), 3),
        "frac_has_bullets":               round(bullets_count / len(window), 3),
        "mean_inter_commit_hours":        round(mean_inter_hours, 2),
        "frac_burst_commits":             round(frac_burst, 3),
        "sampled_test_cowrite_rate":      round(sampled_test_cowrite, 3),
        "file_sample_count":              file_sample_count,
        "mean_pr_body_length":            0.0,  # overwritten by PR features below
        "frac_pr_has_body":               0.0,
    }


def _window_pr_features(prs, after, before=None):
    """Compute PR features for PRs in [after, before)."""
    window = []
    for pr in prs:
        dt = _parse_dt(pr.get("created_at"))
        if dt is None:
            continue
        if dt < after:
            continue
        if before and dt >= before:
            continue
        window.append(pr)

    if not window:
        return {"mean_pr_body_length": 0.0, "frac_pr_has_body": 0.0}

    body_lengths = [pr.get("body_length", 0) for pr in window]
    mean_body = sum(body_lengths) / len(body_lengths)
    frac_has_body = sum(1 for bl in body_lengths if bl > 50) / len(body_lengths)

    return {
        "mean_pr_body_length": round(mean_body, 2),
        "frac_pr_has_body": round(frac_has_body, 3),
    }


def stage4_features(positives, negatives, all_data):
    """Extract pre/post features and compute deltas.

    [FIX: per-account split] High-confidence positives use first_marker_date;
    low-confidence fall back to global cutoff.
    [FIX: symmetric filter] Both-window threshold applied to ALL accounts.
    [FIX: dedup] Commits deduplicated by SHA before feature extraction.
    """
    print("\n=== STAGE 4: Feature extraction ===")
    rows = []
    skipped_both_window = 0

    for login, data in all_data.items():
        if data.get("error"):
            print(f"  {login}: skipped ({data['error']})")
            continue

        # [FIX: dedup] Deduplicate commits across repos
        commits = _deduplicate_commits(data.get("commits", []))
        prs = data.get("prs", [])
        is_positive = login in positives
        label = 1 if is_positive else 0

        # [FIX: per-account split] Only use marker date for high-confidence
        # positives (GH Archive co-author).  Code Search positives use global
        # cutoff because repo.created_at != CLAUDE.md commit date.
        if is_positive:
            confidence = positives[login].get("marker_confidence", "low")
            if confidence == "high":
                marker_dt = _parse_dt(positives[login].get("first_marker_date", ""))
                if marker_dt and marker_dt > PRE_START:
                    account_post_start = marker_dt
                    account_pre_cutoff = marker_dt
                else:
                    account_post_start = POST_START
                    account_pre_cutoff = PRE_CUTOFF
            else:
                account_post_start = POST_START
                account_pre_cutoff = PRE_CUTOFF
        else:
            account_post_start = POST_START
            account_pre_cutoff = PRE_CUTOFF

        # [FIX: symmetric filter] Applied to positives AND negatives
        pre_count  = _count_commits_in_window(commits, after=PRE_START, before=account_pre_cutoff)
        post_count = _count_commits_in_window(commits, after=account_post_start)

        if pre_count < MIN_PRE_COMMITS or post_count < MIN_POST_COMMITS:
            skipped_both_window += 1
            print(f"  {login} (label={label}): SKIPPED both-window filter "
                  f"({pre_count} pre, {post_count} post)")
            continue

        pre_commit_feats  = _window_commit_features(commits, after=PRE_START, before=account_pre_cutoff)
        post_commit_feats = _window_commit_features(commits, after=account_post_start)

        pre_pr_feats  = _window_pr_features(prs, after=PRE_START, before=account_pre_cutoff)
        post_pr_feats = _window_pr_features(prs, after=account_post_start)

        pre_commit_feats.update(pre_pr_feats)
        post_commit_feats.update(post_pr_feats)

        row = {"login": login, "label": label}

        # Include metadata for downstream stratification
        if is_positive:
            row["discovery_method"]  = positives[login].get("discovery_method", "")
            row["marker_confidence"] = positives[login].get("marker_confidence", "")
        else:
            row["discovery_method"]  = "negative"
            row["marker_confidence"] = ""

        for k, v in pre_commit_feats.items():
            row[f"pre_{k}"] = v
        for k, v in post_commit_feats.items():
            row[f"post_{k}"] = v
        for k in pre_commit_feats:
            row[f"delta_{k}"] = round(post_commit_feats[k] - pre_commit_feats[k], 3)

        rows.append(row)

        print(f"  {login} (label={label}, conf={row['marker_confidence'] or 'n/a'}): "
              f"pre={pre_count} commits, post={post_count} commits, "
              f"Δmsg_len={row['delta_mean_message_length']:+.1f}")

    feat_path = DATA_DIR / "classifier_subsample_features.csv"
    if rows:
        with open(feat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nFeatures saved → {feat_path}  ({len(rows)} rows × {len(rows[0])} cols)")
    else:
        print("No features extracted")

    if skipped_both_window:
        print(f"  ({skipped_both_window} accounts dropped by both-window filter)")

    return rows


# ---------------------------------------------------------------------------
# Login list persistence
# ---------------------------------------------------------------------------

def save_login_lists(positives, negatives):
    fields = ["login", "discovery_method", "first_marker_date", "marker_type", "marker_confidence"]
    for name, d in [("classifier_positive_logins", positives),
                    ("classifier_negative_logins", negatives)]:
        path = DATA_DIR / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(d.values())
        print(f"  {path.name}: {len(d)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Classifier subsample scraper  v2.1")
    print("=" * 60)

    if not ensure_gh_archive():
        print("WARNING: Some GH Archive hours failed to download. Continuing with available data.")

    # Stage 1 — positives
    positives = stage1a_code_search()
    positives.update(stage1b_gh_archive())
    print(f"\nTotal unique positives: {len(positives)}")
    if positives:
        high_conf = sum(1 for p in positives.values() if p.get("marker_confidence") == "high")
        low_conf  = sum(1 for p in positives.values() if p.get("marker_confidence") == "low")
        print(f"  high confidence: {high_conf}  (GH Archive co-author)")
        print(f"  low confidence:  {low_conf}  (Code Search, global cutoff)")

    # Stage 2 — negatives
    negatives = stage2_negatives(set(positives))
    print(f"Total negatives: {len(negatives)}")

    # Save login lists
    print("\n=== Saving login lists ===")
    save_login_lists(positives, negatives)

    # Stage 3 — scrape
    all_data = stage3_scrape_all(positives, negatives)

    # Stage 4 — features
    features = stage4_features(positives, negatives, all_data)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Positives found:       {len(positives)}")
    print(f"  Negatives found:       {len(negatives)}")
    print(f"  Accounts scraped:      {len(all_data)}")
    print(f"  Features rows:         {len(features)}")
    if features:
        pos_rows = sum(1 for r in features if r["label"] == 1)
        neg_rows = sum(1 for r in features if r["label"] == 0)
        print(f"    positive rows:       {pos_rows}")
        print(f"    negative rows:       {neg_rows}")
        all_commits = [r["pre_commit_count"] + r["post_commit_count"] for r in features]
        print(f"  Mean commits/account:  {sum(all_commits)/len(all_commits):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
