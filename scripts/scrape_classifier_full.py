#!/usr/bin/env python3
"""
Full-scale scraper for Claude Code user classifier.

Stages:
1a. Ground truth positives via GitHub Code Search (filename:CLAUDE.md, pages 1-10)
1b. Ground truth positives via GH Archive (Co-Authored-By: Claude trailer)
2.  Negative candidates via random GH Archive sampling (no activity threshold)
3.  Per-account deep scrape — profile, repos, commit history, PRs, file samples
4.  Feature extraction with pre/post temporal split

Key improvements over subsample:
- Negative sampling: random, no activity threshold; dynamic loop until 200 accepted
- Both-window filter: pre_commit_count >= 10 AND post_commit_count >= 10
- Commit message features: multiline, conventional, test mentions, bullets
- Inter-commit burst features: mean_inter_commit_hours, frac_burst_commits
- File list sampling: 20% of commits, fetch file changes, test_cowrite_rate
- PR body length features: mean_pr_body_length, frac_pr_has_body
- Resume safety: separate cache dir, login lists saved, negative status tracking
- Progress reporting: every 10 accounts
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

# ---------------------------------------------------------------------------
# TEST MODE — flip to False for the full overnight run
# True:  20 positives, 20 accepted negatives, 60 candidates, separate cache dir
# False: 200 positives, 200 accepted negatives, 400 candidates, full cache dir
# ---------------------------------------------------------------------------
TEST_RUN = False

if TEST_RUN:
    CACHE_DIR               = DATA_DIR / "classifier_cache_test"
    MAX_POSITIVES           = 20
    MAX_NEGATIVES_TARGET    = 20   # accepted negatives
    MAX_NEGATIVES_CANDIDATES = 60  # pool to draw from
    SCRAPE_CAP_POSITIVE     = 20
    _RUN_TAG                = "test"
    print("*** TEST MODE — caps: 20 positives, 20 negatives ***")
else:
    CACHE_DIR               = DATA_DIR / "classifier_cache_full"
    MAX_POSITIVES           = 200
    MAX_NEGATIVES_TARGET    = 200
    MAX_NEGATIVES_CANDIDATES = 400
    SCRAPE_CAP_POSITIVE     = 200
    _RUN_TAG                = "full"

CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_DELAY = 1.0        # seconds between REST calls — 1.0s → ~3600 req/hr, safely under 5000/hr limit
MAX_RETRIES = 3
BACKOFF_BASE = 2
REQUEST_TIMEOUT = 15   # seconds

# GH Archive — single hour, streamed and cached to disk line-by-line
GH_ARCHIVE_DATE = "2025-01-15"
GH_ARCHIVE_HOUR = 3
GH_ARCHIVE_CACHE = DATA_DIR / f"gharchive_{GH_ARCHIVE_DATE}-{GH_ARCHIVE_HOUR}.jsonl"

# Temporal split
PRE_CUTOFF = datetime(2024, 1, 1)
POST_START  = datetime(2024, 1, 1)
PRE_START   = datetime(2022, 1, 1)

# Both-window threshold
MIN_PRE_COMMITS  = 10
MIN_POST_COMMITS = 10

# File sampling
MAX_FILE_SAMPLE_PER_ACCOUNT = 40
FILE_SAMPLE_RATE = 0.20  # sample 20% of commits for file details
FILE_SAMPLE_DELAY = 1.0   # delay after each commit detail call — match API_DELAY

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
        "User-Agent": "classifier-scraper-full/1.0",
    }


def gh_get(url):
    """GET a GitHub API URL with retry + rate-limit-aware backoff.

    On 403/429:
      - Reads X-RateLimit-Reset header and sleeps until the reset timestamp
        if the bucket is empty (X-RateLimit-Remaining == 0).
      - Falls back to exponential backoff (2/4/8s) when the header is absent
        or the limit is not fully exhausted (e.g. secondary rate limits).
    Returns parsed JSON or None on unrecoverable error.
    """
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=_gh_headers())
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}

        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                # Try to read rate-limit headers from the error response
                remaining = e.headers.get("X-RateLimit-Remaining", "1")
                reset_ts  = e.headers.get("X-RateLimit-Reset",     "0")
                try:
                    remaining = int(remaining)
                    reset_ts  = int(reset_ts)
                except ValueError:
                    remaining, reset_ts = 1, 0

                if remaining == 0 and reset_ts > 0:
                    # Primary rate limit exhausted — sleep until the reset window
                    wait = max(reset_ts - int(time.time()) + 5, 5)
                    print(f"    Rate limit exhausted. Sleeping {wait}s until reset "
                          f"(attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    # Secondary rate limit or no header — exponential backoff
                    wait = (BACKOFF_BASE ** attempt) * 2
                    print(f"    Rate-limited (secondary). Waiting {wait}s "
                          f"(attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)

            elif e.code == 404:
                return None   # not found — not worth retrying
            elif e.code == 409:
                return None   # empty/unborn repo — not an error
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
# GH Archive helpers
# ---------------------------------------------------------------------------

def ensure_gh_archive():
    """Download and cache GH Archive hour to disk if not already present."""
    if GH_ARCHIVE_CACHE.exists():
        print(f"GH Archive cache exists: {GH_ARCHIVE_CACHE}")
        return True

    url = f"https://data.gharchive.org/{GH_ARCHIVE_DATE}-{GH_ARCHIVE_HOUR}.json.gz"
    print(f"Streaming GH Archive: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "classifier-scraper-full/1.0"})
        with urllib.request.urlopen(req, timeout=180) as resp:
            with gzip.open(resp, "rt", encoding="utf-8", errors="replace") as gz:
                with open(GH_ARCHIVE_CACHE, "w", encoding="utf-8") as out:
                    count = 0
                    for line in gz:
                        line = line.rstrip("\n")
                        if line:
                            out.write(line + "\n")
                            count += 1
        print(f"Streamed {count} events → {GH_ARCHIVE_CACHE}")
        return True
    except Exception as e:
        print(f"Failed to download GH Archive: {e}")
        return False


def iter_gh_archive():
    """Yield parsed events from cache one at a time."""
    if not GH_ARCHIVE_CACHE.exists():
        return
    with open(GH_ARCHIVE_CACHE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# ---------------------------------------------------------------------------
# Stage 1a — Code Search for CLAUDE.md (pages 1-10)
# ---------------------------------------------------------------------------

def stage1a_code_search():
    """Find accounts with CLAUDE.md via GitHub Code Search API (pages 1-10)."""
    print("\n=== STAGE 1a: Code Search for CLAUDE.md (pages 1-10) ===")
    positives = {}
    page = 1

    while len(positives) < MAX_POSITIVES and page <= 10:
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

            # repo.created_at is already present in the Code Search response —
            # no extra API call needed (saves ~200 calls in Stage 1a)
            repo_full    = item.get("repository", {}).get("full_name", "")
            repo_created = item.get("repository", {}).get("created_at", "")

            positives[login] = {
                "login": login,
                "discovery_method": "code_search",
                "first_marker_date": repo_created,
                "marker_type": "CLAUDE.md",
            }
            print(f"    {login}  ({repo_full}, created {repo_created[:10] or '?'})")

        page += 1

    print(f"Code search: {len(positives)} unique user accounts")
    return positives


# ---------------------------------------------------------------------------
# Stage 1b — GH Archive co-author scan
# ---------------------------------------------------------------------------

CLAUDE_COAUTHOR_RE = re.compile(
    r"Co-[Aa]uthored-[Bb]y:\s*Claude\s*<.*@anthropic\.com>",
    re.IGNORECASE,
)


def stage1b_gh_archive():
    """Find accounts with Co-Authored-By: Claude in GH Archive PushEvents."""
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
                }
                print(f"    {login}")
                break

    print(f"GH Archive co-author scan: {len(positives)} accounts")
    return positives


# ---------------------------------------------------------------------------
# Stage 2 — Negative candidates (random sampling, no activity threshold)
# ---------------------------------------------------------------------------

def stage2_negatives(positive_logins):
    """Sample random developers from GH Archive (no activity threshold)."""
    print("\n=== STAGE 2: Negative candidate discovery (random sampling) ===")
    all_actors = set()

    for event in iter_gh_archive():
        if event.get("type") == "PushEvent":
            login = event.get("actor", {}).get("login", "")
            if login and login not in positive_logins:
                all_actors.add(login)

    print(f"  {len(all_actors)} unique actors in GH Archive (excluding positives)")

    # Sample up to MAX_NEGATIVES_CANDIDATES randomly
    candidates = list(all_actors)
    sampled = random.sample(candidates, min(MAX_NEGATIVES_CANDIDATES, len(candidates)))
    
    negatives = {
        login: {
            "login": login,
            "discovery_method": "gh_archive_random",
            "first_marker_date": None,
            "marker_type": None,
        }
        for login in sampled
    }
    print(f"  sampled {len(negatives)} negative candidates (pool)")
    return negatives


# ---------------------------------------------------------------------------
# Stage 3 — Per-account deep scrape with file sampling
# ---------------------------------------------------------------------------

def _scrape_commits_for_repo(owner, repo_name, max_commits=200):
    """Fetch up to max_commits commits for one repo via /commits API."""
    commits = []
    url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/commits"
        f"?per_page=100&page=1"
    )
    page = 1
    while len(commits) < max_commits and page <= 2:
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
                "has_test_file": None,  # populated by file sampling
                "has_impl_file": None,
            })
        if len(result) < 100:
            break
        page += 1
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            f"?per_page=100&page={page}"
        )
    return commits


def _scrape_prs_for_repo(owner, repo_name, max_prs=100):
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
    """For 20% of commits, fetch file changes. Populate has_test_file, has_impl_file."""
    if not commits:
        return
    
    # Determine which commits to sample
    sample_indices = random.sample(
        range(len(commits)),
        min(MAX_FILE_SAMPLE_PER_ACCOUNT, max(1, int(len(commits) * FILE_SAMPLE_RATE)))
    )
    
    impl_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"}
    test_keywords = {"test", "spec"}
    
    for idx in sample_indices:
        commit = commits[idx]
        sha = commit["sha"]
        
        # Fetch commit detail to get files changed
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
        detail = gh_get(url)
        time.sleep(FILE_SAMPLE_DELAY)
        
        if not detail or "files" not in detail:
            continue
        
        files = detail.get("files", [])
        has_test = False
        has_impl = False
        
        for file_obj in files:
            filename = file_obj.get("filename", "").lower()
            # Check for test file
            if any(kw in filename for kw in test_keywords):
                has_test = True
            # Check for impl file
            if any(filename.endswith(ext) for ext in impl_extensions):
                has_impl = True
        
        commit["has_test_file"] = has_test
        commit["has_impl_file"] = has_impl


def scrape_account(login):
    """Deep scrape one account. Returns from cache if available."""
    cache_file = CACHE_DIR / f"{login}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    data = {"login": login, "profile": None, "repos": [], "commits": [], "prs": [], "error": None}

    # Profile
    profile = gh_get(f"https://api.github.com/users/{login}")
    _sleep()
    if profile is None:
        data["error"] = "profile fetch failed"
        cache_file.write_text(json.dumps(data, indent=2))
        return data

    data["profile"] = {
        "login":        profile.get("login"),
        "created_at":   profile.get("created_at"),
        "location":     profile.get("location"),
        "public_repos": profile.get("public_repos"),
    }

    # Repos
    repos_raw = gh_get(
        f"https://api.github.com/users/{login}/repos?sort=updated&per_page=30"
    )
    _sleep()
    if not repos_raw or not isinstance(repos_raw, list):
        repos_raw = []

    repos_to_scrape = repos_raw[:5]  # MAX_REPOS_PER_ACCT

    for repo in repos_to_scrape:
        repo_name  = repo.get("name", "")
        owner_name = repo.get("owner", {}).get("login", login)

        # Top-level file tree — for labeling only, NOT features
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

        data["repos"].append({
            "name":               repo_name,
            "created_at":         repo.get("created_at"),
            "language":           repo.get("language"),
            "size":               repo.get("size"),
            "has_claude_markers": has_claude_marker,
        })

        # Commits from this repo (full history)
        commits = _scrape_commits_for_repo(owner_name, repo_name)
        
        # Sample 20% of commits for file details
        _sample_commit_files(owner_name, repo_name, commits)
        
        data["commits"].extend(commits)

        # PRs from this repo
        prs = _scrape_prs_for_repo(owner_name, repo_name)
        data["prs"].extend(prs)

    cache_file.write_text(json.dumps(data, indent=2))
    return data


def stage3_scrape_account(login):
    """Scrape one account and return data."""
    return scrape_account(login)


# ---------------------------------------------------------------------------
# Stage 3b — Dynamic negative scraping with both-window filter
# ---------------------------------------------------------------------------

def stage3_scrape_negatives_dynamic(negative_candidates):
    """
    Scrape negatives dynamically until we have MAX_NEGATIVES_TARGET accepted.
    Both-window filter: pre_commit_count >= 10 AND post_commit_count >= 10
    """
    print("\n=== STAGE 3b: Dynamic negative scraping (both-window filter) ===")
    
    # Load existing status if resuming
    status_file = DATA_DIR / f"{_RUN_TAG}_negative_status.csv"
    existing_status = {}
    if status_file.exists():
        with open(status_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_status[row["login"]] = row["status"]
    
    accepted = []
    rejected = []

    # Open status file in append mode so each decision is durable immediately.
    # On resume, already-processed logins are in existing_status and skipped.
    status_fh = open(status_file, "a", newline="")
    status_writer = csv.DictWriter(status_fh, fieldnames=["login", "status"])
    if not status_file.exists() or status_file.stat().st_size == 0:
        status_writer.writeheader()
        status_fh.flush()

    # Shuffle candidates for random order
    candidates_list = list(negative_candidates.keys())
    random.shuffle(candidates_list)

    for i, login in enumerate(candidates_list):
        # Check if already processed (resume path)
        if login in existing_status:
            status = existing_status[login]
            if status == "accepted":
                accepted.append(login)
            elif status == "rejected":
                rejected.append(login)
            continue

        # Scrape account
        data = stage3_scrape_account(login)

        if data.get("error"):
            existing_status[login] = "rejected"
            rejected.append(login)
            status_writer.writerow({"login": login, "status": "rejected"})
            status_fh.flush()
            continue

        # Count commits in each window.
        # Pre window must respect PRE_START lower bound (2022-01-01) to stay
        # consistent with feature extraction — otherwise old accounts with
        # commits only in 2018-2019 pass the filter but yield pre_count=0 in features.
        commits = data.get("commits", [])
        pre_count = sum(
            1 for c in commits
            if (dt := _parse_dt(c.get("created_at"))) and PRE_START <= dt < PRE_CUTOFF
        )
        post_count = sum(
            1 for c in commits
            if (dt := _parse_dt(c.get("created_at"))) and dt >= POST_START
        )
        
        # Apply both-window filter
        if pre_count >= MIN_PRE_COMMITS and post_count >= MIN_POST_COMMITS:
            existing_status[login] = "accepted"
            accepted.append(login)
            status_writer.writerow({"login": login, "status": "accepted"})
            status_fh.flush()
            print(f"  {login}: ACCEPTED ({pre_count} pre, {post_count} post)")
        else:
            existing_status[login] = "rejected"
            rejected.append(login)
            status_writer.writerow({"login": login, "status": "rejected"})
            status_fh.flush()
            print(f"  {login}: rejected ({pre_count} pre, {post_count} post)")

        # Progress reporting
        if (len(accepted) + len(rejected)) % PROGRESS_INTERVAL == 0:
            print(f"Progress: {len(accepted)}/{MAX_NEGATIVES_TARGET} negatives accepted "
                  f"({len(rejected)} rejected so far)")

        # Stop when we have enough accepted
        if len(accepted) >= MAX_NEGATIVES_TARGET:
            print(f"Reached target of {MAX_NEGATIVES_TARGET} accepted negatives")
            break

    status_fh.close()
    
    print(f"\nNegative scraping complete:")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")
    
    return accepted[:MAX_NEGATIVES_TARGET]


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


def _window_commit_features(commits, after, before=None):
    """Compute commit features for commits in [after, before)."""
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
            "commit_count":             0,
            "mean_message_length":      0.0,
            "active_weeks":             0,
            "repos_touched":            0,
            "mean_commits_per_active_week": 0.0,
            "frac_multiline":           0.0,
            "frac_conventional":        0.0,
            "frac_mentions_test":       0.0,
            "frac_has_bullets":         0.0,
            "mean_inter_commit_hours":  0.0,
            "frac_burst_commits":       0.0,
            "test_cowrite_rate":        0.0,
            "mean_pr_body_length":      0.0,
            "frac_pr_has_body":         0.0,
        }

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2] for c in window
                        if _parse_dt(c["created_at"])})
    repos        = len({c.get("repo", "") for c in window if c.get("repo")})
    msg_lengths  = [len(c.get("message", "")) for c in window]
    
    # Commit message structure features
    multiline_count = sum(1 for c in window if "\n" in c.get("message", ""))
    
    conventional_re = re.compile(r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?:", re.IGNORECASE)
    conventional_count = sum(1 for c in window if conventional_re.match(c.get("message", "")))
    
    _test_re = re.compile(r"\btest[s]?\b", re.IGNORECASE)
    test_count = sum(1 for c in window if _test_re.search(c.get("message", "")))
    
    bullets_count = sum(1 for c in window if "- " in c.get("message", "") or "* " in c.get("message", ""))
    
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
    
    # Test cowrite rate (commits with both test and impl files)
    commits_with_impl = sum(1 for c in window if c.get("has_impl_file"))
    commits_with_both = sum(1 for c in window if c.get("has_impl_file") and c.get("has_test_file"))
    test_cowrite = commits_with_both / commits_with_impl if commits_with_impl > 0 else 0.0

    return {
        "commit_count":                 len(window),
        "mean_message_length":          round(sum(msg_lengths) / len(msg_lengths), 2),
        "active_weeks":                 active_weeks,
        "repos_touched":                repos,
        "mean_commits_per_active_week": round(len(window) / max(active_weeks, 1), 2),
        "frac_multiline":               round(multiline_count / len(window), 3),
        "frac_conventional":            round(conventional_count / len(window), 3),
        "frac_mentions_test":           round(test_count / len(window), 3),
        "frac_has_bullets":             round(bullets_count / len(window), 3),
        "mean_inter_commit_hours":      round(mean_inter_hours, 2),
        "frac_burst_commits":           round(frac_burst, 3),
        "test_cowrite_rate":            round(test_cowrite, 3),
        "mean_pr_body_length":          0.0,  # computed separately below
        "frac_pr_has_body":             0.0,
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
    mean_body = sum(body_lengths) / len(body_lengths) if body_lengths else 0.0
    frac_has_body = sum(1 for bl in body_lengths if bl > 50) / len(body_lengths) if body_lengths else 0.0
    
    return {
        "mean_pr_body_length": round(mean_body, 2),
        "frac_pr_has_body": round(frac_has_body, 3),
    }


def stage4_features(positives, negatives_accepted, all_data):
    """Extract pre/post features and compute deltas."""
    print("\n=== STAGE 4: Feature extraction ===")
    rows = []

    for login, data in all_data.items():
        if data.get("error"):
            print(f"  {login}: skipped ({data['error']})")
            continue

        commits = data.get("commits", [])
        prs = data.get("prs", [])
        label = 1 if login in positives else 0

        pre_commit_feats  = _window_commit_features(commits, after=PRE_START,  before=PRE_CUTOFF)
        post_commit_feats = _window_commit_features(commits, after=POST_START)
        
        pre_pr_feats  = _window_pr_features(prs, after=PRE_START,  before=PRE_CUTOFF)
        post_pr_feats = _window_pr_features(prs, after=POST_START)
        
        # Merge PR features into commit features
        pre_commit_feats.update(pre_pr_feats)
        post_commit_feats.update(post_pr_feats)

        row = {"login": login, "label": label}
        for k, v in pre_commit_feats.items():
            row[f"pre_{k}"] = v
        for k, v in post_commit_feats.items():
            row[f"post_{k}"] = v
        # Delta features
        for k in pre_commit_feats:
            row[f"delta_{k}"] = round(post_commit_feats[k] - pre_commit_feats[k], 3)

        rows.append(row)
        
        pre_total = pre_commit_feats["commit_count"]
        post_total = post_commit_feats["commit_count"]
        print(f"  {login} (label={label}): pre={pre_total} commits, "
              f"post={post_total} commits, "
              f"Δmsg_len={row['delta_mean_message_length']:+.1f}")

    feat_path = DATA_DIR / f"classifier_{_RUN_TAG}_features.csv"
    if rows:
        with open(feat_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nFeatures saved → {feat_path}  ({len(rows)} rows × {len(rows[0])} cols)")
    else:
        print("No features extracted")

    return rows


# ---------------------------------------------------------------------------
# Login list persistence
# ---------------------------------------------------------------------------

def _save_csv(path, rows, fieldnames):
    """Write a list of dicts to a CSV file."""
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def save_login_lists(positives, negatives_accepted):
    """Save positive and negative login lists."""
    print("\n=== Saving login lists ===")
    
    # Positives
    pos_path = DATA_DIR / "full_positive_logins.csv"
    fields = ["login", "discovery_method", "first_marker_date", "marker_type"]
    with open(pos_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(positives.values())
    print(f"  {pos_path.name}: {len(positives)} rows")
    
    # Negatives (accepted only)
    neg_path = DATA_DIR / "full_negative_candidates.csv"
    with open(neg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["login"])
        for login in negatives_accepted:
            w.writerow([login])
    print(f"  {neg_path.name}: {len(negatives_accepted)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Classifier full-scale scraper  v1.0")
    print("=" * 70)

    # Download GH Archive once to disk (streamed, low memory)
    if not ensure_gh_archive():
        print("ERROR: Could not download GH Archive. Exiting.")
        return

    # --- Resume-safe Stage 1+2 ---
    # If login lists already exist on disk, skip discovery and reload from CSV.
    # Files are tagged with _RUN_TAG so test and full runs don't collide.
    pos_csv   = DATA_DIR / f"{_RUN_TAG}_positive_logins.csv"
    neg_csv   = DATA_DIR / f"{_RUN_TAG}_negative_candidates.csv"

    if pos_csv.exists() and neg_csv.exists():
        print(f"\nResume: loading existing login lists from disk...")
        positives = {}
        with open(pos_csv, newline="") as f:
            for row in csv.DictReader(f):
                positives[row["login"]] = row
        negatives_candidates = {}
        with open(neg_csv, newline="") as f:
            for row in csv.DictReader(f):
                negatives_candidates[row["login"]] = row
        print(f"  Loaded {len(positives)} positives, {len(negatives_candidates)} negative candidates")
    else:
        # Stage 1 — positives
        positives = stage1a_code_search()
        positives.update(stage1b_gh_archive())
        print(f"\nTotal unique positives: {len(positives)}")

        # Stage 2 — negative candidates (random pool, no activity filter)
        negatives_candidates = stage2_negatives(set(positives))
        print(f"Total negative candidates: {len(negatives_candidates)}")

        # Persist immediately so a restart can skip these expensive stages
        _save_csv(pos_csv,
                  list(positives.values()),
                  ["login", "discovery_method", "first_marker_date", "marker_type"])
        _save_csv(neg_csv,
                  list(negatives_candidates.values()),
                  ["login", "discovery_method", "first_marker_date", "marker_type"])
        print(f"  Saved login lists to disk for resume safety")

    # Stage 3 — scrape positives
    print("\n=== STAGE 3a: Scraping positives ===")
    all_data = {}
    pos_logins = list(positives)[:SCRAPE_CAP_POSITIVE]
    
    for i, login in enumerate(pos_logins):
        all_data[login] = stage3_scrape_account(login)
        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Progress: {i+1}/{len(pos_logins)} positives scraped")

    # Stage 3b — scrape negatives dynamically with both-window filter
    negatives_accepted = stage3_scrape_negatives_dynamic(negatives_candidates)
    
    for login in negatives_accepted:
        all_data[login] = scrape_account(login)

    # Save final login lists
    save_login_lists(positives, negatives_accepted)

    # Save raw data
    raw_path = DATA_DIR / f"classifier_{_RUN_TAG}_raw.json"
    raw_path.write_text(json.dumps(all_data, indent=2))
    print(f"Raw data saved → {raw_path}")

    # Stage 4 — features
    features = stage4_features(positives, negatives_accepted, all_data)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Positives found:       {len(positives)}")
    print(f"  Negatives candidates:  {len(negatives_candidates)}")
    print(f"  Negatives accepted:    {len(negatives_accepted)}")
    print(f"  Accounts scraped:      {len(all_data)}")
    print(f"  Features rows:         {len(features)}")
    if features:
        pos_rows = sum(1 for r in features if r["label"] == 1)
        neg_rows = sum(1 for r in features if r["label"] == 0)
        print(f"    positive rows:       {pos_rows}")
        print(f"    negative rows:       {neg_rows}")
        all_commits = [r["pre_commit_count"] + r["post_commit_count"] for r in features]
        print(f"  Mean commits/account:  {sum(all_commits)/len(all_commits):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
