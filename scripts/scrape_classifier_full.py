#!/usr/bin/env python3
"""
Full-scale scraper for Claude Code user classifier — v2.

Improvements over v1 (each marked with # [IMPROVEMENT: ...] in-line):

1.  Multi-hour GH Archive sampling — scans multiple hours across different
    days/times to dramatically improve recall for co-author discovery.
2.  Contributors API discovery — new Stage 1c queries the GitHub contributors
    endpoint for repos found via Code Search, catching users whose repos list
    "Claude" as a contributor even without a CLAUDE.md file.
3.  Per-account temporal split — positive accounts use their first_marker_date
    as the post-window start instead of a single global cutoff, avoiding
    contamination of the post window with pre-adoption commits.
4.  Symmetric both-window filter — positives are now also required to meet the
    minimum pre- and post-commit thresholds, preventing structural asymmetry
    between groups that the classifier could exploit.
5.  Matched negative sampling — after the both-window filter, negatives are
    propensity-matched to positives on account age and pre-period commit volume
    to reduce confounding with general developer activity.
6.  Explicit file-sample features — test_cowrite_rate is computed only over
    the subset of commits that were actually file-sampled, eliminating noise
    from unsampled commits with None values. Feature is renamed to
    sampled_test_cowrite_rate for clarity.
7.  Improved rate-limit handling — secondary rate limits use a 60-second floor
    and MAX_RETRIES is raised to 5 to survive longer throttle windows.
8.  Resume safety for positive scraping — a progress file tracks which
    positives have been scraped, so restarts skip already-completed accounts
    without re-checking hundreds of cache files.
9.  Feature leakage guard — has_claude_markers is no longer stored in raw data.
    A separate labeling file is written instead, fully decoupled from features.
10. Commit deduplication — commits are deduplicated by SHA across repos before
    feature extraction, preventing double-counting from forks.
11. Repo sort documented — repos are fetched sorted by 'pushed' (least
    recently pushed first) instead of 'updated' to reduce recency bias toward
    Claude-assisted repos.
"""

import os
import json
import csv
import time
import gzip
import random
import re
from datetime import datetime, timedelta
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
# ---------------------------------------------------------------------------
TEST_RUN = False

if TEST_RUN:
    CACHE_DIR               = DATA_DIR / "classifier_cache_test"
    MAX_POSITIVES           = 20
    MAX_NEGATIVES_TARGET    = 20
    MAX_NEGATIVES_CANDIDATES = 60
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

API_DELAY = 1.0
REQUEST_TIMEOUT = 15

# [IMPROVEMENT 7]: Increased retries (3→5) and added secondary rate-limit
# floor so the scraper survives longer GitHub throttle windows.
MAX_RETRIES = 5
BACKOFF_BASE = 2
SECONDARY_RATE_LIMIT_FLOOR = 60  # seconds — minimum wait on secondary limits

# [IMPROVEMENT 1]: Multiple GH Archive hours spread across different days and
# times of day.  A single hour captured a tiny slice of push activity; this
# samples 6 hours across 3 weekdays (morning, afternoon, evening UTC) for
# much broader coverage of co-author commits.
GH_ARCHIVE_HOURS = [
    ("2025-01-13", 9),   # Monday morning UTC
    ("2025-01-13", 18),  # Monday evening UTC
    ("2025-01-15", 3),   # Wednesday early morning UTC (original hour)
    ("2025-01-15", 14),  # Wednesday afternoon UTC
    ("2025-01-17", 11),  # Friday midday UTC
    ("2025-01-17", 21),  # Friday night UTC
]

# Temporal split defaults (positives override POST_START per-account)
PRE_CUTOFF  = datetime(2024, 1, 1)
POST_START  = datetime(2024, 1, 1)  # global fallback for negatives
PRE_START   = datetime(2022, 1, 1)

# Both-window threshold
MIN_PRE_COMMITS  = 10
MIN_POST_COMMITS = 10

# File sampling
MAX_FILE_SAMPLE_PER_ACCOUNT = 40
FILE_SAMPLE_RATE = 0.20
FILE_SAMPLE_DELAY = 1.0

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
        "User-Agent": "classifier-scraper-v2/1.0",
    }


def gh_get(url):
    """GET a GitHub API URL with retry + rate-limit-aware backoff.

    [IMPROVEMENT 7]: Secondary rate limits now sleep for at least
    SECONDARY_RATE_LIMIT_FLOOR (60s) instead of short exponential backoff
    (2/4/8s), and MAX_RETRIES is 5 to survive longer throttle windows.
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
                    # Primary rate limit — sleep until reset
                    wait = max(reset_ts - int(time.time()) + 5, 5)
                    print(f"    Rate limit exhausted. Sleeping {wait}s until reset "
                          f"(attempt {attempt+1}/{MAX_RETRIES})")
                    time.sleep(wait)
                else:
                    # [IMPROVEMENT 7]: Secondary rate limit — use a much longer
                    # floor (60s) because GitHub's secondary limits can persist
                    # for 60-120s.  The old 2/4/8s backoff often wasn't enough.
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
# GH Archive helpers
# [IMPROVEMENT 1]: Now handles multiple archive hours.
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
            req = urllib.request.Request(url, headers={"User-Agent": "classifier-scraper-v2/1.0"})
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
# [IMPROVEMENT 1]: Now scans all configured archive hours.
# ---------------------------------------------------------------------------

CLAUDE_COAUTHOR_RE = re.compile(
    r"Co-[Aa]uthored-[Bb]y:\s*Claude\s*<.*@anthropic\.com>",
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
                }
                print(f"    {login}")
                break

    print(f"GH Archive co-author scan: {len(positives)} accounts")
    return positives


# ---------------------------------------------------------------------------
# Stage 1c — Contributors API discovery
# [IMPROVEMENT 2]: New discovery method.  For repos found in Stage 1a, query
# the contributors endpoint to find other repos (by the same or different
# owners) that list a "Claude"-named contributor.  This catches users who
# don't have a CLAUDE.md but whose commit history includes Claude co-authored
# commits (which GitHub surfaces as a contributor).
# ---------------------------------------------------------------------------

def stage1c_contributors_api(existing_positives):
    """Discover additional positives by checking the contributors API.

    Strategy: for each repo already discovered in Stage 1a, check the
    contributors endpoint for a bot/user named 'Claude'.  Also search for
    repos owned by known positives that might not have CLAUDE.md but do have
    Claude as a listed contributor.
    """
    print("\n=== STAGE 1c: Contributors API discovery ===")
    new_positives = {}

    # Collect repos to check from existing positives' public repos
    logins_to_check = list(existing_positives.keys())[:50]  # cap API calls

    for login in logins_to_check:
        if len(existing_positives) + len(new_positives) >= MAX_POSITIVES:
            break

        repos_url = f"https://api.github.com/users/{login}/repos?sort=pushed&per_page=10"
        repos = gh_get(repos_url)
        _sleep()

        if not repos or not isinstance(repos, list):
            continue

        for repo in repos:
            repo_name = repo.get("name", "")
            owner = repo.get("owner", {}).get("login", login)

            # Check contributors for this repo
            contrib_url = f"https://api.github.com/repos/{owner}/{repo_name}/contributors?per_page=100"
            contributors = gh_get(contrib_url)
            _sleep()

            if not contributors or not isinstance(contributors, list):
                continue

            for contrib in contributors:
                contrib_login = (contrib.get("login") or "").lower()
                # Look for Claude bot contributor entries
                if "claude" in contrib_login and contrib.get("type") in ("Bot", "User"):
                    # The *repo owner* is the positive (they used Claude Code),
                    # not the Claude bot itself.
                    repo_owner = repo.get("owner", {}).get("login", "")
                    if (repo_owner
                            and repo_owner not in existing_positives
                            and repo_owner not in new_positives):
                        new_positives[repo_owner] = {
                            "login": repo_owner,
                            "discovery_method": "contributors_api",
                            "first_marker_date": repo.get("pushed_at", ""),
                            "marker_type": "Claude contributor",
                        }
                        print(f"    {repo_owner} (via {owner}/{repo_name})")
                    break  # no need to check remaining contributors

    print(f"Contributors API: {len(new_positives)} additional accounts")
    return new_positives


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
    page = 1
    while len(commits) < max_commits and page <= 2:
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            f"?per_page=100&page={page}"
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
                # [IMPROVEMENT 6]: These are explicitly None until file-sampled.
                # Feature extraction treats None differently from True/False.
                "has_test_file": None,
                "has_impl_file": None,
                "file_sampled": False,  # flag for whether this commit was sampled
            })
        if len(result) < 100:
            break
        page += 1
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
    """For ~20% of commits, fetch file changes.

    [IMPROVEMENT 6]: Each sampled commit gets file_sampled=True so feature
    extraction can compute test_cowrite_rate over only the sampled subset.
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

        # Mark this commit as having been file-sampled regardless of results
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

    [IMPROVEMENT 9]: has_claude_markers is NOT stored in the returned data.
    It is written to a separate labeling file to prevent accidental feature
    leakage.  The raw data used for feature extraction contains no label
    information.

    [IMPROVEMENT 11]: Repos are fetched sorted by 'pushed' ascending (oldest
    push first) instead of 'updated' descending.  This reduces recency bias
    that would over-represent Claude-assisted repos for positive accounts.
    """
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

    # [IMPROVEMENT 11]: Sort by 'pushed' ascending so we get a mix of old and
    # new repos rather than biasing toward the most recently touched ones.
    repos_raw = gh_get(
        f"https://api.github.com/users/{login}/repos?sort=pushed&direction=asc&per_page=30"
    )
    _sleep()
    if not repos_raw or not isinstance(repos_raw, list):
        repos_raw = []

    repos_to_scrape = repos_raw[:5]

    # [IMPROVEMENT 9]: Claude marker detection is separated into its own file.
    # We still check for markers (useful for validation) but store them in a
    # separate labeling CSV, NOT in the per-account JSON that feeds features.
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

        # [IMPROVEMENT 9]: Repo data stored WITHOUT has_claude_markers
        data["repos"].append({
            "name":       repo_name,
            "created_at": repo.get("created_at"),
            "language":   repo.get("language"),
            "size":       repo.get("size"),
        })

        commits = _scrape_commits_for_repo(owner_name, repo_name)
        _sample_commit_files(owner_name, repo_name, commits)
        data["commits"].extend(commits)

        prs = _scrape_prs_for_repo(owner_name, repo_name)
        data["prs"].extend(prs)

    # Write marker info to separate labeling file (append-safe)
    if claude_marker_repos:
        marker_path = DATA_DIR / f"{_RUN_TAG}_claude_markers.csv"
        write_header = not marker_path.exists()
        with open(marker_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["login", "repo", "marker_found"])
            for repo_full in claude_marker_repos:
                w.writerow([login, repo_full, True])

    cache_file.write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------------
# Stage 3a — Scrape positives with resume tracking
# [IMPROVEMENT 8]: Progress file tracks completed positives so restarts
# don't waste time re-checking hundreds of cache files.
# ---------------------------------------------------------------------------

def stage3a_scrape_positives(positives):
    """Scrape positive accounts with resume-safe progress tracking."""
    print("\n=== STAGE 3a: Scraping positives ===")

    progress_path = DATA_DIR / f"{_RUN_TAG}_positive_progress.json"
    completed = set()
    if progress_path.exists():
        completed = set(json.loads(progress_path.read_text()))
        print(f"  Resuming: {len(completed)} positives already scraped")

    all_data = {}
    pos_logins = list(positives)[:SCRAPE_CAP_POSITIVE]

    for i, login in enumerate(pos_logins):
        if login in completed:
            # Load from cache without re-scraping
            cache_file = CACHE_DIR / f"{login}.json"
            if cache_file.exists():
                with open(cache_file) as f:
                    all_data[login] = json.load(f)
            continue

        all_data[login] = scrape_account(login)
        completed.add(login)

        # [IMPROVEMENT 8]: Persist progress after each account
        progress_path.write_text(json.dumps(sorted(completed)))

        if (i + 1) % PROGRESS_INTERVAL == 0:
            print(f"Progress: {i+1}/{len(pos_logins)} positives scraped")

    print(f"  Positive scraping complete: {len(all_data)} accounts")
    return all_data


# ---------------------------------------------------------------------------
# Stage 3b — Dynamic negative scraping with both-window filter
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


def stage3b_scrape_negatives_dynamic(negative_candidates):
    """Scrape negatives dynamically until we have enough accepted.

    [IMPROVEMENT 4 context]: The both-window filter here is the same as the
    one applied to positives in stage4_features.  Both groups must have
    >= MIN_PRE_COMMITS and >= MIN_POST_COMMITS.
    """
    print("\n=== STAGE 3b: Dynamic negative scraping (both-window filter) ===")

    status_file = DATA_DIR / f"{_RUN_TAG}_negative_status.csv"
    existing_status = {}
    if status_file.exists():
        with open(status_file) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_status[row["login"]] = row["status"]

    accepted = []
    rejected = []

    status_fh = open(status_file, "a", newline="")
    status_writer = csv.DictWriter(status_fh, fieldnames=["login", "status"])
    if not existing_status:
        status_writer.writeheader()
        status_fh.flush()

    candidates_list = list(negative_candidates.keys())
    random.shuffle(candidates_list)

    for i, login in enumerate(candidates_list):
        if login in existing_status:
            status = existing_status[login]
            if status == "accepted":
                accepted.append(login)
            elif status == "rejected":
                rejected.append(login)
            continue

        data = scrape_account(login)

        if data.get("error"):
            existing_status[login] = "rejected"
            rejected.append(login)
            status_writer.writerow({"login": login, "status": "rejected"})
            status_fh.flush()
            continue

        commits = data.get("commits", [])
        pre_count  = _count_commits_in_window(commits, after=PRE_START, before=PRE_CUTOFF)
        post_count = _count_commits_in_window(commits, after=POST_START)

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

        if (len(accepted) + len(rejected)) % PROGRESS_INTERVAL == 0:
            print(f"Progress: {len(accepted)}/{MAX_NEGATIVES_TARGET} negatives accepted "
                  f"({len(rejected)} rejected so far)")

        if len(accepted) >= MAX_NEGATIVES_TARGET:
            print(f"Reached target of {MAX_NEGATIVES_TARGET} accepted negatives")
            break

    status_fh.close()

    print(f"\nNegative scraping complete:")
    print(f"  Accepted: {len(accepted)}")
    print(f"  Rejected: {len(rejected)}")

    return accepted[:MAX_NEGATIVES_TARGET]


# ---------------------------------------------------------------------------
# Stage 3c — Matched negative selection
# [IMPROVEMENT 5]: After both-window filtering, select negatives that are
# similar to positives on account age and pre-period commit volume.  This is
# a simple nearest-neighbour propensity match to reduce confounding.
# ---------------------------------------------------------------------------

def stage3c_match_negatives(positives_data, negatives_accepted, all_data):
    """Select negatives matched to positives on account age + pre commit count.

    For each positive, find the closest unmatched negative (Euclidean distance
    on normalised account_age_days and pre_commit_count).  If there are more
    negatives than positives, extras are dropped.  If fewer, all are kept.
    """
    print("\n=== STAGE 3c: Propensity-matched negative selection ===")

    def _account_features(login):
        data = all_data.get(login, {})
        profile = data.get("profile") or {}
        created = _parse_dt(profile.get("created_at"))
        age_days = (datetime(2025, 1, 15) - created).days if created else 0
        pre_commits = _count_commits_in_window(
            data.get("commits", []), after=PRE_START, before=PRE_CUTOFF
        )
        return age_days, pre_commits

    # Build positive feature vectors
    pos_features = []
    for login in positives_data:
        if all_data.get(login, {}).get("error"):
            continue
        age, pre = _account_features(login)
        pos_features.append((login, age, pre))

    # Build negative feature vectors
    neg_features = []
    for login in negatives_accepted:
        if all_data.get(login, {}).get("error"):
            continue
        age, pre = _account_features(login)
        neg_features.append((login, age, pre))

    if not pos_features or not neg_features:
        print("  Not enough data for matching; returning all accepted negatives")
        return negatives_accepted

    # Normalise features (z-score) for distance computation
    all_ages = [f[1] for f in pos_features + neg_features]
    all_pres = [f[2] for f in pos_features + neg_features]
    age_mean = sum(all_ages) / len(all_ages)
    age_std  = max((sum((a - age_mean)**2 for a in all_ages) / len(all_ages)) ** 0.5, 1)
    pre_mean = sum(all_pres) / len(all_pres)
    pre_std  = max((sum((p - pre_mean)**2 for p in all_pres) / len(all_pres)) ** 0.5, 1)

    def _norm(age, pre):
        return ((age - age_mean) / age_std, (pre - pre_mean) / pre_std)

    def _dist(a, b):
        return ((a[0] - b[0])**2 + (a[1] - b[1])**2) ** 0.5

    # Greedy nearest-neighbour matching
    matched = set()
    unmatched_neg = {login: _norm(age, pre) for login, age, pre in neg_features}

    for pos_login, pos_age, pos_pre in pos_features:
        if not unmatched_neg:
            break
        pos_norm = _norm(pos_age, pos_pre)
        best_login = min(unmatched_neg, key=lambda n: _dist(pos_norm, unmatched_neg[n]))
        matched.add(best_login)
        del unmatched_neg[best_login]

    matched_list = [login for login in negatives_accepted if login in matched]
    print(f"  Matched {len(matched_list)} negatives to {len(pos_features)} positives")

    # Report match quality
    matched_ages = [_account_features(l)[0] for l in matched_list]
    pos_ages     = [f[1] for f in pos_features]
    if matched_ages and pos_ages:
        print(f"  Positive mean account age: {sum(pos_ages)/len(pos_ages):.0f} days")
        print(f"  Matched neg mean age:      {sum(matched_ages)/len(matched_ages):.0f} days")

    return matched_list


# ---------------------------------------------------------------------------
# Stage 4 — Feature extraction
# [IMPROVEMENT 3]: Per-account temporal split for positives.
# [IMPROVEMENT 4]: Symmetric both-window filter for positives.
# [IMPROVEMENT 6]: test_cowrite_rate computed only over file-sampled commits.
# [IMPROVEMENT 10]: Commits deduplicated by SHA before feature extraction.
# ---------------------------------------------------------------------------

def _deduplicate_commits(commits):
    """[IMPROVEMENT 10]: Remove duplicate commits (same SHA from forked repos).

    If a developer has both a fork and the original repo in their top 5,
    shared commits would be counted twice.  Dedup on SHA prefix.
    """
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

    [IMPROVEMENT 6]: sampled_test_cowrite_rate is computed only over commits
    where file_sampled is True, so None values from unsampled commits don't
    pollute the denominator.
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

    # [IMPROVEMENT 6]: File-sample-aware test cowrite rate.
    # Only consider commits where file_sampled is True — these are the ones
    # where we actually fetched file lists.  Commits with file_sampled=False
    # have has_test_file=None and would silently drop out of the old logic,
    # making the denominator unpredictable.
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


def stage4_features(positives, negatives_matched, all_data):
    """Extract pre/post features and compute deltas.

    [IMPROVEMENT 3]: For positive accounts, the post window starts at their
    first_marker_date (the earliest evidence of Claude Code usage) instead of
    the global POST_START.  This prevents early post-period commits that were
    actually pre-adoption from diluting the signal.

    [IMPROVEMENT 4]: Positives are also filtered by the both-window threshold.
    Any positive without enough commits in both windows is dropped, ensuring
    symmetric treatment of both groups.

    [IMPROVEMENT 10]: Commits are deduplicated by SHA before feature extraction.
    """
    print("\n=== STAGE 4: Feature extraction ===")
    rows = []
    skipped_both_window = 0

    for login, data in all_data.items():
        if data.get("error"):
            print(f"  {login}: skipped ({data['error']})")
            continue

        # [IMPROVEMENT 10]: Deduplicate commits across repos
        commits = _deduplicate_commits(data.get("commits", []))
        prs = data.get("prs", [])
        is_positive = login in positives
        label = 1 if is_positive else 0

        # [IMPROVEMENT 3]: Per-account post window for positives.
        # Use the first_marker_date as the start of the post window so we
        # only measure behaviour AFTER the developer started using Claude Code.
        # Fall back to global POST_START if marker date is missing or invalid.
        if is_positive:
            marker_date_str = positives[login].get("first_marker_date", "")
            marker_dt = _parse_dt(marker_date_str)
            if marker_dt and marker_dt > PRE_START:
                account_post_start = marker_dt
                account_pre_cutoff = marker_dt  # pre ends where post begins
            else:
                account_post_start = POST_START
                account_pre_cutoff = PRE_CUTOFF
        else:
            account_post_start = POST_START
            account_pre_cutoff = PRE_CUTOFF

        # [IMPROVEMENT 4]: Apply both-window filter symmetrically to ALL
        # accounts (positives AND negatives).  The v1 code only filtered
        # negatives, creating structural asymmetry in activity distributions
        # that the classifier could exploit as a shortcut.
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
        for k, v in pre_commit_feats.items():
            row[f"pre_{k}"] = v
        for k, v in post_commit_feats.items():
            row[f"post_{k}"] = v
        # Delta features
        for k in pre_commit_feats:
            row[f"delta_{k}"] = round(post_commit_feats[k] - pre_commit_feats[k], 3)

        rows.append(row)

        print(f"  {login} (label={label}): pre={pre_count} commits, "
              f"post={post_count} commits, "
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

    if skipped_both_window:
        print(f"  ({skipped_both_window} accounts dropped by both-window filter)")

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


def save_login_lists(positives, negatives_matched):
    """Save positive and negative login lists."""
    print("\n=== Saving login lists ===")

    pos_path = DATA_DIR / "full_positive_logins.csv"
    fields = ["login", "discovery_method", "first_marker_date", "marker_type"]
    with open(pos_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(positives.values())
    print(f"  {pos_path.name}: {len(positives)} rows")

    neg_path = DATA_DIR / "full_negative_candidates.csv"
    with open(neg_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["login"])
        for login in negatives_matched:
            w.writerow([login])
    print(f"  {neg_path.name}: {len(negatives_matched)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Classifier full-scale scraper  v2.0")
    print("=" * 70)

    if not ensure_gh_archive():
        print("WARNING: Some GH Archive hours failed to download. Continuing with available data.")

    # --- Resume-safe Stage 1+2 ---
    pos_csv = DATA_DIR / f"{_RUN_TAG}_positive_logins.csv"
    neg_csv = DATA_DIR / f"{_RUN_TAG}_negative_candidates.csv"

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
        # Stage 1 — positives from multiple sources
        positives = stage1a_code_search()
        positives.update(stage1b_gh_archive())

        # [IMPROVEMENT 2]: Additional discovery via contributors API
        extra = stage1c_contributors_api(positives)
        positives.update(extra)

        print(f"\nTotal unique positives: {len(positives)}")

        # Stage 2 — negative candidates
        negatives_candidates = stage2_negatives(set(positives))
        print(f"Total negative candidates: {len(negatives_candidates)}")

        _save_csv(pos_csv,
                  list(positives.values()),
                  ["login", "discovery_method", "first_marker_date", "marker_type"])
        _save_csv(neg_csv,
                  list(negatives_candidates.values()),
                  ["login", "discovery_method", "first_marker_date", "marker_type"])
        print(f"  Saved login lists to disk for resume safety")

    # Stage 3a — scrape positives (with resume tracking)
    all_data = stage3a_scrape_positives(positives)

    # Stage 3b — scrape negatives dynamically with both-window filter
    negatives_accepted = stage3b_scrape_negatives_dynamic(negatives_candidates)

    for login in negatives_accepted:
        if login not in all_data:
            all_data[login] = scrape_account(login)

    # [IMPROVEMENT 5]: Stage 3c — match negatives to positives on observables
    negatives_matched = stage3c_match_negatives(positives, negatives_accepted, all_data)

    # Save final login lists
    save_login_lists(positives, negatives_matched)

    # Save raw data
    raw_path = DATA_DIR / f"classifier_{_RUN_TAG}_raw.json"
    raw_path.write_text(json.dumps(all_data, indent=2))
    print(f"Raw data saved → {raw_path}")

    # Stage 4 — features (uses matched negatives only)
    # Build filtered all_data containing only positives + matched negatives
    filtered_data = {}
    for login in positives:
        if login in all_data:
            filtered_data[login] = all_data[login]
    for login in negatives_matched:
        if login in all_data:
            filtered_data[login] = all_data[login]

    features = stage4_features(positives, negatives_matched, filtered_data)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Positives found:        {len(positives)}")
    print(f"  Negatives candidates:   {len(negatives_candidates)}")
    print(f"  Negatives accepted:     {len(negatives_accepted)}")
    print(f"  Negatives matched:      {len(negatives_matched)}")
    print(f"  Accounts scraped:       {len(all_data)}")
    print(f"  Features rows:          {len(features)}")
    if features:
        pos_rows = sum(1 for r in features if r["label"] == 1)
        neg_rows = sum(1 for r in features if r["label"] == 0)
        print(f"    positive rows:        {pos_rows}")
        print(f"    negative rows:        {neg_rows}")
        all_commits = [r["pre_commit_count"] + r["post_commit_count"] for r in features]
        print(f"  Mean commits/account:   {sum(all_commits)/len(all_commits):.1f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
