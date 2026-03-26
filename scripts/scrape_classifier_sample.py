#!/usr/bin/env python3
"""
Subsample test scraper for Claude Code user classifier.

Stages:
1a. Ground truth positives via GitHub Code Search (filename:CLAUDE.md)
1b. Ground truth positives via GH Archive (Co-Authored-By: Claude trailer)
2.  Negative candidates via GH Archive (active devs, no AI markers)
3.  Per-account deep scrape — profile, repos, commit history, PRs
4.  Feature extraction with pre/post temporal split

Fixes vs v1:
- GH Archive downloaded once and cached locally (was re-downloaded 3x)
- first_marker_date populated via follow-up /repos API call (was blank)
- Commits fetched via /repos/{owner}/{repo}/commits (full history, not 90-day events)
- PR data fetched via /repos/{owner}/{repo}/pulls (correct payload, not events)
- Explicit timeout on all urllib calls to prevent hang
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
CACHE_DIR = DATA_DIR / "classifier_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

API_DELAY = 0.35       # seconds between REST calls
MAX_RETRIES = 3
BACKOFF_BASE = 2
REQUEST_TIMEOUT = 15   # seconds — prevents hanging on slow responses

# GH Archive — single hour, streamed and cached to disk line-by-line
# Hour 3 (3am UTC) is the quietest hour — smallest file, ~20-40MB compressed
GH_ARCHIVE_DATE = "2025-01-15"
GH_ARCHIVE_HOUR = 3
GH_ARCHIVE_CACHE = DATA_DIR / f"gharchive_{GH_ARCHIVE_DATE}-{GH_ARCHIVE_HOUR}.jsonl"

# Temporal split
PRE_CUTOFF = datetime(2024, 1, 1)   # before this = pre-AI window
POST_START  = datetime(2024, 1, 1)  # on/after this = post-AI window

# Test-run caps
MAX_POSITIVES       = 50
MAX_NEGATIVES       = 50
SCRAPE_CAP_POSITIVE = 30   # accounts to deep-scrape in this test run
SCRAPE_CAP_NEGATIVE = 30
MAX_REPOS_PER_ACCT  = 5
MAX_COMMITS_PER_ACCT = 200  # per repo; capped at 2 repos = ~400 total
MAX_PRS_PER_ACCT    = 50

random.seed(42)

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "classifier-scraper/2.0",
    }


def gh_get(url):
    """GET a GitHub API URL with retry + backoff. Returns parsed JSON or None."""
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=_gh_headers())
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                wait = (BACKOFF_BASE ** attempt) * 2
                print(f"    rate-limited, waiting {wait}s (attempt {attempt+1})")
                time.sleep(wait)
            elif e.code == 404:
                return None   # not found — not an error worth retrying
            else:
                print(f"    HTTP {e.code} on {url}")
                return None
        except Exception as e:
            wait = (BACKOFF_BASE ** attempt)
            print(f"    request error ({e}), waiting {wait}s")
            time.sleep(wait)
    return None


def _sleep():
    time.sleep(API_DELAY)


# ---------------------------------------------------------------------------
# GH Archive helpers
# ---------------------------------------------------------------------------

def ensure_gh_archive():
    """Download and cache GH Archive hour to disk if not already present.

    Streams gzip line-by-line to avoid loading 500MB+ into memory.
    Returns True on success, False on failure.
    """
    if GH_ARCHIVE_CACHE.exists():
        print(f"GH Archive cache exists: {GH_ARCHIVE_CACHE}")
        return True

    url = f"https://data.gharchive.org/{GH_ARCHIVE_DATE}-{GH_ARCHIVE_HOUR}.json.gz"
    print(f"Streaming GH Archive: {url}")
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "classifier-scraper/2.0"})
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
    """Yield parsed events from cache one at a time — no full load into memory."""
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
# Stage 1a — Code Search for CLAUDE.md
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

            # Fetch repo metadata to get first_marker_date
            repo_full = item.get("repository", {}).get("full_name", "")
            repo_created = ""
            if repo_full:
                repo_meta = gh_get(f"https://api.github.com/repos/{repo_full}")
                _sleep()
                if repo_meta:
                    repo_created = repo_meta.get("created_at", "")

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
        }
        for login in sampled
    }
    print(f"  sampled {len(negatives)} negatives")
    return negatives


# ---------------------------------------------------------------------------
# Stage 3 — Per-account deep scrape
# ---------------------------------------------------------------------------

def _scrape_commits_for_repo(owner, repo_name):
    """Fetch up to MAX_COMMITS_PER_ACCT commits for one repo via /commits API."""
    commits = []
    url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/commits"
        f"?per_page=100&page=1"
    )
    page = 1
    while len(commits) < MAX_COMMITS_PER_ACCT and page <= 2:
        result = gh_get(url)
        _sleep()
        if not result or not isinstance(result, list):
            break
        for c in result:
            commit_obj = c.get("commit", {})
            author_date = commit_obj.get("author", {}).get("date", "")
            stats = c.get("stats", {})  # not always present in list endpoint
            commits.append({
                "sha": c.get("sha", "")[:12],
                "message": commit_obj.get("message", "")[:200],
                "created_at": author_date,
                "repo": f"{owner}/{repo_name}",
                "additions": stats.get("additions"),
                "deletions": stats.get("deletions"),
            })
        if len(result) < 100:
            break
        page += 1
        url = (
            f"https://api.github.com/repos/{owner}/{repo_name}/commits"
            f"?per_page=100&page={page}"
        )
    return commits


def _scrape_prs_for_repo(owner, repo_name):
    """Fetch up to MAX_PRS_PER_ACCT closed/merged PRs for one repo."""
    url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
        f"?state=closed&per_page={MAX_PRS_PER_ACCT}&sort=updated&direction=desc"
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


def scrape_account(login):
    """Deep scrape one account. Returns from cache if available."""
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

    # Repos
    repos_raw = gh_get(
        f"https://api.github.com/users/{login}/repos?sort=updated&per_page=30"
    )
    _sleep()
    if not repos_raw or not isinstance(repos_raw, list):
        repos_raw = []

    repos_to_scrape = repos_raw[:MAX_REPOS_PER_ACCT]

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
            "has_claude_markers": has_claude_marker,  # label signal only
        })

        # Commits from this repo (full history, not 90-day events)
        commits = _scrape_commits_for_repo(owner_name, repo_name)
        data["commits"].extend(commits)

        # PRs from this repo
        prs = _scrape_prs_for_repo(owner_name, repo_name)
        data["prs"].extend(prs)

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
    for login in pos_logins:
        all_data[login] = scrape_account(login)

    print(f"Negatives to scrape: {len(neg_logins)}")
    for login in neg_logins:
        all_data[login] = scrape_account(login)

    raw_path = DATA_DIR / "classifier_sample_raw.json"
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
        }

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2] for c in window
                        if _parse_dt(c["created_at"])})
    repos        = len({c.get("repo", "") for c in window if c.get("repo")})
    msg_lengths  = [len(c.get("message", "")) for c in window]

    return {
        "commit_count":                 len(window),
        "mean_message_length":          round(sum(msg_lengths) / len(msg_lengths), 2),
        "active_weeks":                 active_weeks,
        "repos_touched":                repos,
        "mean_commits_per_active_week": round(len(window) / max(active_weeks, 1), 2),
    }


def stage4_features(positives, negatives, all_data):
    """Extract pre/post features and compute deltas."""
    print("\n=== STAGE 4: Feature extraction ===")
    rows = []

    # Pre-window: 2022-01-01 → 2024-01-01
    # Post-window: 2024-01-01 → now
    PRE_START = datetime(2022, 1, 1)

    for login, data in all_data.items():
        if data.get("error"):
            print(f"  {login}: skipped ({data['error']})")
            continue

        commits = data.get("commits", [])
        label   = 1 if login in positives else 0

        pre  = _window_commit_features(commits, after=PRE_START,  before=PRE_CUTOFF)
        post = _window_commit_features(commits, after=POST_START)

        row = {"login": login, "label": label}
        for k, v in pre.items():
            row[f"pre_{k}"] = v
        for k, v in post.items():
            row[f"post_{k}"] = v
        # Delta features
        for k in pre:
            row[f"delta_{k}"] = round(post[k] - pre[k], 2)

        rows.append(row)
        print(f"  {login} (label={label}): pre={pre['commit_count']} commits, "
              f"post={post['commit_count']} commits, "
              f"Δmsg_len={row['delta_mean_message_length']:+.1f}")

    feat_path = DATA_DIR / "classifier_sample_features.csv"
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

def save_login_lists(positives, negatives):
    fields = ["login", "discovery_method", "first_marker_date", "marker_type"]
    for name, d in [("classifier_positive_logins", positives),
                    ("classifier_negative_logins", negatives)]:
        path = DATA_DIR / f"{name}.csv"
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(d.values())
        print(f"  {path.name}: {len(d)} rows")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("Classifier subsample scraper  v2")
    print("=" * 60)

    # Download GH Archive once to disk (streamed, low memory)
    if not ensure_gh_archive():
        print("ERROR: Could not download GH Archive. Exiting.")
        return

    # Stage 1 — positives
    positives = stage1a_code_search()
    positives.update(stage1b_gh_archive())
    print(f"\nTotal unique positives: {len(positives)}")

    # Stage 2 — negatives (iterates cache file, not an in-memory list)
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
