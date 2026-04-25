#!/usr/bin/env python3
"""
Expanded positive-account scraper — v1.0

Two goals:
  1. Find Claude Code users who adopted EARLIER (2023–2025) by prioritising
     older date chunks in the commit search. The original scraper filled its
     cap from the 2026 chunk, leaving the earlier windows untouched. This
     gives us temporal spread for a proper staggered DiD.

  2. Find Aider users. Aider leaves an identical co-author trail format and
     should 2-3x the positive pool. All accounts are tagged with tool_type
     so we can pool or compare.

Accounts already in classifier_full_features.csv are skipped automatically.

Output:
  data/expansion_raw.json           — per-account scraped data (keyed by login)
  data/expansion_features.csv       — identical schema to classifier_full_features.csv
                                       + tool_type column (claude / aider)
  data/expansion_progress.json      — resume checkpoint (list of completed logins)
  data/expansion_markers.csv        — marker artefacts found during scraping

Run with:
  uv run scripts/scrape_expanded_positives.py

Leave it running — it checkpoints after every account and resumes cleanly on
restart. Ctrl-C at any time; progress is not lost.
"""

import os
import json
import csv
import time
import re
import random
from datetime import datetime
from pathlib import Path
import urllib.request
import urllib.error
import socket

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

PROJECT_ROOT = Path("/home/avery/projects/ai_productivity_analysis")
DATA_DIR     = PROJECT_ROOT / "data"
CACHE_DIR    = DATA_DIR / "expansion_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# How many new positives to collect total (across both tools)
MAX_NEW_POSITIVES = 300

API_DELAY     = 1.0
REQUEST_TIMEOUT = 15
MAX_RETRIES   = 5
BACKOFF_BASE  = 2
SECONDARY_RATE_LIMIT_FLOOR = 60
NETWORK_RETRY_FLOOR        = 120
NETWORK_MAX_RETRIES        = 8
CONSECUTIVE_NETWORK_FAIL_LIMIT = 5
CIRCUIT_BREAKER_PAUSE      = 300

# Temporal windows — same as original scraper
PRE_START  = datetime(2022, 1, 1)
PRE_CUTOFF = datetime(2024, 1, 1)
POST_START = datetime(2024, 1, 1)
MIN_PRE_COMMITS  = 10
MIN_POST_COMMITS = 10

# File sampling
MAX_FILE_SAMPLE_PER_ACCOUNT = 40
FILE_SAMPLE_RATE  = 0.20
FILE_SAMPLE_DELAY = 1.0

# Date sanity bounds
TOOL_LAUNCH  = datetime(2023, 6, 1)   # Aider public + Claude Code beta
DATE_MAX     = datetime(2027, 1, 1)

random.seed(42)


# ---------------------------------------------------------------------------
# Commit search queries
# ---------------------------------------------------------------------------
# Claude: prioritise 2023–2025 first (the existing data is all from 2026).
# Aider: aider@aider.chat is the canonical co-author email.
# Each tuple: (query_string, label, tool_type)

COMMIT_SEARCH_QUERIES = [
    # ── Claude Code — oldest chunks first ──────────────────────────────────
    ("noreply%40anthropic.com+committer-date%3A2023-11-01..2023-12-31",
     "Claude (2023-Q4)",    "claude"),
    ("noreply%40anthropic.com+committer-date%3A2024-01-01..2024-06-30",
     "Claude (2024-H1)",    "claude"),
    ("noreply%40anthropic.com+committer-date%3A2024-07-01..2024-12-31",
     "Claude (2024-H2)",    "claude"),
    ("noreply%40anthropic.com+committer-date%3A2025-01-01..2025-06-30",
     "Claude (2025-H1)",    "claude"),
    ("noreply%40anthropic.com+committer-date%3A2025-07-01..2025-12-31",
     "Claude (2025-H2)",    "claude"),
    ("claude.ai%2Fcode",
     "Claude (footer)",     "claude"),

    # ── Aider — date-chunked the same way ──────────────────────────────────
    ("aider%40aider.chat+committer-date%3A2023-06-01..2023-12-31",
     "Aider (2023)",        "aider"),
    ("aider%40aider.chat+committer-date%3A2024-01-01..2024-06-30",
     "Aider (2024-H1)",     "aider"),
    ("aider%40aider.chat+committer-date%3A2024-07-01..2024-12-31",
     "Aider (2024-H2)",     "aider"),
    ("aider%40aider.chat+committer-date%3A2025-01-01..2025-06-30",
     "Aider (2025-H1)",     "aider"),
    ("aider%40aider.chat+committer-date%3A2025-07-01..2025-12-31",
     "Aider (2025-H2)",     "aider"),
    ("aider%40aider.chat+committer-date%3A2026-01-01..2026-12-31",
     "Aider (2026)",        "aider"),
]


# ---------------------------------------------------------------------------
# Marker strip regexes (prevent label leakage into features)
# ---------------------------------------------------------------------------
CLAUDE_COAUTHOR_RE = re.compile(
    r"Co-[Aa]uthored-[Bb]y:\s*Claude[\s\w.]*<?noreply@anthropic\.com>?",
    re.IGNORECASE,
)
CLAUDE_FOOTER_RE = re.compile(
    r"🤖\s*Generated with.*?claude\.ai/code[^\n]*",
    re.IGNORECASE,
)
AIDER_COAUTHOR_RE = re.compile(
    r"Co-[Aa]uthored-[Bb]y:\s*aider\s*<?aider@aider\.chat>?",
    re.IGNORECASE,
)


def _strip_markers(msg):
    msg = CLAUDE_COAUTHOR_RE.sub("", msg)
    msg = CLAUDE_FOOTER_RE.sub("", msg)
    msg = AIDER_COAUTHOR_RE.sub("", msg)
    return msg.strip()


# ---------------------------------------------------------------------------
# HTTP helpers (same pattern as original scraper)
# ---------------------------------------------------------------------------

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "expansion-scraper-v1/1.0",
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
    headers = {**_gh_headers(), **(extra_headers or {})}
    last_network_error = None

    for attempt in range(NETWORK_MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                body = resp.read().decode("utf-8")
                return json.loads(body) if body else {}

        except urllib.error.HTTPError as e:
            if e.code in (403, 429):
                remaining = e.headers.get("X-RateLimit-Remaining", "1")
                reset_ts  = e.headers.get("X-RateLimit-Reset", "0")
                try:
                    remaining = int(remaining)
                    reset_ts  = int(reset_ts)
                except ValueError:
                    remaining, reset_ts = 1, 0

                if remaining == 0 and reset_ts > 0:
                    wait = max(reset_ts - int(time.time()) + 5, 5)
                    print(f"    Rate limit exhausted. Sleeping {wait}s until reset")
                    time.sleep(wait)
                else:
                    wait = max(SECONDARY_RATE_LIMIT_FLOOR, (BACKOFF_BASE ** attempt) * 2)
                    print(f"    Secondary rate limit. Waiting {wait}s")
                    time.sleep(wait)

                if attempt >= MAX_RETRIES - 1:
                    print(f"    Failed after {MAX_RETRIES} rate-limit retries: {url}")
                    return None

            elif e.code in (404, 409):
                return None
            else:
                print(f"    HTTP {e.code} on {url}")
                return None

        except Exception as e:
            if _is_network_error(e):
                last_network_error = e
                wait = max(NETWORK_RETRY_FLOOR, (BACKOFF_BASE ** attempt) * 2)
                print(f"    Network error ({e}), waiting {wait}s "
                      f"(attempt {attempt+1}/{NETWORK_MAX_RETRIES})")
                time.sleep(wait)
            else:
                wait = (BACKOFF_BASE ** attempt)
                print(f"    Request error ({e}), waiting {wait}s")
                time.sleep(wait)
                if attempt >= MAX_RETRIES - 1:
                    return None

    if last_network_error:
        raise NetworkError(f"Network unavailable after {NETWORK_MAX_RETRIES} retries: "
                           f"{last_network_error}")
    return None


def _sleep():
    time.sleep(API_DELAY)


# ---------------------------------------------------------------------------
# Stage 1 — Discover positive accounts via commit search
# ---------------------------------------------------------------------------

def load_existing_logins():
    """Load logins already in the main dataset so we can skip them."""
    existing = set()
    feat_path = DATA_DIR / "classifier_full_features.csv"
    if feat_path.exists():
        with open(feat_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["login"])
    # Also check any previous expansion run
    exp_path = DATA_DIR / "expansion_features.csv"
    if exp_path.exists():
        with open(exp_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing.add(row["login"])
    print(f"  Existing accounts to skip: {len(existing)}")
    return existing


def stage1_commit_search(existing_logins):
    """Discover new positive accounts via GitHub commit search."""
    print("\n=== STAGE 1: Commit search (Claude + Aider) ===")
    positives = {}

    for query_str, label, tool_type in COMMIT_SEARCH_QUERIES:
        if len(positives) >= MAX_NEW_POSITIVES:
            print(f"  Cap reached ({MAX_NEW_POSITIVES}), stopping search")
            break

        print(f"\n  Query: {label}")
        page = 1

        while len(positives) < MAX_NEW_POSITIVES and page <= 10:
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
                if len(positives) >= MAX_NEW_POSITIVES:
                    break
                author = item.get("author") or {}
                login = author.get("login", "")
                if (not login
                        or author.get("type") != "User"
                        or login in positives
                        or login in existing_logins):
                    continue

                commit_date = (
                    item.get("commit", {}).get("committer", {}).get("date", "")
                    or item.get("commit", {}).get("author", {}).get("date", "")
                )
                try:
                    cdt = datetime.fromisoformat(
                        commit_date.replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                    if not (TOOL_LAUNCH <= cdt <= DATE_MAX):
                        continue
                except (ValueError, AttributeError):
                    continue

                positives[login] = {
                    "login":             login,
                    "discovery_method":  "commit_search_coauthor",
                    "first_marker_date": commit_date,
                    "marker_type":       f"commit_search: {label}",
                    "marker_confidence": "high",
                    "tool_type":         tool_type,
                }
                new_this_page += 1
                print(f"    {login}  ({tool_type}, {commit_date[:10]})")

            total = result.get("total_count", "?")
            print(f"    Page {page}: {len(items)} results (total={total}), "
                  f"+{new_this_page} new, {len(positives)} total")
            if len(items) < 100:
                break
            page += 1

    print(f"\nCommit search total: {len(positives)} new unique accounts")
    tool_counts = {}
    for p in positives.values():
        t = p["tool_type"]
        tool_counts[t] = tool_counts.get(t, 0) + 1
    for t, n in tool_counts.items():
        print(f"  {t}: {n}")
    return positives


# ---------------------------------------------------------------------------
# Stage 2 — Deep scrape each account
# ---------------------------------------------------------------------------

def _scrape_commits_for_repo(owner, repo_name, account_login, max_commits=200):
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
                "sha":          c.get("sha", "")[:12],
                "message":      commit_obj.get("message", "")[:500],
                "created_at":   author_date,
                "repo":         f"{owner}/{repo_name}",
                "additions":    stats.get("additions"),
                "deletions":    stats.get("deletions"),
                "has_test_file": None,
                "has_impl_file": None,
                "file_sampled": False,
            })
        if len(result) < 100:
            break
        page += 1
    return commits


def _scrape_prs_for_repo(owner, repo_name, account_login, max_prs=100):
    url = (
        f"https://api.github.com/repos/{owner}/{repo_name}/pulls"
        f"?state=closed&creator={account_login}&per_page={max_prs}"
        f"&sort=updated&direction=desc"
    )
    result = gh_get(url)
    _sleep()
    if not result or not isinstance(result, list):
        return []
    prs = []
    for pr in result:
        prs.append({
            "author_login": (pr.get("user") or {}).get("login", ""),
            "body_length":  len(pr.get("body") or ""),
            "created_at":   pr.get("created_at"),
            "merged_at":    pr.get("merged_at"),
            "state":        pr.get("state"),
        })
    return prs


def _sample_commit_files(owner, repo_name, commits):
    if not commits:
        return
    sample_indices = random.sample(
        range(len(commits)),
        min(MAX_FILE_SAMPLE_PER_ACCOUNT, max(1, int(len(commits) * FILE_SAMPLE_RATE))),
    )
    impl_extensions = {".py", ".js", ".ts", ".go", ".rs", ".java", ".cpp", ".c"}
    test_keywords   = {"test", "spec"}
    for idx in sample_indices:
        commit = commits[idx]
        sha = commit["sha"]
        url = f"https://api.github.com/repos/{owner}/{repo_name}/commits/{sha}"
        detail = gh_get(url)
        time.sleep(FILE_SAMPLE_DELAY)
        if not detail or "files" not in detail:
            continue
        commit["file_sampled"] = True
        has_test = has_impl = False
        for file_obj in detail.get("files", []):
            fname = file_obj.get("filename", "").lower()
            if any(kw in fname for kw in test_keywords):
                has_test = True
            if any(fname.endswith(ext) for ext in impl_extensions):
                has_impl = True
        commit["has_test_file"] = has_test
        commit["has_impl_file"] = has_impl


def scrape_account(login):
    cache_file = CACHE_DIR / f"{login}.json"
    if cache_file.exists():
        with open(cache_file) as f:
            return json.load(f)

    data = {
        "login":   login,
        "profile": None,
        "repos":   [],
        "commits": [],
        "prs":     [],
        "error":   None,
    }

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

    repos_raw = gh_get(
        f"https://api.github.com/users/{login}/repos"
        f"?sort=created&direction=asc&per_page=30"
    )
    _sleep()
    if not repos_raw or not isinstance(repos_raw, list):
        repos_raw = []

    marker_repos = []
    for repo in repos_raw[:5]:
        repo_name  = repo.get("name", "")
        owner_name = repo.get("owner", {}).get("login", login)

        contents = gh_get(
            f"https://api.github.com/repos/{owner_name}/{repo_name}/contents/"
        )
        _sleep()

        has_marker = False
        if contents and isinstance(contents, list):
            for item in contents:
                if item.get("name", "").lower() in (
                    "claude.md", ".claude", ".hermes", "agents.md",
                    ".aider.conf.yml", ".aider",
                ):
                    has_marker = True
                    break
        if has_marker:
            marker_repos.append(f"{owner_name}/{repo_name}")

        data["repos"].append({
            "name":       repo_name,
            "created_at": repo.get("created_at"),
            "language":   repo.get("language"),
            "size":       repo.get("size"),
        })

        commits = _scrape_commits_for_repo(owner_name, repo_name, login)
        _sample_commit_files(owner_name, repo_name, commits)
        data["commits"].extend(commits)

        prs = _scrape_prs_for_repo(owner_name, repo_name, login)
        data["prs"].extend(prs)

    if marker_repos:
        marker_path = DATA_DIR / "expansion_markers.csv"
        write_header = not marker_path.exists()
        with open(marker_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["login", "repo", "marker_found"])
            for repo_full in marker_repos:
                w.writerow([login, repo_full, True])

    cache_file.write_text(json.dumps(data, indent=2))
    return data


# ---------------------------------------------------------------------------
# Stage 3 — Feature extraction
# ---------------------------------------------------------------------------

def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, TypeError):
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
    window = []
    for c in commits:
        dt = _parse_dt(c.get("created_at"))
        if dt is None or dt < after:
            continue
        if before and dt >= before:
            continue
        window.append(c)

    if not window:
        return {
            "commit_count":                  0,
            "mean_message_length":           0.0,
            "active_weeks":                  0,
            "repos_touched":                 0,
            "mean_commits_per_active_week":  0.0,
            "frac_multiline":                0.0,
            "frac_conventional":             0.0,
            "frac_mentions_test":            0.0,
            "frac_has_bullets":              0.0,
            "mean_inter_commit_hours":       0.0,
            "frac_burst_commits":            0.0,
            "sampled_test_cowrite_rate":     0.0,
            "file_sample_count":             0,
            "mean_pr_body_length":           0.0,
            "frac_pr_has_body":              0.0,
        }

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2]
                        for c in window if _parse_dt(c["created_at"])})
    repos = len({c.get("repo", "") for c in window if c.get("repo")})

    cleaned = [_strip_markers(c.get("message", "")) for c in window]
    msg_lengths = [len(m) for m in cleaned]

    multiline_count = sum(1 for m in cleaned if "\n" in m)

    conventional_re = re.compile(
        r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?:",
        re.IGNORECASE,
    )
    conventional_count = sum(1 for m in cleaned if conventional_re.match(m))
    test_re = re.compile(r"\btests?\b", re.IGNORECASE)
    test_count = sum(1 for m in cleaned if test_re.search(m))
    bullets_count = sum(1 for m in cleaned if "- " in m or "* " in m)

    sorted_w = sorted(window, key=lambda c: _parse_dt(c.get("created_at")) or datetime.min)
    inter_hours = []
    for i in range(1, len(sorted_w)):
        dt1 = _parse_dt(sorted_w[i-1].get("created_at"))
        dt2 = _parse_dt(sorted_w[i].get("created_at"))
        if dt1 and dt2:
            inter_hours.append((dt2 - dt1).total_seconds() / 3600.0)

    mean_inter = sum(inter_hours) / len(inter_hours) if inter_hours else 0.0
    burst_count = sum(1 for h in inter_hours if h <= 2.0)
    frac_burst  = burst_count / len(inter_hours) if inter_hours else 0.0

    sampled = [c for c in window if c.get("file_sampled")]
    file_sample_count = len(sampled)
    if sampled:
        with_impl = sum(1 for c in sampled if c.get("has_impl_file"))
        with_both = sum(1 for c in sampled
                        if c.get("has_impl_file") and c.get("has_test_file"))
        test_cowrite = with_both / with_impl if with_impl > 0 else 0.0
    else:
        test_cowrite = 0.0

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
        "sampled_test_cowrite_rate":     round(test_cowrite, 3),
        "file_sample_count":             file_sample_count,
        "mean_pr_body_length":           0.0,
        "frac_pr_has_body":              0.0,
    }


def _window_pr_features(prs, after, before=None):
    window = [pr for pr in prs
              if (dt := _parse_dt(pr.get("created_at"))) is not None
              and dt >= after
              and (before is None or dt < before)]
    if not window:
        return {"mean_pr_body_length": 0.0, "frac_pr_has_body": 0.0}
    body_lengths = [pr.get("body_length", 0) for pr in window]
    return {
        "mean_pr_body_length": round(sum(body_lengths) / len(body_lengths), 2),
        "frac_pr_has_body":    round(
            sum(1 for bl in body_lengths if bl > 50) / len(body_lengths), 3
        ),
    }


def extract_features(login, data, positive_meta):
    """Extract pre/post features for one account. Returns a row dict or None."""
    if data.get("error"):
        return None

    commits = _deduplicate_commits(data.get("commits", []))
    prs = data.get("prs", [])

    confidence = positive_meta.get("marker_confidence", "high")
    if confidence == "high":
        marker_dt = _parse_dt(positive_meta.get("first_marker_date", ""))
        if marker_dt and marker_dt > PRE_START:
            account_post_start = marker_dt
            account_pre_cutoff = marker_dt
        else:
            account_post_start = POST_START
            account_pre_cutoff = PRE_CUTOFF
    else:
        account_post_start = POST_START
        account_pre_cutoff = PRE_CUTOFF

    pre_count  = _count_commits_in_window(commits, after=PRE_START, before=account_pre_cutoff)
    post_count = _count_commits_in_window(commits, after=account_post_start)

    if pre_count < MIN_PRE_COMMITS or post_count < MIN_POST_COMMITS:
        print(f"  {login}: DROPPED both-window filter ({pre_count} pre, {post_count} post)")
        return None

    pre_cf  = _window_commit_features(commits, after=PRE_START, before=account_pre_cutoff)
    post_cf = _window_commit_features(commits, after=account_post_start)
    pre_pf  = _window_pr_features(prs, after=PRE_START, before=account_pre_cutoff)
    post_pf = _window_pr_features(prs, after=account_post_start)

    pre_cf.update(pre_pf)
    post_cf.update(post_pf)

    row = {
        "login":             login,
        "label":             1,
        "discovery_method":  positive_meta.get("discovery_method", ""),
        "marker_confidence": confidence,
        "tool_type":         positive_meta.get("tool_type", "unknown"),
    }
    for k, v in pre_cf.items():
        row[f"pre_{k}"] = v
    for k, v in post_cf.items():
        row[f"post_{k}"] = v
    for k in pre_cf:
        row[f"delta_{k}"] = round(post_cf[k] - pre_cf[k], 3)

    print(f"  {login} ({positive_meta.get('tool_type','?')}, "
          f"conf={confidence}): pre={pre_count}, post={post_count}, "
          f"marker={positive_meta.get('first_marker_date','?')[:10]}")
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Expanded positive scraper — Claude + Aider, v1.0")
    print("=" * 70)

    existing = load_existing_logins()

    # Load or initialise progress checkpoint
    progress_path = DATA_DIR / "expansion_progress.json"
    completed = set()
    if progress_path.exists():
        completed = set(json.loads(progress_path.read_text()))
        print(f"Resuming: {len(completed)} accounts already done")

    # Load existing expansion raw data
    raw_path = DATA_DIR / "expansion_raw.json"
    all_raw = {}
    if raw_path.exists():
        with open(raw_path) as f:
            all_raw = json.load(f)

    # Stage 1 — discover
    positives = stage1_commit_search(existing | completed)

    if not positives:
        print("\nNo new accounts found. Exiting.")
        return

    # Save positive login list for reference
    pos_list_path = DATA_DIR / "expansion_positive_logins.csv"
    with open(pos_list_path, "w", newline="") as f:
        fields = ["login", "discovery_method", "first_marker_date",
                  "marker_type", "marker_confidence", "tool_type"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(positives.values())
    print(f"\nLogin list saved: {pos_list_path.name} ({len(positives)} accounts)")

    # Stage 2 — scrape with circuit breaker
    print(f"\n=== STAGE 2: Scraping {len(positives)} accounts ===")
    consecutive_fails = 0

    for i, (login, meta) in enumerate(positives.items()):
        if login in completed:
            if login not in all_raw:
                cache_file = CACHE_DIR / f"{login}.json"
                if cache_file.exists():
                    with open(cache_file) as f:
                        all_raw[login] = json.load(f)
            continue

        try:
            data = scrape_account(login)
            all_raw[login] = data
            consecutive_fails = 0
        except NetworkError as e:
            consecutive_fails += 1
            print(f"  {login}: SKIPPED (network: {e})")
            if consecutive_fails >= CONSECUTIVE_NETWORK_FAIL_LIMIT:
                print(f"  Circuit breaker tripped. Pausing {CIRCUIT_BREAKER_PAUSE}s...")
                time.sleep(CIRCUIT_BREAKER_PAUSE)
                consecutive_fails = 0
            continue

        completed.add(login)
        progress_path.write_text(json.dumps(sorted(completed)))

        # Persist raw data incrementally (safe restart)
        with open(raw_path, "w") as f:
            json.dump(all_raw, f, indent=2)

        if (i + 1) % 10 == 0:
            print(f"  Progress: {i+1}/{len(positives)} scraped")

    print(f"\nScraping done: {len(all_raw)} accounts in raw data")

    # Stage 3 — features
    print("\n=== STAGE 3: Feature extraction ===")
    rows = []
    for login, meta in positives.items():
        if login not in all_raw:
            continue
        row = extract_features(login, all_raw[login], meta)
        if row:
            rows.append(row)

    if not rows:
        print("No rows passed the both-window filter.")
        return

    feat_path = DATA_DIR / "expansion_features.csv"
    with open(feat_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    claude_n = sum(1 for r in rows if r.get("tool_type") == "claude")
    aider_n  = sum(1 for r in rows if r.get("tool_type") == "aider")

    print(f"\nDone. {len(rows)} accounts written to {feat_path.name}")
    print(f"  Claude: {claude_n}  |  Aider: {aider_n}")
    print(f"\nTo merge with main dataset:")
    print(f"  python3 -c \"")
    print(f"    import pandas as pd")
    print(f"    main = pd.read_csv('data/classifier_full_features.csv')")
    print(f"    exp  = pd.read_csv('data/expansion_features.csv')")
    print(f"    # tool_type col only in expansion — add it to main for consistency")
    print(f"    main['tool_type'] = 'claude'")
    print(f"    merged = pd.concat([main, exp], ignore_index=True)")
    print(f"    merged.to_csv('data/classifier_expanded_features.csv', index=False)")
    print(f"    print(merged.groupby(['label','tool_type']).size())\"")


if __name__ == "__main__":
    main()
