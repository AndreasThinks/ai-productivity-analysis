#!/usr/bin/env python
"""
v4 re-scrape: event-time windows of equal length for every account.

Why v4 exists
-------------
The v2.x scrapers split high-confidence adopters at the date of a recent
Claude-attributed commit (all in March 2026, days before the scrape), while
controls were split at 1 January 2024. Adopters therefore had a post-window of
two to three weeks and controls one of about two years, which mechanically
produced the headline tempo effects and most of the classifier's AUC. See
methodology_decisions.md, decision 14.

Design
------
1. Adoption date. For each adopter, the earliest commit they authored that
   carries the tool's co-author trailer (commit search, oldest first). Claude
   markers dated before Claude Code's launch (24 Feb 2025) are not Claude
   Code; those accounts are flagged ``pre_launch`` and left out of the main
   sample.
2. Placebo dates. Each control gets a date drawn from the adopters' adoption
   dates, snapped to the control's nearest commit, so that both groups are
   anchored on a day they were active.
3. Equal windows. Every account is observed for WINDOW_DAYS before and after
   its date. Only dates at least WINDOW_DAYS before the scrape are eligible,
   so every post-window is complete.
4. All repositories. Commits come from commit search across every public
   repository (default branches), not the five oldest owned repos, and each
   window is split into smaller date ranges until no query hits the
   1,000-result cap.
5. Authorship is kept. Author and committer names and emails and the full
   message are stored, so agent-authored commits (Claude Code, Aider, Copilot
   agent, Cursor agent, Codex, Devin) can be separated from human ones.
6. Contamination. Controls with any agent marker before the end of their
   post-window are excluded and counted.

Raw caches hold commit messages and logins, so they go to the private
directory (scripts/paths.py), never to data/.

Run (needs a personal access token; ~40 s per account because commit and
issue search are limited to 30 requests a minute):

  GITHUB_TOKEN=... uv run scripts/rescrape_v4.py --accounts private/accounts_v4.csv

Smoke test on a handful of accounts:

  GITHUB_TOKEN=... uv run scripts/rescrape_v4.py --accounts private/accounts_v4.csv --limit 3
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

try:
    from scripts.paths import PRIVATE_DIR
except ImportError:  # run as a plain script
    from paths import PRIVATE_DIR

CACHE_DIR = PRIVATE_DIR / "v4_cache"
DATES_PATH = PRIVATE_DIR / "v4_event_dates.csv"

WINDOW_DAYS = 182
MIN_PRE_COMMITS = 5

TOOL_LAUNCH = {
    # Claude Code research preview: 24 February 2025.
    "claude": datetime(2025, 2, 24, tzinfo=timezone.utc),
    # Aider's first public release: June 2023.
    "aider": datetime(2023, 6, 1, tzinfo=timezone.utc),
}

# Commit-search text that finds a tool's co-author trailer.
MARKER_QUERY = {
    "claude": "noreply@anthropic.com",
    "aider": "aider.chat",
}

SEARCH_DELAY = 2.2  # search API: 30 requests / minute
CORE_DELAY = 0.75
REQUEST_TIMEOUT = 30
SEARCH_RESULT_CAP = 1000

# ---------------------------------------------------------------------------
# Agent detection
# ---------------------------------------------------------------------------

# Patterns on the message (trailers and footers) per tool.
_MESSAGE_PATTERNS = {
    "claude": [
        r"co-authored-by:\s*claude[^\n]*<?noreply@anthropic\.com>?",
        r"generated with \[?claude code\]?",
        r"claude\.(ai|com)/(code|claude-code)",
    ],
    "aider": [r"co-authored-by:\s*aider[^\n]*aider\.chat"],
    "copilot": [r"co-authored-by:\s*copilot[^\n]*<[^>]*copilot[^>]*>"],
    "cursor": [r"co-authored-by:\s*cursor[^\n]*cursoragent@cursor\.com"],
    "codex": [r"co-authored-by:\s*codex[^\n]*", r"chatgpt-codex-connector"],
    "devin": [r"co-authored-by:\s*devin[^\n]*devin-ai-integration"],
}
_MESSAGE_RE = {
    tool: re.compile("|".join(pats), re.IGNORECASE) for tool, pats in _MESSAGE_PATTERNS.items()
}
# Patterns on author / committer identity.
_IDENTITY_PATTERNS = {
    "aider": re.compile(r"\(aider\)\s*$", re.IGNORECASE),
    "claude": re.compile(r"noreply@anthropic\.com", re.IGNORECASE),
    "copilot": re.compile(r"copilot-swe-agent|198982749\+copilot@users\.noreply", re.IGNORECASE),
    "cursor": re.compile(r"cursoragent@cursor\.com", re.IGNORECASE),
    "devin": re.compile(r"devin-ai-integration", re.IGNORECASE),
}

_STRIP_RE = re.compile(
    "|".join(p for pats in _MESSAGE_PATTERNS.values() for p in pats)
    + r"|🤖\s*generated with[^\n]*",
    re.IGNORECASE,
)


def detect_agent(commit: dict[str, Any]) -> str | None:
    """Return the agent tool that authored or co-authored a commit, if any.

    Looks at trailers and footers in the message, then at the author and
    committer identity (Aider marks itself with "(aider)" in the author name).
    A Claude Code user who switched trailers off is not detected, so
    "human" commits are an upper bound on human authorship.
    """
    message = commit.get("message") or ""
    for tool, rx in _MESSAGE_RE.items():
        if rx.search(message):
            return tool
    identities = " ".join(
        str(commit.get(k) or "")
        for k in ("author_name", "author_email", "committer_name", "committer_email")
    )
    for tool, rx in _IDENTITY_PATTERNS.items():
        if rx.search(identities):
            return tool
    return None


def strip_agent_markers(message: str) -> str:
    """Remove agent trailers and footers so they cannot leak into style features."""
    return _STRIP_RE.sub("", message or "").strip()


# ---------------------------------------------------------------------------
# Dates and windows
# ---------------------------------------------------------------------------


def parse_ts(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class Windows:
    event: datetime
    pre_start: datetime
    post_end: datetime

    def label(self, ts: datetime | None) -> str | None:
        if ts is None:
            return None
        if self.pre_start <= ts < self.event:
            return "pre"
        if self.event <= ts < self.post_end:
            return "post"
        return None


def event_windows(event: datetime, days: int = WINDOW_DAYS) -> Windows:
    """Equal pre and post windows of ``days`` either side of the event date."""
    event = event.astimezone(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    return Windows(event=event, pre_start=event - timedelta(days=days), post_end=event + timedelta(days=days))


def latest_eligible_event(scrape_time: datetime, days: int = WINDOW_DAYS) -> datetime:
    """The last event date whose post-window is complete at scrape time."""
    return scrape_time - timedelta(days=days)


def classify_adopter(first_marker: datetime | None, tool: str, scrape_time: datetime) -> str:
    """Eligibility of an adopter given the date of their first marker commit."""
    if first_marker is None:
        return "no_marker_found"
    if first_marker < TOOL_LAUNCH[tool]:
        return "pre_launch"
    if first_marker > latest_eligible_event(scrape_time):
        return "post_window_incomplete"
    return "eligible"


def draw_placebo_target(adopter_dates: list[datetime], rng: random.Random) -> datetime:
    """Draw a target date from the empirical distribution of adoption dates."""
    if not adopter_dates:
        raise ValueError("need at least one adopter date to draw placebo dates")
    return rng.choice(adopter_dates)


def date_ranges(start: datetime, end: datetime, parts: int) -> list[tuple[datetime, datetime]]:
    """Split [start, end) into ``parts`` contiguous ranges."""
    step = (end - start) / parts
    edges = [start + step * i for i in range(parts)] + [end]
    return list(zip(edges[:-1], edges[1:]))


def _search_date(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# GitHub client
# ---------------------------------------------------------------------------


class GitHubClient:
    def __init__(self, token: str | None):
        if not token:
            raise SystemExit("GITHUB_TOKEN is required for the v4 scrape")
        self.token = token
        self.calls = 0

    def get(self, path: str, params: dict[str, Any] | None = None, *, search: bool = False) -> Any:
        url = "https://api.github.com" + path
        if params:
            url += "?" + urllib.parse.urlencode(params)
        headers = {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.token}",
            "User-Agent": "ai-productivity-v4-scraper",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        for attempt in range(6):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=REQUEST_TIMEOUT) as resp:
                    self.calls += 1
                    data = json.loads(resp.read().decode("utf-8"))
                    time.sleep(SEARCH_DELAY if search else CORE_DELAY)
                    return data
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    reset = e.headers.get("X-RateLimit-Reset")
                    retry_after = e.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        wait = int(retry_after) + 2
                    elif reset and reset.isdigit():
                        wait = min(max(int(reset) - int(time.time()) + 3, 10), 900)
                    else:
                        wait = min(60 * (attempt + 1), 600)
                    print(f"  rate limited; sleeping {wait}s")
                    time.sleep(wait)
                    continue
                if e.code in (404, 422):
                    return None
                if e.code >= 500:
                    time.sleep(15 * (attempt + 1))
                    continue
                raise
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                time.sleep(min(15 * (attempt + 1), 120))
        return None


# ---------------------------------------------------------------------------
# Collection
# ---------------------------------------------------------------------------


def normalise_commit(item: dict[str, Any]) -> dict[str, Any]:
    commit = item.get("commit") or {}
    author = commit.get("author") or {}
    committer = commit.get("committer") or {}
    return {
        "sha": item.get("sha"),
        "repo": (item.get("repository") or {}).get("full_name"),
        "repo_owner": ((item.get("repository") or {}).get("owner") or {}).get("login"),
        "authored_at": author.get("date"),
        "committed_at": committer.get("date"),
        "author_name": author.get("name"),
        "author_email": author.get("email"),
        "committer_name": committer.get("name"),
        "committer_email": committer.get("email"),
        "n_parents": len(item.get("parents") or []),
        "message": commit.get("message") or "",
    }


def search_commits(client: GitHubClient, query: str, start: datetime, end: datetime,
                   *, depth: int = 0) -> tuple[list[dict[str, Any]], bool]:
    """All commits matching ``query`` committed in [start, end).

    Splits the range until every query is under the 1,000-result cap. Returns
    (commits, truncated); truncated is True only if a single day still holds
    more than 1,000 commits.
    """
    q = f"{query} committer-date:{_search_date(start)}..{_search_date(end - timedelta(seconds=1))}"
    first = client.get("/search/commits", {"q": q, "sort": "committer-date", "order": "asc", "per_page": 100}, search=True)
    if not first:
        return [], False
    total = first.get("total_count", 0)
    if total > SEARCH_RESULT_CAP and (end - start) > timedelta(days=1) and depth < 8:
        commits: list[dict[str, Any]] = []
        truncated = False
        for a, b in date_ranges(start, end, 4):
            part, t = search_commits(client, query, a, b, depth=depth + 1)
            commits.extend(part)
            truncated = truncated or t
        return commits, truncated
    items = list(first.get("items", []))
    page = 2
    while len(items) < min(total, SEARCH_RESULT_CAP):
        nxt = client.get("/search/commits", {"q": q, "sort": "committer-date", "order": "asc", "per_page": 100, "page": page}, search=True)
        batch = (nxt or {}).get("items", [])
        if not batch:
            break
        items.extend(batch)
        page += 1
    return [normalise_commit(i) for i in items], total > SEARCH_RESULT_CAP


def find_first_marker(client: GitHubClient, login: str, tool: str) -> dict[str, Any]:
    """Earliest commit by ``login`` carrying the tool's marker."""
    q = f"author:{login} {MARKER_QUERY[tool]}"
    res = client.get("/search/commits", {"q": q, "sort": "author-date", "order": "asc", "per_page": 20}, search=True)
    items = (res or {}).get("items", [])
    for item in items:
        c = normalise_commit(item)
        if detect_agent(c) == tool:
            return {"first_marker_at": c["authored_at"], "first_marker_sha": c["sha"],
                    "marker_total": res.get("total_count", 0)}
    return {"first_marker_at": None, "first_marker_sha": None, "marker_total": (res or {}).get("total_count", 0)}


def nearest_commit_date(client: GitHubClient, login: str, target: datetime, *,
                        lo: datetime, hi: datetime, max_gap_days: int = 60) -> datetime | None:
    """The control's commit closest to ``target`` within [lo, hi], or None."""
    best: datetime | None = None
    for direction in ("after", "before"):
        if direction == "after":
            a, b, order = target, min(target + timedelta(days=max_gap_days), hi), "asc"
        else:
            a, b, order = max(target - timedelta(days=max_gap_days), lo), target, "desc"
        if a >= b:
            continue
        q = f"author:{login} committer-date:{_search_date(a)}..{_search_date(b)}"
        res = client.get("/search/commits", {"q": q, "sort": "committer-date", "order": order, "per_page": 1}, search=True)
        items = (res or {}).get("items", [])
        if items:
            ts = parse_ts(normalise_commit(items[0])["committed_at"])
            if ts and (best is None or abs(ts - target) < abs(best - target)):
                best = ts
    return best


def count_prs(client: GitHubClient, login: str, start: datetime, end: datetime) -> dict[str, int]:
    """Authored PR counts in [start, end): opened, merged, merged into others' repos."""
    rng = f"created:{start.date().isoformat()}..{(end - timedelta(days=1)).date().isoformat()}"
    out = {}
    for key, extra in (("prs_opened", ""), ("prs_merged", " is:merged"), ("prs_merged_external", f" is:merged -user:{login}")):
        res = client.get("/search/issues", {"q": f"type:pr author:{login} {rng}{extra}", "per_page": 1}, search=True)
        out[key] = int((res or {}).get("total_count", 0))
    return out


def scrape_window_data(client: GitHubClient, login: str, w: Windows) -> dict[str, Any]:
    commits, truncated = search_commits(client, f"author:{login}", w.pre_start, w.post_end)
    seen, unique = set(), []
    for c in commits:
        if c["sha"] in seen:
            continue
        seen.add(c["sha"])
        unique.append(c)
    return {
        "commits": unique,
        "commits_truncated": truncated,
        "prs_pre": count_prs(client, login, w.pre_start, w.event),
        "prs_post": count_prs(client, login, w.event, w.post_end),
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def load_accounts(path: Path) -> list[dict[str, str]]:
    """CSV with columns login, group (adopter|control) and tool (claude|aider|'')."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if r["group"] not in ("adopter", "control"):
            raise ValueError(f"bad group {r['group']!r} for {r['login']}")
    return rows


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def phase_dates(client: GitHubClient, accounts: list[dict[str, str]], scrape_time: datetime,
                seed: int = 2026) -> list[dict[str, Any]]:
    """Find adoption dates, check controls for contamination, draw placebo dates."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for acc in accounts:
        if acc["group"] != "adopter":
            continue
        m = find_first_marker(client, acc["login"], acc["tool"])
        first = parse_ts(m["first_marker_at"])
        rows.append({**acc, **m, "status": classify_adopter(first, acc["tool"], scrape_time),
                     "event_date": first.isoformat() if first else None})
        print(f"  adopter {acc['login']}: {rows[-1]['status']} ({m['first_marker_at']})")

    eligible = [parse_ts(r["event_date"]) for r in rows if r["status"] == "eligible" and r["tool"] == "claude"]
    lo, hi = TOOL_LAUNCH["claude"], latest_eligible_event(scrape_time)
    for acc in accounts:
        if acc["group"] != "control":
            continue
        markers = {t: find_first_marker(client, acc["login"], t) for t in ("claude", "aider")}
        earliest = min((parse_ts(v["first_marker_at"]) for v in markers.values() if v["first_marker_at"]), default=None)
        event = None
        for _ in range(3):
            target = draw_placebo_target(eligible, rng)
            event = nearest_commit_date(client, acc["login"], target, lo=lo, hi=hi)
            if event:
                break
        if event is None:
            status = "inactive_near_placebo"
        elif earliest and earliest < event_windows(event).post_end:
            status = "contaminated"
        else:
            status = "eligible"
        rows.append({**acc, "first_marker_at": earliest.isoformat() if earliest else None,
                     "first_marker_sha": None, "marker_total": sum(v["marker_total"] for v in markers.values()),
                     "status": status, "event_date": event.isoformat() if event else None})
        print(f"  control {acc['login']}: {status}")
    return rows


def phase_windows(client: GitHubClient, dated: list[dict[str, Any]], scrape_time: datetime) -> None:
    for r in dated:
        if r["status"] != "eligible":
            continue
        path = CACHE_DIR / f"{r['login']}.json"
        if path.exists():
            continue
        w = event_windows(parse_ts(r["event_date"]))
        print(f"  {r['login']}: {w.pre_start.date()} | {w.event.date()} | {w.post_end.date()}")
        data = scrape_window_data(client, r["login"], w)
        _write_json(path, {
            "login": r["login"], "group": r["group"], "tool": r["tool"],
            "event_date": w.event.isoformat(), "pre_start": w.pre_start.isoformat(),
            "post_end": w.post_end.isoformat(), "scraped_at": scrape_time.isoformat(), **data,
        })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--accounts", type=Path, default=PRIVATE_DIR / "accounts_v4.csv")
    ap.add_argument("--limit", type=int, default=None, help="only the first N adopters and N controls")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--skip-dates", action="store_true", help="reuse an existing event-dates file")
    args = ap.parse_args()

    client = GitHubClient(os.environ.get("GITHUB_TOKEN"))
    scrape_time = datetime.now(timezone.utc)
    accounts = load_accounts(args.accounts)
    if args.limit:
        adopters = [a for a in accounts if a["group"] == "adopter"][: args.limit]
        controls = [a for a in accounts if a["group"] == "control"][: args.limit]
        accounts = adopters + controls

    if args.skip_dates and DATES_PATH.exists():
        with open(DATES_PATH, newline="") as f:
            dated = list(csv.DictReader(f))
    else:
        print("=== Phase 1: adoption and placebo dates ===")
        dated = phase_dates(client, accounts, scrape_time, seed=args.seed)
        DATES_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DATES_PATH, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(dated[0].keys()))
            writer.writeheader()
            writer.writerows(dated)

    print("=== Phase 2: equal-length windows ===")
    phase_windows(client, dated, scrape_time)
    print(f"done; {client.calls} API calls")


if __name__ == "__main__":
    main()
