#!/usr/bin/env python3
"""
GitHub Productivity Panel Scraper
==================================
Scrapes 9 quarterly windows from Q4 2022 to Q4 2024, 500 users per window.
Produces a quarterly panel of per-developer productivity metrics by country.

Usage:
    uv run scripts/scrape_github_panel.py

Output:
    data/github_panel_raw.json     — raw results per window
    data/github_panel_flat.csv     — flat panel (country × quarter)
"""

import gzip
import io
import json
import os
import random
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

# ── Configuration ──────────────────────────────────────────────────────────────

# 9 quarterly windows: Q4 2022 through Q4 2024
# One representative hour per quarter (mid-month, midday)
QUARTERLY_WINDOWS = [
    ("2022-Q4", "2022-11-15-12"),   # Q4 2022 — first full quarter post-ChatGPT launch
    ("2023-Q1", "2023-02-15-12"),
    ("2023-Q2", "2023-05-15-12"),
    ("2023-Q3", "2023-08-15-12"),
    ("2023-Q4", "2023-11-15-12"),
    ("2024-Q1", "2024-02-15-12"),
    ("2024-Q2", "2024-05-15-12"),
    ("2024-Q3", "2024-08-15-12"),
    ("2024-Q4", "2024-11-15-12"),
]

PRODUCTIVE_EVENT_TYPES = {
    "PushEvent":         "commits",
    "PullRequestEvent":  "pull_requests",
    "CreateEvent":       "creates",
    "IssueCommentEvent": "comments",
    "ReleaseEvent":      "releases",
    "IssuesEvent":       "issues",
}

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
CACHE_DIR = Path("/tmp/github_productivity_cache")
CACHE_DIR.mkdir(exist_ok=True)

MAX_USERS_PER_WINDOW = 500
API_DELAY = 0.2  # seconds between GitHub API calls

OUTPUT_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Country normalisation ──────────────────────────────────────────────────────

COUNTRY_MAP = {
    "united states": "US", "usa": "US", "u.s.a": "US", "u.s.": "US",
    "united kingdom": "GB", "uk": "GB", "u.k.": "GB", "england": "GB",
    "scotland": "GB", "wales": "GB", "great britain": "GB",
    "germany": "DE", "deutschland": "DE",
    "france": "FR",
    "canada": "CA",
    "australia": "AU",
    "india": "IN",
    "china": "CN",
    "japan": "JP",
    "brazil": "BR",
    "russia": "RU",
    "netherlands": "NL", "the netherlands": "NL", "holland": "NL",
    "sweden": "SE",
    "norway": "NO",
    "denmark": "DK",
    "finland": "FI",
    "spain": "ES",
    "italy": "IT",
    "poland": "PL",
    "switzerland": "CH",
    "austria": "AT",
    "belgium": "BE",
    "portugal": "PT",
    "south korea": "KR", "korea": "KR",
    "taiwan": "TW",
    "singapore": "SG",
    "israel": "IL",
    "turkey": "TR", "türkiye": "TR",
    "ukraine": "UA",
    "czech republic": "CZ", "czechia": "CZ",
    "argentina": "AR",
    "mexico": "MX",
    "new zealand": "NZ",
    "south africa": "ZA",
    "indonesia": "ID",
    "thailand": "TH",
    "vietnam": "VN",
    "pakistan": "PK",
    "nigeria": "NG",
    "kenya": "KE",
    "egypt": "EG",
    "iran": "IR",
    "romania": "RO",
    "hungary": "HU",
    "greece": "GR",
    "colombia": "CO",
    "chile": "CL",
    "peru": "PE",
    "bangladesh": "BD",
    "malaysia": "MY",
    "philippines": "PH",
    "saudi arabia": "SA",
    "uae": "AE", "united arab emirates": "AE",
    "hong kong": "HK",
    "ireland": "IE",
    "sweden": "SE",
    # US state abbrevs
    "ca": "US", "ny": "US", "tx": "US", "wa": "US", "or": "US",
    "sf": "US", "nyc": "US", "la": "US", "ma": "US", "il": "US",
    # City → country
    "london": "GB", "berlin": "DE", "paris": "FR", "toronto": "CA",
    "sydney": "AU", "melbourne": "AU", "bangalore": "IN", "mumbai": "IN",
    "delhi": "IN", "hyderabad": "IN", "pune": "IN", "chennai": "IN",
    "beijing": "CN", "shanghai": "CN", "shenzhen": "CN",
    "tokyo": "JP", "osaka": "JP",
    "amsterdam": "NL", "stockholm": "SE", "oslo": "NO", "copenhagen": "DK",
    "helsinki": "FI", "zurich": "CH", "vienna": "AT", "brussels": "BE",
    "lisbon": "PT", "madrid": "ES", "barcelona": "ES", "rome": "IT",
    "milan": "IT", "warsaw": "PL", "prague": "CZ", "budapest": "HU",
    "athens": "GR", "kyiv": "UA", "kiev": "UA", "moscow": "RU",
    "istanbul": "TR", "tel aviv": "IL", "seoul": "KR", "taipei": "TW",
    "jakarta": "ID", "bangkok": "TH", "nairobi": "KE", "lagos": "NG",
    "cairo": "EG", "cape town": "ZA", "johannesburg": "ZA",
    "buenos aires": "AR", "bogota": "CO", "lima": "PE", "santiago": "CL",
    "mexico city": "MX",
    "san francisco": "US", "new york": "US", "seattle": "US",
    "boston": "US", "chicago": "US", "austin": "US", "denver": "US",
    "mountain view": "US", "palo alto": "US", "los angeles": "US",
    "montreal": "CA", "vancouver": "CA", "ottawa": "CA",
    "dubai": "AE", "singapore city": "SG",
    "kuala lumpur": "MY", "manila": "PH",
}


def normalize_location(location: str | None) -> str | None:
    if not location:
        return None
    loc = location.strip().lower()
    parts = [p.strip() for p in loc.split(",")]
    for part in reversed(parts):
        part = part.strip()
        if part in COUNTRY_MAP:
            return COUNTRY_MAP[part]
    if loc in COUNTRY_MAP:
        return COUNTRY_MAP[loc]
    for key, code in COUNTRY_MAP.items():
        if len(key) > 4 and key in loc:
            return code
    return None


# ── GitHub API ────────────────────────────────────────────────────────────────

def github_api(path: str) -> dict | None:
    cache_key = re.sub(r'[^a-zA-Z0-9_.-]', '_', path.lstrip('/')) + ".json"
    cache_file = CACHE_DIR / cache_key
    if cache_file.exists():
        with cache_file.open() as f:
            return json.load(f)
    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "github-productivity-study/0.2",
    }
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"
    try:
        req = Request(url, headers=headers)
        with urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            with cache_file.open("w") as f:
                json.dump(data, f)
            time.sleep(API_DELAY)
            return data
    except HTTPError as e:
        if e.code == 404:
            return None
        if e.code == 403:
            print(f"  [RATE LIMITED] Sleeping 60s...")
            time.sleep(60)
            return None
        print(f"  [HTTP {e.code}] {path}")
        return None
    except Exception as e:
        print(f"  [ERROR] {path}: {e}")
        return None


def download_gharchive(date_hour: str) -> list[dict]:
    cache_file = CACHE_DIR / f"gharchive_{date_hour.replace('-', '_')}.jsonl"
    if cache_file.exists():
        print(f"  [cache] {date_hour}")
        events = []
        with cache_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        events.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return events
    url = f"https://data.gharchive.org/{date_hour}.json.gz"
    print(f"  [download] {url}")
    try:
        req = Request(url, headers={"User-Agent": "github-productivity-study/0.2"})
        with urlopen(req, timeout=60) as resp:
            compressed = resp.read()
        events = []
        with gzip.open(io.BytesIO(compressed)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        with cache_file.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")
        print(f"  → {len(events):,} events cached")
        return events
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return []


# ── Processing ────────────────────────────────────────────────────────────────

def process_window(label: str, date_hour: str) -> dict:
    print(f"\n{'='*60}")
    print(f"Window: {label} — {date_hour}")
    print(f"{'='*60}")

    print("\n[1] Downloading GH Archive...")
    events = download_gharchive(date_hour)
    if not events:
        return {}

    productive = [e for e in events if e.get("type") in PRODUCTIVE_EVENT_TYPES]
    print(f"  Total: {len(events):,} | Productive: {len(productive):,}")

    actor_events = defaultdict(lambda: defaultdict(int))
    for e in productive:
        login = e.get("actor", {}).get("login", "")
        if login and not login.endswith("[bot]"):
            actor_events[login][PRODUCTIVE_EVENT_TYPES[e["type"]]] += 1

    all_actors = list(actor_events.keys())
    print(f"  Unique human actors: {len(all_actors):,}")

    random.seed(42)
    sampled = random.sample(all_actors, min(MAX_USERS_PER_WINDOW, len(all_actors)))
    print(f"  Sampling {len(sampled)} actors")

    print(f"\n[2] Scraping user locations...")
    actor_country = {}
    located = 0
    for i, login in enumerate(sampled):
        if i % 100 == 0 and i > 0:
            print(f"  {i}/{len(sampled)} ({located} located so far)")
        user = github_api(f"/users/{login}")
        if user:
            country = normalize_location(user.get("location"))
            if country:
                actor_country[login] = country
                located += 1

    hit_rate = located / len(sampled) * 100
    print(f"  Located: {located}/{len(sampled)} ({hit_rate:.1f}%)")

    country_events = defaultdict(lambda: defaultdict(int))
    country_actors = defaultdict(set)
    for login, evts in actor_events.items():
        country = actor_country.get(login)
        if not country:
            continue
        for etype, count in evts.items():
            country_events[country][etype] += count
        country_actors[country].add(login)

    return {
        "label": label,
        "date_hour": date_hour,
        "events_total": len(events),
        "productive_events": len(productive),
        "unique_actors": len(all_actors),
        "sampled_actors": len(sampled),
        "actors_with_location": located,
        "location_hit_rate": round(hit_rate, 2),
        "country_events": {c: dict(e) for c, e in country_events.items()},
        "country_actor_count": {c: len(a) for c, a in country_actors.items()},
    }


def build_flat_csv(results: dict) -> str:
    rows = ["country,quarter,n_developers,commits,pull_requests,creates,comments,releases,issues,total_events,commits_per_dev,prs_per_dev,creates_per_dev,total_events_per_dev"]
    for label, r in results.items():
        for country, n_dev in r["country_actor_count"].items():
            evts = r["country_events"].get(country, {})
            commits = evts.get("commits", 0)
            prs = evts.get("pull_requests", 0)
            creates = evts.get("creates", 0)
            comments = evts.get("comments", 0)
            releases = evts.get("releases", 0)
            issues = evts.get("issues", 0)
            total = commits + prs + creates + comments + releases + issues
            rows.append(
                f"{country},{label},{n_dev},"
                f"{commits},{prs},{creates},{comments},{releases},{issues},{total},"
                f"{commits/n_dev:.4f},{prs/n_dev:.4f},{creates/n_dev:.4f},{total/n_dev:.4f}"
            )
    return "\n".join(rows)


def main():
    print("GitHub Productivity Panel Scraper")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Windows: {len(QUARTERLY_WINDOWS)} | Users/window: {MAX_USERS_PER_WINDOW}")
    print(f"Cache: {CACHE_DIR}")

    raw_output = OUTPUT_DIR / "github_panel_raw.json"
    flat_output = OUTPUT_DIR / "github_panel_flat.csv"

    # Load existing results if partial run
    results = {}
    if raw_output.exists():
        with raw_output.open() as f:
            results = json.load(f)
        print(f"\nResuming — {len(results)} windows already done")

    for label, date_hour in QUARTERLY_WINDOWS:
        if label in results:
            print(f"  Skipping {label} (already done)")
            continue
        r = process_window(label, date_hour)
        if r:
            results[label] = r
            # Save after each window so progress isn't lost
            with raw_output.open("w") as f:
                json.dump(results, f, indent=2)
            flat_csv = build_flat_csv(results)
            with flat_output.open("w") as f:
                f.write(flat_csv)
            print(f"  ✓ {label} saved ({r['actors_with_location']} countries located)")

    print(f"\n{'='*60}")
    print(f"DONE: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Raw results: {raw_output}")
    print(f"Flat CSV:    {flat_output}")

    total_located = sum(r["actors_with_location"] for r in results.values())
    all_countries = set()
    for r in results.values():
        all_countries.update(r["country_actor_count"].keys())
    print(f"Total located developers: {total_located}")
    print(f"Countries covered: {len(all_countries)} — {sorted(all_countries)}")


if __name__ == "__main__":
    main()
