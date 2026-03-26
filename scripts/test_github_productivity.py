#!/usr/bin/env python3
"""
GitHub Productivity by Country — Test Script
============================================
Samples a small window of GH Archive data, scrapes user locations via
GitHub REST API, and produces per-country event counts.

Usage:
    uv run test_github_productivity.py

Configuration:
    GITHUB_TOKEN — set to your PAT for higher rate limits (5000 req/hr)
    Adjust SAMPLE_DATES and SAMPLE_HOURS to control data volume.
"""

import gzip
import json
import os
import re
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from urllib.request import urlopen, Request
from urllib.error import HTTPError, URLError

# ── Configuration ──────────────────────────────────────────────────────────────
# One hour of data per year (should be ~5-20k events each)
SAMPLE_DATES = [
    ("2022", "2022-03-15-12"),  # Baseline: pre-ChatGPT
    ("2024", "2024-03-15-12"),  # Treatment: post-ChatGPT
]

# Event types we care about as productivity proxies
PRODUCTIVE_EVENT_TYPES = {
    "PushEvent":         "commits",
    "PullRequestEvent":  "pull_requests",
    "CreateEvent":       "creates",
    "IssueCommentEvent": "comments",
    "ReleaseEvent":      "releases",
}

# GitHub API rate-limit: 60/hr anon, 5000/hr with token
GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN", "")
CACHE_DIR = Path("/tmp/github_productivity_cache")
CACHE_DIR.mkdir(exist_ok=True)

# Max unique users to look up (keeps test fast)
MAX_USERS_PER_PERIOD = 100
API_DELAY = 0.2  # seconds between API calls (stay well within rate limits)


# ── Helper functions ────────────────────────────────────────────────────────────

def github_api(path: str) -> dict | None:
    """Call GitHub REST API with caching."""
    cache_key = path.replace("/", "_").replace(":", "").lstrip("_") + ".json"
    cache_file = CACHE_DIR / cache_key

    if cache_file.exists():
        with cache_file.open() as f:
            return json.load(f)

    url = f"https://api.github.com{path}"
    headers = {
        "Accept": "application/vnd.github.v3+json",
        "User-Agent": "github-productivity-study/0.1",
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
            return None  # User deleted / not found
        if e.code == 403:
            print(f"  [RATE LIMITED] Sleeping 60s...")
            time.sleep(60)
            return None
        print(f"  [HTTP ERROR {e.code}] {path}")
        return None
    except (URLError, Exception) as e:
        print(f"  [ERROR] {path}: {e}")
        return None


def download_gharchive(date_hour: str) -> list[dict]:
    """Download and parse one hour of GH Archive data."""
    cache_file = CACHE_DIR / f"gharchive_{date_hour.replace('-', '_')}.jsonl"

    if cache_file.exists():
        print(f"  [cache] Loading {date_hour}")
        events = []
        with cache_file.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events

    url = f"https://data.gharchive.org/{date_hour}.json.gz"
    print(f"  [download] {url}")

    try:
        req = Request(url, headers={"User-Agent": "github-productivity-study/0.1"})
        with urlopen(req, timeout=60) as resp:
            compressed = resp.read()

        events = []
        with gzip.open(__import__("io").BytesIO(compressed)) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    pass

        # Cache to disk
        with cache_file.open("w") as f:
            for e in events:
                f.write(json.dumps(e) + "\n")

        print(f"  → {len(events):,} events")
        return events

    except Exception as e:
        print(f"  [ERROR] Could not download {url}: {e}")
        return []


def normalize_location(location: str | None) -> str | None:
    """
    Very rough country extraction from free-text location strings.
    Returns ISO 3166-1 alpha-2 country name or None if we can't identify it.
    This is intentionally simple — a real version would use geocoding.
    """
    if not location:
        return None

    loc = location.strip().lower()

    # Common patterns: "City, Country" or "Country" or "Country, City"
    # We match the LAST comma-delimited token first, then try the whole string
    COUNTRY_MAP = {
        # Full names
        "united states": "US", "usa": "US", "u.s.a": "US", "u.s.": "US",
        "united kingdom": "GB", "uk": "GB", "u.k.": "GB", "england": "GB",
        "scotland": "GB", "wales": "GB",
        "germany": "DE", "deutschland": "DE",
        "france": "FR",
        "canada": "CA",
        "australia": "AU",
        "india": "IN",
        "china": "CN",
        "japan": "JP",
        "brazil": "BR",
        "russia": "RU",
        "netherlands": "NL", "the netherlands": "NL",
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
        "turkey": "TR",
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
        # US state abbreviations (common in GitHub locations)
        "ca": "US", "ny": "US", "tx": "US", "wa": "US", "or": "US",
        "sf": "US", "nyc": "US", "la": "US",
        # City shortcuts (map to country)
        "london": "GB", "berlin": "DE", "paris": "FR", "toronto": "CA",
        "sydney": "AU", "melbourne": "AU", "bangalore": "IN", "mumbai": "IN",
        "delhi": "IN", "beijing": "CN", "shanghai": "CN", "tokyo": "JP",
        "amsterdam": "NL", "stockholm": "SE", "oslo": "NO", "copenhagen": "DK",
        "helsinki": "FI", "zurich": "CH", "vienna": "AT", "brussels": "BE",
        "lisbon": "PT", "madrid": "ES", "barcelona": "ES", "rome": "IT",
        "warsaw": "PL", "prague": "CZ", "budapest": "HU", "athens": "GR",
        "kyiv": "UA", "kiev": "UA", "moscow": "RU", "istanbul": "TR",
        "tel aviv": "IL", "singapore": "SG", "seoul": "KR", "taipei": "TW",
        "jakarta": "ID", "bangkok": "TH", "nairobi": "KE", "lagos": "NG",
        "cairo": "EG", "cape town": "ZA", "johannesburg": "ZA",
        "buenos aires": "AR", "bogota": "CO", "lima": "PE", "santiago": "CL",
        "mexico city": "MX",
        "san francisco": "US", "new york": "US", "seattle": "US",
        "boston": "US", "chicago": "US", "austin": "US", "denver": "US",
        "mountain view": "US", "palo alto": "US",
        "montreal": "CA", "vancouver": "CA", "ottawa": "CA",
        "hong kong": "HK",
        "dubai": "AE", "united arab emirates": "AE", "uae": "AE",
        "netherlands": "NL",
    }

    # Try last comma-delimited segment first (usually the country)
    parts = [p.strip() for p in loc.split(",")]
    for part in reversed(parts):
        if part in COUNTRY_MAP:
            return COUNTRY_MAP[part]

    # Try full string
    if loc in COUNTRY_MAP:
        return COUNTRY_MAP[loc]

    # Partial match for common country names
    for key, code in COUNTRY_MAP.items():
        if len(key) > 4 and key in loc:  # skip short abbreviations for substring match
            return code

    return None


# ── Main analysis ───────────────────────────────────────────────────────────────

def process_period(label: str, date_hour: str) -> dict:
    """
    Process one hour of GH Archive data.
    Returns: {
        "events_total": int,
        "productive_events": int,
        "unique_actors": int,
        "actors_with_location": int,
        "country_events": {country: {event_type: count}},
        "country_actor_count": {country: int},
    }
    """
    print(f"\n{'='*60}")
    print(f"Period: {label} — {date_hour}")
    print(f"{'='*60}")

    # Step 1: Download & filter events
    print("\n[1] Downloading GH Archive data...")
    events = download_gharchive(date_hour)
    if not events:
        return {}

    print(f"  Total events: {len(events):,}")

    # Step 2: Filter to productive events and extract actors
    productive_events = [e for e in events if e.get("type") in PRODUCTIVE_EVENT_TYPES]
    print(f"  Productive events: {len(productive_events):,} ({len(productive_events)/len(events)*100:.1f}%)")

    # Extract unique actor logins (sample for speed)
    actor_event_counts = defaultdict(lambda: defaultdict(int))
    for e in productive_events:
        login = e.get("actor", {}).get("login", "")
        if login:
            event_type = PRODUCTIVE_EVENT_TYPES[e["type"]]
            actor_event_counts[login][event_type] += 1

    all_actors = list(actor_event_counts.keys())
    print(f"  Unique actors: {len(all_actors):,}")

    # Sample if too many
    if len(all_actors) > MAX_USERS_PER_PERIOD:
        import random
        random.seed(42)
        sampled_actors = random.sample(all_actors, MAX_USERS_PER_PERIOD)
        print(f"  → Sampling {MAX_USERS_PER_PERIOD} actors for API calls")
    else:
        sampled_actors = all_actors

    # Step 3: Scrape user locations via GitHub API
    print(f"\n[2] Scraping user locations (max {len(sampled_actors)} API calls)...")
    actor_country = {}
    located = 0
    unlocated = 0
    not_found = 0

    for i, login in enumerate(sampled_actors):
        if i % 50 == 0 and i > 0:
            print(f"  Progress: {i}/{len(sampled_actors)} users ({located} with country)")

        user_data = github_api(f"/users/{login}")
        if user_data is None:
            not_found += 1
            continue

        raw_location = user_data.get("location")
        country = normalize_location(raw_location)
        if country:
            actor_country[login] = country
            located += 1
        else:
            unlocated += 1

    print(f"\n  Located: {located} | No country match: {unlocated} | Not found: {not_found}")
    print(f"  Location hit rate: {located/len(sampled_actors)*100:.1f}%")

    # Step 4: Aggregate by country
    country_events = defaultdict(lambda: defaultdict(int))
    country_actors = defaultdict(set)

    for login, events_by_type in actor_event_counts.items():
        country = actor_country.get(login)
        if not country:
            continue
        for event_type, count in events_by_type.items():
            country_events[country][event_type] += count
        country_actors[country].add(login)

    # Convert to plain dicts for reporting
    country_events_plain = {c: dict(e) for c, e in country_events.items()}
    country_actor_count = {c: len(actors) for c, actors in country_actors.items()}

    return {
        "events_total": len(events),
        "productive_events": len(productive_events),
        "unique_actors": len(all_actors),
        "sampled_actors": len(sampled_actors),
        "actors_with_location": located,
        "country_events": country_events_plain,
        "country_actor_count": country_actor_count,
    }


def print_country_table(results: dict[str, dict], top_n: int = 20):
    """Print a comparison table: 2022 vs 2024 per-country event totals."""
    print(f"\n{'='*70}")
    print("COUNTRY COMPARISON: 2022 vs 2024 (sampled 1-hour window each)")
    print(f"{'='*70}")

    # Gather all countries
    all_countries = set()
    for r in results.values():
        all_countries.update(r.get("country_events", {}).keys())

    # Build rows: country, 2022_total, 2024_total, pct_change
    rows = []
    labels = list(results.keys())
    for country in all_countries:
        counts = []
        for label in labels:
            evs = results[label].get("country_events", {}).get(country, {})
            counts.append(sum(evs.values()))
        rows.append((country, *counts))

    # Sort by 2024 total (or latest period)
    rows.sort(key=lambda r: r[-1], reverse=True)

    header = f"{'Country':<8}" + "".join(f"{label:>12}" for label in labels)
    if len(labels) == 2:
        header += f"{'Δ%':>10}"
    print(header)
    print("-" * len(header))

    for row in rows[:top_n]:
        country = row[0]
        vals = row[1:]
        line = f"{country:<8}" + "".join(f"{v:>12,}" for v in vals)
        if len(vals) == 2 and vals[0] > 0:
            pct = (vals[1] - vals[0]) / vals[0] * 100
            line += f"{pct:>+10.1f}%"
        elif len(vals) == 2:
            line += f"{'N/A':>10}"
        print(line)

    print()

    # Summary stats
    for label, r in results.items():
        print(f"[{label}] Total events: {r.get('events_total',0):,} | "
              f"Productive: {r.get('productive_events',0):,} | "
              f"Unique actors: {r.get('unique_actors',0):,} | "
              f"Located: {r.get('actors_with_location',0)}/{r.get('sampled_actors',0)}")


def main():
    print("GitHub Productivity by Country — Test Run")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data: {', '.join(d for _, d in SAMPLE_DATES)}")
    print(f"Cache: {CACHE_DIR}")
    print()

    results = {}
    for label, date_hour in SAMPLE_DATES:
        r = process_period(label, date_hour)
        if r:
            results[label] = r

    if results:
        print_country_table(results)

        # Save results to JSON
        output_file = Path("/tmp/github_productivity_results.json")
        with output_file.open("w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {output_file}")

    print(f"\nDone: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
