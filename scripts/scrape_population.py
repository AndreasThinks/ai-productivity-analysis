#!/usr/bin/env python3
# /// script
# dependencies = ["joblib", "scikit-learn", "requests"]
# ///
"""
Population Scrape — lightweight scoring of GitHub accounts for country-quarter IV

Generates a per-country-quarter AI adoption fraction (pct_ai_users) by:
  1. Sampling random GitHub accounts from GH Archive with parseable location fields
  2. Light scrape: 5 repos max, 100 commits max, NO file sampling (~25 API calls/account)
  3. Extracting behavioural features using the same pipeline as the classifier
  4. Scoring with the trained RF pkl
  5. Aggregating to per-country-quarter adoption rates

Target: 3,000 accounts covering all 54 panel countries (≥15 per country).
Estimated runtime: ~15 hours at 1s API delay.

Key design constraints:
- Resume-safe: status CSV tracks every decision, restart skips completed accounts
- Memory-safe: no large JSON loaded whole; process account-by-account
- Light mode: no file sampling, fewer repos, capped commits — ~25 calls vs 122 in classifier scraper
- Location-first sampling: only process accounts with a GH profile location that
  maps to a panel country
"""

import os
import json
import csv
import time
import gzip
import re
import socket
import random
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import urllib.request
import urllib.error

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

PROJECT_ROOT = Path("/home/avery/projects/ai_productivity_analysis")
DATA_DIR     = PROJECT_ROOT / "data"

# Reuse the GH Archive cache from the classifier scraper — already downloaded
GH_ARCHIVE_CACHE_DIR = DATA_DIR / "classifier_cache_full"
# Population-specific cache — scraped account data
POP_CACHE_DIR = DATA_DIR / "population_cache"
POP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH       = DATA_DIR / "classifier_model.pkl"
STATUS_PATH      = DATA_DIR / "population_scrape_status.csv"
FEATURES_PATH    = DATA_DIR / "population_features.csv"
SCORES_PATH      = DATA_DIR / "population_scores.csv"
ADOPTION_PATH    = DATA_DIR / "country_quarter_ai_adoption.csv"

# Target accounts per country (panel has 54 countries)
TARGET_PER_COUNTRY    = 30   # enough for stable rate estimates
MAX_TOTAL_ACCOUNTS    = 3000
PANEL_MIN_ACCOUNTS    = 15   # drop country-quarters with fewer scored accounts

# Light scrape caps (vs full classifier scraper)
MAX_REPOS_PER_ACCOUNT    = 5    # vs effectively unlimited in classifier
MAX_COMMITS_PER_ACCOUNT  = 100  # vs 200 per repo in classifier
SKIP_FILE_SAMPLING       = True # ~30-40 fewer API calls per account

# Rate limiting — same as classifier scraper
API_DELAY                   = 1.0
REQUEST_TIMEOUT             = 15
MAX_RETRIES                 = 5
SECONDARY_RATE_LIMIT_FLOOR  = 60
NETWORK_RETRY_FLOOR         = 120
NETWORK_MAX_RETRIES         = 8
CONSECUTIVE_NETWORK_FAIL_LIMIT = 5
CIRCUIT_BREAKER_PAUSE       = 300

# Temporal windows — must match classifier training
PRE_START   = datetime(2022, 1, 1)
PRE_CUTOFF  = datetime(2024, 1, 1)
POST_START  = datetime(2024, 1, 1)
MIN_PRE_COMMITS  = 5   # relaxed from 10 — population scoring, not training
MIN_POST_COMMITS = 5

# GH Archive hours — reuse same set as classifier scraper (likely cached)
GH_ARCHIVE_HOURS = [
    ("2024-11-05", 10),
    ("2024-11-07", 16),
    ("2025-01-13", 9),
    ("2025-01-13", 18),
    ("2025-01-15", 3),
    ("2025-01-15", 14),
    ("2025-01-17", 11),
    ("2025-01-17", 21),
    ("2025-03-04", 8),
    ("2025-03-04", 20),
    ("2025-03-06", 12),
    ("2025-03-07", 5),
]

# AI tool markers — accounts with these are NOT negatives; exclude from random sample
AI_MARKER_RE = re.compile(
    r"(noreply@anthropic\.com|claude\.ai/code|noreply@aider\.chat|aider@aider\.chat"
    r"|copilot\[bot\]|kiro.agent|noreply@github\.com.*copilot)",
    re.IGNORECASE,
)

random.seed(99)  # different seed from classifier scraper


# ---------------------------------------------------------------------------
# Country map (reused from build_panel.py)
# ---------------------------------------------------------------------------

COUNTRY_NAME_MAP = {
    # United States
    "united states of america": "US", "united states": "US", "usa": "US",
    "us": "US", "u.s.": "US", "u.s.a.": "US", "united state": "US",
    "san francisco": "US", "new york": "US", "san francisco, ca": "US",
    "new york, ny": "US", "seattle, wa": "US", "austin, tx": "US",
    "los angeles, ca": "US", "chicago, il": "US", "boston, ma": "US",
    "cambridge, ma": "US", "washington, dc": "US", "washington dc": "US",
    "brooklyn, ny": "US", "oakland, ca": "US", "san jose, ca": "US",
    "menlo park": "US", "mountain view": "US", "mountain view, ca": "US",
    "palo alto": "US", "seattle": "US", "austin": "US", "boston": "US",
    "chicago": "US", "los angeles": "US", "portland": "US",
    "portland, or": "US", "denver": "US", "denver, co": "US",
    "atlanta": "US", "atlanta, ga": "US", "houston": "US",
    "houston, tx": "US", "dallas": "US", "dallas, tx": "US",
    "san diego": "US", "san diego, ca": "US", "phoenix": "US",
    "hawaii": "US", "oklahoma city": "US", "oklahoma city, oklahoma": "US",
    "san jose california": "US",
    # US state abbreviations (common standalone entries)
    "ca": "US", "ny": "US", "tx": "US", "wa": "US", "or": "US",
    "ma": "US", "il": "US", "co": "US", "ga": "US", "fl": "US",
    "tn": "US", "nc": "US", "va": "US", "pa": "US", "oh": "US",
    # United Kingdom
    "united kingdom": "GB", "uk": "GB", "england": "GB", "london": "GB",
    "great britain": "GB", "scotland": "GB", "wales": "GB",
    "manchester": "GB", "birmingham": "GB", "leeds": "GB",
    "cambridge": "GB", "oxford": "GB", "bristol": "GB",
    "edinburgh": "GB", "glasgow": "GB", "cornwall": "GB",
    # Canada
    "canada": "CA", "toronto": "CA", "vancouver": "CA", "montreal": "CA",
    "ottawa": "CA", "calgary": "CA", "edmonton": "CA",
    "nanaimo, bc": "CA", "nanaimo": "CA", "bc": "CA",
    "ontario": "CA", "cornwall/kingston ontario": "CA",
    # Germany
    "germany": "DE", "berlin": "DE", "munich": "DE", "hamburg": "DE",
    "frankfurt": "DE", "cologne": "DE", "düsseldorf": "DE",
    "stuttgart": "DE", "dortmund": "DE", "essen": "DE",
    # France
    "france": "FR", "paris": "FR", "lyon": "FR", "marseille": "FR",
    "toulouse": "FR", "nice": "FR", "nantes": "FR", "strasbourg": "FR",
    "bordeaux": "FR",
    # Singapore
    "singapore": "SG", "sg": "SG",
    # Finland
    "finland": "FI", "helsinki": "FI",
    # South Korea
    "republic of korea": "KR", "south korea": "KR", "korea": "KR",
    "seoul": "KR", "s. korea": "KR", "busan": "KR", "대한민국": "KR",
    # Japan
    "japan": "JP", "tokyo": "JP", "osaka": "JP", "kyoto": "JP",
    "tochigi": "JP", "tochigi,tochigi": "JP",
    # Australia
    "australia": "AU", "sydney": "AU", "melbourne": "AU",
    "brisbane": "AU", "perth": "AU", "adelaide": "AU",
    # Sweden
    "sweden": "SE", "stockholm": "SE", "gothenburg": "SE",
    # Netherlands
    "netherlands": "NL", "the netherlands": "NL", "amsterdam": "NL",
    "rotterdam": "NL", "the hague": "NL", "venlo": "NL",
    "venlo (the netherlands)": "NL",
    # Denmark
    "denmark": "DK", "copenhagen": "DK",
    # New Zealand
    "new zealand": "NZ", "auckland": "NZ", "wellington": "NZ",
    # Norway
    "norway": "NO", "oslo": "NO",
    # Austria
    "austria": "AT", "vienna": "AT",
    # Switzerland
    "switzerland": "CH", "zurich": "CH", "zürich": "CH", "geneva": "CH",
    "bern": "CH",
    # Israel
    "israel": "IL", "tel aviv": "IL",
    # China
    "china": "CN", "beijing": "CN", "shanghai": "CN", "shenzhen": "CN",
    "hangzhou": "CN", "chengdu": "CN", "chengdu,sichuan": "CN",
    "guangzhou": "CN", "wuhan": "CN", "nanjing": "CN",
    # Estonia
    "estonia": "EE", "tallinn": "EE",
    # Ireland
    "ireland": "IE", "dublin": "IE",
    # Spain
    "spain": "ES", "madrid": "ES", "barcelona": "ES", "seville": "ES",
    "canary islands": "ES",
    # Belgium
    "belgium": "BE", "brussels": "BE",
    # Portugal
    "portugal": "PT", "lisbon": "PT", "porto": "PT",
    # Czech Republic
    "czech republic": "CZ", "czechia": "CZ", "prague": "CZ",
    # Italy
    "italy": "IT", "milan": "IT", "rome": "IT", "naples": "IT",
    "turin": "IT",
    # Taiwan
    "taiwan": "TW", "taipei": "TW",
    # Russia
    "russian federation": "RU", "russia": "RU", "moscow": "RU",
    "saint-petersburg": "RU", "st. petersburg": "RU",
    "saint petersburg": "RU", "novosibirsk": "RU",
    # Brazil
    "brazil": "BR", "brasil": "BR", "são paulo": "BR", "sao paulo": "BR",
    "rio de janeiro": "BR", "rio": "BR", "florianópolis": "BR",
    "brasil, mg": "BR", "americana-sp, brasil": "BR",
    "florianópolis / sc / brazil": "BR", "jundiaí, sp": "BR",
    "são paulo - sp": "BR", "quixadá": "BR",
    # India
    "india": "IN", "bangalore": "IN", "bengaluru": "IN", "mumbai": "IN",
    "delhi": "IN", "new delhi": "IN", "new delhi": "IN", "new delhi": "IN",
    "new delhi": "IN", "new elhi": "IN", "new delhi": "IN",
    "hyderabad": "IN", "chennai": "IN", "pune": "IN",
    "kolkata": "IN", "ahmedabad": "IN", "jaipur": "IN",
    "nashik": "IN", "kochi": "IN", "banglore": "IN",
    "panchkula, haryana": "IN", "maharashtra": "IN",
    "panchkula": "IN", "haryana": "IN",
    # Ukraine
    "ukraine": "UA", "kyiv": "UA", "cherkassy": "UA",
    "kharkiv": "UA", "odessa": "UA", "lviv": "UA",
    # Bangladesh
    "bangladesh": "BD", "dhaka": "BD",
    # Poland
    "poland": "PL", "warsaw": "PL", "krakow": "PL", "kraków": "PL",
    "wroclaw": "PL", "wrocław": "PL", "gdansk": "PL",
    # Pakistan
    "pakistan": "PK", "karachi": "PK", "lahore": "PK",
    "pakistan punjab lahore": "PK", "lahore, punjab pakistan": "PK",
    "islamabad": "PK",
    # Kenya
    "kenya": "KE", "nairobi": "KE",
    # Egypt
    "egypt": "EG", "cairo": "EG",
    # Turkey
    "turkey": "TR", "türkiye": "TR", "istanbul": "TR", "ankara": "TR",
    "i̇stanbul": "TR", "ankara/türkiye": "TR",
    # Hungary
    "hungary": "HU", "budapest": "HU",
    # Latvia
    "latvia": "LV", "riga": "LV",
    # Lithuania
    "lithuania": "LT", "vilnius": "LT",
    # Croatia
    "croatia": "HR", "zagreb": "HR",
    # Slovakia
    "slovakia": "SK", "bratislava": "SK",
    # Slovenia
    "slovenia": "SI", "ljubljana": "SI",
    # Romania
    "romania": "RO", "bucharest": "RO", "cluj": "RO",
    # Bulgaria
    "bulgaria": "BG", "sofia": "BG",
    # Greece
    "greece": "GR", "athens": "GR", "thessaloniki": "GR",
    # Moldova
    "moldova": "MD", "chisinau": "MD", "chișinău": "MD",
    "chisinau, moldova": "MD",
    # UAE
    "united arab emirates": "AE", "uae": "AE", "dubai": "AE",
    "abu dhabi": "AE",
    # Saudi Arabia
    "saudi arabia": "SA", "riyadh": "SA", "jeddah": "SA",
    # South Africa
    "south africa": "ZA", "cape town": "ZA", "johannesburg": "ZA",
    "durban": "ZA",
    # Nigeria
    "nigeria": "NG", "lagos": "NG", "abuja": "NG",
    # Ethiopia
    "ethiopia": "ET", "addis ababa": "ET", "addis ababa, ethiopia": "ET",
    # Madagascar
    "madagascar": "MG",
    # Mexico
    "mexico": "MX", "mexico city": "MX", "ciudad de méxico": "MX",
    "guadalajara": "MX", "monterrey": "MX",
    "merida, yucatan": "MX", "merida": "MX",
    # Argentina
    "argentina": "AR", "buenos aires": "AR", "córdoba": "AR",
    # Colombia
    "colombia": "CO", "bogota": "CO", "bogotá": "CO", "medellín": "CO",
    # Chile
    "chile": "CL", "santiago": "CL",
    # Malaysia
    "malaysia": "MY", "kuala lumpur": "MY",
    # Thailand
    "thailand": "TH", "bangkok": "TH",
    # Indonesia
    "indonesia": "ID", "jakarta": "ID",
    # Philippines
    "philippines": "PH", "manila": "PH",
    # Vietnam
    "vietnam": "VN", "viet nam": "VN", "ho chi minh city": "VN",
    "hanoi": "VN", "hcmc": "VN",
    # Sri Lanka
    "sri lanka": "LK", "colombo": "LK",
    # Nepal
    "nepal": "NP", "kathmandu": "NP",
    # Iraq
    "iraq": "IQ", "baghdad": "IQ", "iraq - sulaimaiyah": "IQ",
    # Ecuador
    "ecuador": "EC", "quito": "EC", "quito, ecuador.": "EC",
    # Hong Kong
    "hong kong": "HK",
    # Bharat (Hindi name for India)
    "bharat": "IN",
    # Additional cities/regions from observed failures
    "sevilla": "ES", "seville": "ES", "tarragona": "ES",
    "dresden": "DE", "heidelberg": "DE",
    "gdańsk": "PL", "gdansk": "PL",
    "hull": "GB",
    "florida": "US", "gainesville, florida": "US", "gainesville": "US",
    "iowa": "US", "des moines": "US", "des moines,, ia": "US",
    "indore": "IN", "patna": "IN", "patna, bihar india": "IN",
    "zirakpur, punjab": "IN", "zirakpur": "IN",
    "karachi pakistan": "PK", "e11/3 islamabad": "PK",
    "fortaleza": "BR", "fortaleza,ce": "BR", "salvador": "BR",
    "salvador,ba": "BR", "piauí, teresina": "BR",
    "iran": "IR", "tehran": "IR",
    "uzbekistan": "UZ", "tashkent": "UZ", "tashkent, uzbekistan": "UZ",
    "serbia": "RS", "belgrade": "RS",
    "brno": "CZ", "brno, cz": "CZ",
    "jerusalem": "IL",
    "pekanbaru": "ID",
    "lisbon": "PT", "lisboa": "PT", "lisboa - portugal": "PT",
    "lagos state": "NG",
}

# Countries present in the panel dataset
PANEL_COUNTRIES = {
    "US", "GB", "DE", "FR", "IN", "CN", "BR", "CA", "AU", "RU",
    "JP", "KR", "NL", "SE", "CH", "PL", "NG", "BD", "KE", "TR",
    "ES", "IT", "NO", "DK", "FI", "PT", "BE", "AT", "GR", "CZ",
    "RO", "HU", "UA", "PK", "EG", "ZA", "MX", "AR", "CO", "CL",
    "SG", "MY", "ID", "TH", "PH", "VN", "IL", "AE", "SA", "TW",
    "IE", "NZ", "LK", "NP",
}


def parse_location(location_str):
    """Map a GitHub profile location string to an ISO2 country code."""
    if not location_str:
        return None
    key = location_str.strip().lower()
    key = re.sub(r"\s*\(.*?\)", "", key).strip()
    # Direct lookup
    if key in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[key]
    # Try last comma-separated segment (e.g. "Berlin, Germany")
    parts = [p.strip() for p in key.split(",")]
    for part in reversed(parts):
        if part in COUNTRY_NAME_MAP:
            return COUNTRY_NAME_MAP[part]
    # Try first segment
    if parts[0] in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[parts[0]]
    return None


# ---------------------------------------------------------------------------
# HTTP helpers (same as other scripts)
# ---------------------------------------------------------------------------

def _gh_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "population-scraper/1.0",
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


def gh_get(url):
    headers = _gh_headers()
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
                wait = max(5, int(reset_ts) - int(time.time()) + 5) if reset_ts else SECONDARY_RATE_LIMIT_FLOOR
                print(f"    Rate limit (403), sleeping {wait}s...")
                time.sleep(wait)
            elif e.code in (404, 409, 451):
                return None
            elif e.code >= 500:
                time.sleep(API_DELAY * (2 ** attempt))
            else:
                return None
        except Exception as exc:
            if _is_network_error(exc):
                network_attempts += 1
                wait = NETWORK_RETRY_FLOOR * network_attempts
                print(f"    Network error ({exc}), sleeping {wait}s...")
                if network_attempts >= NETWORK_MAX_RETRIES:
                    raise NetworkError(f"Network failed after {NETWORK_MAX_RETRIES} attempts") from exc
                time.sleep(wait)
            else:
                return None
    return None


def _sleep():
    time.sleep(API_DELAY)


# ---------------------------------------------------------------------------
# GH Archive candidate discovery
# ---------------------------------------------------------------------------

def _gh_archive_path(date_str, hour):
    return DATA_DIR / "gh_archive_cache" / f"{date_str}-{hour}.jsonl"


def _download_gh_archive(date_str, hour):
    """Download and decompress a GH Archive hour file, return path."""
    path = _gh_archive_path(date_str, hour)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path
    url = f"https://data.gharchive.org/{date_str}-{hour}.json.gz"
    print(f"  Downloading GH Archive: {url}")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp:
            compressed = resp.read()
        data = gzip.decompress(compressed)
        path.write_bytes(data)
        print(f"    Saved {len(data)//1024}KB → {path}")
        return path
    except Exception as e:
        print(f"    Failed to download {url}: {e}")
        return None


def collect_candidates():
    """
    Scan GH Archive PushEvents for unique actor logins.
    Returns a dict: {login: set_of_event_dates} for accounts with location
    that maps to a panel country.
    We collect logins first, then resolve locations via the profile API.
    """
    print("\n=== STAGE 1: Collecting candidates from GH Archive ===")

    # Try classifier_cache_full directory for already-cached archive files
    classifier_archive_dir = DATA_DIR / "classifier_cache_full"

    all_logins = set()
    for date_str, hour in GH_ARCHIVE_HOURS:
        # Check if already cached by the classifier scraper
        cached_path = DATA_DIR / "gh_archive_cache" / f"{date_str}-{hour}.jsonl"
        alt_path    = DATA_DIR / f"gharchive_{date_str}-{hour}.jsonl"

        path = None
        if cached_path.exists():
            path = cached_path
        elif alt_path.exists():
            path = alt_path
        else:
            path = _download_gh_archive(date_str, hour)

        if not path or not path.exists():
            print(f"  Skipping {date_str}-{hour} (not available)")
            continue

        hour_logins = set()
        try:
            with open(path, "r", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if event.get("type") != "PushEvent":
                        continue
                    actor = event.get("actor", {})
                    login = actor.get("login", "")
                    if not login or "[bot]" in login:
                        continue
                    # Quick filter: skip if any AI marker in payload
                    payload_str = json.dumps(event.get("payload", {}))
                    if AI_MARKER_RE.search(payload_str):
                        continue
                    hour_logins.add(login)
        except Exception as e:
            print(f"  Error reading {path}: {e}")
            continue

        print(f"  {date_str}-{hour}: {len(hour_logins)} unique logins")
        all_logins.update(hour_logins)

    print(f"\nTotal unique candidate logins: {len(all_logins)}")
    return list(all_logins)


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

def load_status():
    """Load the scrape status CSV. Returns dict: login -> status."""
    status = {}
    if not STATUS_PATH.exists():
        return status
    with open(STATUS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            status[row["login"]] = row
    return status


def append_status(row):
    """Append a single row to the status CSV."""
    fieldnames = ["login", "status", "country", "classifier_score",
                  "pre_commits", "post_commits", "timestamp"]
    write_header = not STATUS_PATH.exists()
    with open(STATUS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def load_features():
    """Load already-written feature rows."""
    if not FEATURES_PATH.exists():
        return []
    with open(FEATURES_PATH, newline="") as f:
        return list(csv.DictReader(f))


def append_features(row):
    """Append a feature row. Writes header on first call."""
    write_header = not FEATURES_PATH.exists()
    with open(FEATURES_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Light scrape
# ---------------------------------------------------------------------------

def scrape_account_light(login):
    """
    Lightweight scrape: profile + up to MAX_REPOS_PER_ACCOUNT repos +
    up to MAX_COMMITS_PER_ACCOUNT commits total. No file sampling.
    Returns dict with commits, prs, location.
    """
    cache_path = POP_CACHE_DIR / f"{login}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if not cached.get("error"):
            return cached

    profile = gh_get(f"https://api.github.com/users/{login}")
    _sleep()
    if not profile:
        return {"error": "profile fetch failed", "commits": [], "prs": [], "location": None}

    location = profile.get("location")

    repos_data = gh_get(
        f"https://api.github.com/users/{login}/repos"
        f"?type=owner&sort=created&direction=asc&per_page={MAX_REPOS_PER_ACCOUNT}"
    )
    _sleep()
    if not repos_data or not isinstance(repos_data, list):
        return {"error": "repos fetch failed", "commits": [], "prs": [], "location": location}

    repos = [r["name"] for r in repos_data if not r.get("fork", False)][:MAX_REPOS_PER_ACCOUNT]

    all_commits = []
    all_prs     = []
    commits_remaining = MAX_COMMITS_PER_ACCOUNT

    for repo_name in repos:
        if commits_remaining <= 0:
            break
        per_page = min(100, commits_remaining)
        url = (
            f"https://api.github.com/repos/{login}/{repo_name}/commits"
            f"?author={login}&per_page={per_page}&page=1"
        )
        data = gh_get(url)
        _sleep()
        if data and isinstance(data, list):
            for item in data:
                c = item.get("commit", {})
                all_commits.append({
                    "sha":        item.get("sha", ""),
                    "message":    c.get("message", ""),
                    "created_at": (c.get("committer") or c.get("author") or {}).get("date", ""),
                    "repo":       repo_name,
                    "file_sampled": False,
                    "has_test_file": None,
                    "has_impl_file": None,
                })
            commits_remaining -= len(data)

        # PRs: just first page
        pr_url = (
            f"https://api.github.com/repos/{login}/{repo_name}/pulls"
            f"?state=all&per_page=50&page=1"
        )
        pr_data = gh_get(pr_url)
        _sleep()
        if pr_data and isinstance(pr_data, list):
            for pr in pr_data:
                user = pr.get("user") or {}
                if user.get("login") != login:
                    continue
                body = pr.get("body") or ""
                all_prs.append({
                    "created_at":  pr.get("created_at", ""),
                    "body_length": len(body),
                })

    result = {
        "login":    login,
        "location": location,
        "commits":  all_commits,
        "prs":      all_prs,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Feature extraction (same logic as classify scraper, minus file sampling)
# ---------------------------------------------------------------------------

def _parse_dt(s):
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)
    except (ValueError, AttributeError):
        return None


def _count_in_window(commits, after, before=None):
    n = 0
    for c in commits:
        dt = _parse_dt(c.get("created_at"))
        if dt and dt >= after and (not before or dt < before):
            n += 1
    return n


def _deduplicate(commits):
    seen, out = set(), []
    for c in commits:
        sha = c.get("sha", "")
        if sha and sha in seen:
            continue
        seen.add(sha)
        out.append(c)
    return out


# Strip AI tool markers from commit messages before feature extraction
_STRIP_RE = re.compile(
    r"(Co-[Aa]uthored-[Bb]y:.*?(?:noreply@anthropic\.com|noreply@aider\.chat|aider@aider\.chat)[^\n]*"
    r"|🤖\s*Generated with.*?claude\.ai/code[^\n]*)",
    re.IGNORECASE,
)


def _window_features(commits, prs, after, before=None):
    window_c = [
        c for c in commits
        if _parse_dt(c.get("created_at")) is not None
        and _parse_dt(c.get("created_at")) >= after
        and (not before or _parse_dt(c.get("created_at")) < before)
    ]
    if not window_c:
        return {
            "commit_count": 0, "mean_message_length": 0.0, "active_weeks": 0,
            "repos_touched": 0, "mean_commits_per_active_week": 0.0,
            "frac_multiline": 0.0, "frac_conventional": 0.0,
            "frac_mentions_test": 0.0, "frac_has_bullets": 0.0,
            "mean_inter_commit_hours": 0.0, "frac_burst_commits": 0.0,
            "sampled_test_cowrite_rate": 0.0, "file_sample_count": 0,
            "mean_pr_body_length": 0.0, "frac_pr_has_body": 0.0,
        }

    cleaned = [_STRIP_RE.sub("", c.get("message", "")).strip() for c in window_c]
    msg_lengths = [len(m) for m in cleaned]

    active_weeks = len({_parse_dt(c["created_at"]).isocalendar()[:2]
                        for c in window_c if _parse_dt(c["created_at"])})
    repos = len({c.get("repo", "") for c in window_c if c.get("repo")})

    conv_re = re.compile(
        r"^(feat|fix|chore|refactor|docs|test|style|perf|ci|build)(\(.*\))?:",
        re.IGNORECASE,
    )
    test_re = re.compile(r"\btest[s]?\b", re.IGNORECASE)

    multiline_n    = sum(1 for m in cleaned if "\n" in m)
    conventional_n = sum(1 for m in cleaned if conv_re.match(m))
    test_n         = sum(1 for m in cleaned if test_re.search(m))
    bullets_n      = sum(1 for m in cleaned if "- " in m or "* " in m)

    sorted_w = sorted(window_c, key=lambda c: _parse_dt(c.get("created_at")) or datetime.min)
    inter = []
    for i in range(1, len(sorted_w)):
        dt1 = _parse_dt(sorted_w[i-1].get("created_at"))
        dt2 = _parse_dt(sorted_w[i].get("created_at"))
        if dt1 and dt2:
            inter.append((dt2 - dt1).total_seconds() / 3600.0)

    mean_inter = sum(inter) / len(inter) if inter else 0.0
    frac_burst = sum(1 for h in inter if h <= 2.0) / len(inter) if inter else 0.0

    # PR features
    window_p = [
        pr for pr in prs
        if _parse_dt(pr.get("created_at")) is not None
        and _parse_dt(pr.get("created_at")) >= after
        and (not before or _parse_dt(pr.get("created_at")) < before)
    ]
    if window_p:
        bl = [pr.get("body_length", 0) for pr in window_p]
        mean_body    = sum(bl) / len(bl)
        frac_has_body = sum(1 for b in bl if b > 50) / len(bl)
    else:
        mean_body = frac_has_body = 0.0

    n = len(window_c)
    return {
        "commit_count":                  n,
        "mean_message_length":           round(sum(msg_lengths) / n, 2),
        "active_weeks":                  active_weeks,
        "repos_touched":                 repos,
        "mean_commits_per_active_week":  round(n / max(active_weeks, 1), 2),
        "frac_multiline":                round(multiline_n / n, 3),
        "frac_conventional":             round(conventional_n / n, 3),
        "frac_mentions_test":            round(test_n / n, 3),
        "frac_has_bullets":              round(bullets_n / n, 3),
        "mean_inter_commit_hours":       round(mean_inter, 2),
        "frac_burst_commits":            round(frac_burst, 3),
        "sampled_test_cowrite_rate":     0.0,   # not computed in light mode
        "file_sample_count":             0,
        "mean_pr_body_length":           round(mean_body, 2),
        "frac_pr_has_body":              round(frac_has_body, 3),
    }


def extract_features_for_account(login, data):
    """Extract pre/post/delta features. Returns row dict or None if fails both-window filter."""
    commits = _deduplicate(data.get("commits", []))
    prs     = data.get("prs", [])

    pre_count  = _count_in_window(commits, PRE_START, PRE_CUTOFF)
    post_count = _count_in_window(commits, POST_START)

    if pre_count < MIN_PRE_COMMITS or post_count < MIN_POST_COMMITS:
        return None, pre_count, post_count

    pre_f  = _window_features(commits, prs, PRE_START, PRE_CUTOFF)
    post_f = _window_features(commits, prs, POST_START)

    row = {"login": login}
    for k, v in pre_f.items():
        row[f"pre_{k}"] = v
    for k, v in post_f.items():
        row[f"post_{k}"] = v
    for k in pre_f:
        row[f"delta_{k}"] = round(post_f[k] - pre_f[k], 3)

    return row, pre_count, post_count


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def load_model():
    import joblib
    pkg = joblib.load(MODEL_PATH)
    return pkg["model"], pkg["imputer"], pkg["feature_cols"]


def score_row(model, imputer, feature_cols, row):
    """Score a single feature row dict. Returns probability [0,1]."""
    import numpy as np
    vals = [row.get(fc, 0.0) for fc in feature_cols]
    X = imputer.transform([vals])
    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

QUARTER_MAP = {1: "Q1", 2: "Q1", 3: "Q1",
               4: "Q2", 5: "Q2", 6: "Q2",
               7: "Q3", 8: "Q3", 9: "Q3",
               10: "Q4", 11: "Q4", 12: "Q4"}


def quarter_label(dt_str, window="post"):
    """
    For the population IV we use a simple pre/post split rather than
    per-account dating. Pre-window (2022-2023) is labeled by the midpoint
    year-quarter; post-window (2024+) likewise.
    We assign each account a PRE score and POST score, then aggregate by
    country × period to get pct_ai_users.
    """
    # We actually aggregate by country × period (pre/post), not per quarter.
    # Quarterly granularity requires knowing when each account adopted AI —
    # we don't have that for the random population. Use year-level.
    pass


def build_adoption_table(scores_path):
    """
    Aggregate population scores to country × year adoption fractions.
    Uses the fact that each account has a pre-window and post-window score —
    we treat them as separate observations:
      - pre score → maps to year 2022 and 2023 (pre-period)
      - post score → maps to year 2024 (post-period)
    """
    import pandas as pd
    import numpy as np

    df = pd.read_csv(scores_path)
    print(f"\nScored accounts: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")

    records = []
    for _, row in df.iterrows():
        country = row["country"]
        if country not in PANEL_COUNTRIES:
            continue
        # Pre score → 2022 and 2023 observations
        records.append({"country": country, "year": 2022,
                         "ai_prob": row["pre_classifier_score"]})
        records.append({"country": country, "year": 2023,
                         "ai_prob": row["pre_classifier_score"]})
        # Post score → 2024
        records.append({"country": country, "year": 2024,
                         "ai_prob": row["post_classifier_score"]})

    agg = pd.DataFrame(records)
    summary = (
        agg.groupby(["country", "year"])
           .agg(pct_ai_users=("ai_prob", "mean"),
                n_accounts=("ai_prob", "count"))
           .reset_index()
    )

    # Drop thin groups
    before = len(summary)
    summary = summary[summary["n_accounts"] >= PANEL_MIN_ACCOUNTS]
    print(f"Country-year groups: {before} total, {len(summary)} with ≥{PANEL_MIN_ACCOUNTS} accounts")
    print(f"Countries covered: {summary['country'].nunique()}")

    summary.to_csv(ADOPTION_PATH, index=False)
    print(f"Saved adoption table → {ADOPTION_PATH}")
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    import joblib

    print("=" * 65)
    print("POPULATION SCRAPE")
    print(f"Target: {MAX_TOTAL_ACCOUNTS} accounts across {len(PANEL_COUNTRIES)} panel countries")
    print("=" * 65)

    # Load model once
    print("\nLoading classifier model...")
    model, imputer, feature_cols = load_model()
    print(f"  Model loaded. Feature cols: {len(feature_cols)}")

    # Load resume state
    status = load_status()
    done_logins    = {l for l, s in status.items() if s["status"] in ("scored", "skipped", "error")}
    scored_logins  = {l for l, s in status.items() if s["status"] == "scored"}
    country_counts = defaultdict(int)
    for l, s in status.items():
        if s["status"] == "scored" and s.get("country") in PANEL_COUNTRIES:
            country_counts[s["country"]] += 1

    print(f"\nResume state: {len(scored_logins)} scored, {len(done_logins)} total processed")
    print(f"Countries with ≥{TARGET_PER_COUNTRY} accounts: "
          f"{sum(1 for c in PANEL_COUNTRIES if country_counts[c] >= TARGET_PER_COUNTRY)}"
          f"/{len(PANEL_COUNTRIES)}")

    # Check if we're already done
    total_scored = len(scored_logins)
    if total_scored >= MAX_TOTAL_ACCOUNTS:
        print(f"\nTarget reached ({total_scored} accounts). Building adoption table.")
        build_adoption_table(SCORES_PATH)
        return

    # Collect candidates
    candidates = collect_candidates()
    # Shuffle and filter already-done
    random.shuffle(candidates)
    candidates = [l for l in candidates if l not in done_logins]
    print(f"\nFresh candidates to process: {len(candidates)}")

    # Open scores file for appending
    scores_write_header = not SCORES_PATH.exists()
    scores_fieldnames   = ["login", "country", "pre_classifier_score",
                           "post_classifier_score", "pre_commits", "post_commits"]

    consecutive_failures = 0

    for login in candidates:
        if total_scored >= MAX_TOTAL_ACCOUNTS:
            print(f"\nTarget reached ({MAX_TOTAL_ACCOUNTS}). Stopping.")
            break

        # Check if this country is saturated
        # (don't skip — we need diversity, but log it)

        try:
            data = scrape_account_light(login)
        except NetworkError:
            consecutive_failures += 1
            print(f"  {login}: NetworkError (skip)")
            append_status({"login": login, "status": "network_error", "country": "",
                           "classifier_score": "", "pre_commits": "", "post_commits": "",
                           "timestamp": datetime.now().isoformat()})
            if consecutive_failures >= CONSECUTIVE_NETWORK_FAIL_LIMIT:
                print(f"  Circuit breaker — pausing {CIRCUIT_BREAKER_PAUSE}s")
                time.sleep(CIRCUIT_BREAKER_PAUSE)
                consecutive_failures = 0
            continue

        consecutive_failures = 0

        if data.get("error"):
            append_status({"login": login, "status": "error", "country": "",
                           "classifier_score": "", "pre_commits": "", "post_commits": "",
                           "timestamp": datetime.now().isoformat()})
            continue

        # Resolve country
        country = parse_location(data.get("location"))
        if not country or country not in PANEL_COUNTRIES:
            append_status({"login": login, "status": "skipped", "country": country or "",
                           "classifier_score": "", "pre_commits": "", "post_commits": "",
                           "timestamp": datetime.now().isoformat()})
            continue

        # Extract features
        feat_row, pre_count, post_count = extract_features_for_account(login, data)
        if feat_row is None:
            append_status({"login": login, "status": "skipped", "country": country,
                           "classifier_score": "", "pre_commits": pre_count,
                           "post_commits": post_count,
                           "timestamp": datetime.now().isoformat()})
            continue

        # Score pre and post windows separately
        # Pre score: use pre_* features as both pre and post (score pre-period behaviour)
        pre_feat_row = {k: v for k, v in feat_row.items()}
        # For pre score: set post_* = pre_* so the model sees pre-period behaviour in both windows
        for k in list(pre_feat_row.keys()):
            if k.startswith("post_"):
                base = k[5:]
                pre_feat_row[k] = pre_feat_row.get(f"pre_{base}", 0.0)
        # Recompute deltas to zero
        for k in list(pre_feat_row.keys()):
            if k.startswith("delta_"):
                pre_feat_row[k] = 0.0

        pre_score  = score_row(model, imputer, feature_cols, pre_feat_row)
        post_score = score_row(model, imputer, feature_cols, feat_row)

        # Append scores
        score_row_dict = {
            "login":                login,
            "country":              country,
            "pre_classifier_score": round(pre_score, 4),
            "post_classifier_score": round(post_score, 4),
            "pre_commits":          pre_count,
            "post_commits":         post_count,
        }
        with open(SCORES_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scores_fieldnames)
            if scores_write_header:
                w.writeheader()
                scores_write_header = False
            w.writerow(score_row_dict)

        # Append features
        append_features(feat_row)

        append_status({
            "login": login, "status": "scored", "country": country,
            "classifier_score": round(post_score, 4),
            "pre_commits": pre_count, "post_commits": post_count,
            "timestamp": datetime.now().isoformat(),
        })

        country_counts[country] += 1
        total_scored += 1

        print(f"  {login} ({country}): pre={pre_count}, post={post_count}, "
              f"score={post_score:.3f}  [{total_scored}/{MAX_TOTAL_ACCOUNTS}]")

        # Progress report every 100 accounts
        if total_scored % 100 == 0:
            covered = sum(1 for c in PANEL_COUNTRIES if country_counts[c] >= TARGET_PER_COUNTRY)
            print(f"\n  === Progress: {total_scored} scored, "
                  f"{covered}/{len(PANEL_COUNTRIES)} countries at target ===\n")

    # Build adoption table from whatever we have
    print("\n=== Building country-quarter adoption table ===")
    if SCORES_PATH.exists():
        adoption = build_adoption_table(SCORES_PATH)
        print("\nTop countries by post-2024 AI adoption rate:")
        import pandas as pd
        post = adoption[adoption["year"] == 2024].sort_values("pct_ai_users", ascending=False)
        print(post.head(20).to_string(index=False))
    else:
        print("No scores file found — nothing to aggregate.")

    print("\n" + "=" * 65)
    print("POPULATION SCRAPE COMPLETE")
    print(f"  Scored: {total_scored} accounts")
    print(f"  Status: {STATUS_PATH}")
    print(f"  Scores: {SCORES_PATH}")
    print(f"  Adoption table: {ADOPTION_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
