#!/usr/bin/env python3
"""
Population Scrape v3 — rate-limit-resilient scoring with capped backoff

Fixes the v1/v2 rate-limit coma:
  1. All sleeps capped at MAX_SLEEP (5 min). No more 3600s spirals.
  2. Proactive /rate_limit check before requests when quota is low.
  3. Distinguishes primary rate limit (remaining==0) from abuse detection
     (remaining>0, 403 anyway). Abuse gets shorter, capped backoff.
  4. Global pause after 3 consecutive rate-limit hits — cools the whole
     pattern instead of burning retries per-account.
  5. Jitter on all delays so the request cadence doesn't look robotic.
  6. GraphQL batch location pre-filter (from v2) for throughput.

Output files use _v3 suffix — safe alongside v1 and v2.
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

GITHUB_TOKEN = os.environ.get("GITHUB_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("GITHUB_TOKEN environment variable not set")

PROJECT_ROOT = Path("/home/avery/projects/ai_productivity_analysis")
DATA_DIR = PROJECT_ROOT / "data"

POP_CACHE_DIR = DATA_DIR / "population_cache_v3"
POP_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DATA_DIR / "classifier_model.pkl"

# v3-specific output paths
STATUS_PATH = DATA_DIR / "population_scrape_status_v3.csv"
FEATURES_PATH = DATA_DIR / "population_features_v3.csv"
SCORES_PATH = DATA_DIR / "population_scores_v3.csv"
ADOPTION_PATH = DATA_DIR / "country_quarter_ai_adoption_v3.csv"
LOCATION_CACHE_PATH = DATA_DIR / "population_location_cache_v3.csv"

TARGET_PER_COUNTRY = 30
MAX_TOTAL_ACCOUNTS = 3000
PANEL_MIN_ACCOUNTS = 15

MAX_REPOS_PER_ACCOUNT = 5
MAX_COMMITS_PER_ACCOUNT = 100
SKIP_FILE_SAMPLING = True

# Rate limiting — v3: capped, jittered, proactive
API_DELAY = 0.6                     # REST delay (slightly higher than v2's 0.5)
GRAPHQL_BATCH_DELAY = 1.2           # GraphQL batch delay
LOCATION_BATCH_SIZE = 20
REQUEST_TIMEOUT = 15
MAX_RETRIES = 3                     # reduced from 5
MAX_SLEEP = 300                     # HARD CAP: 5 minutes max per sleep
RATE_LIMIT_MAX_ATTEMPTS = 2         # only retry twice for rate-limit 403s
ABUSE_BACKOFF_BASE = 60             # base seconds when we have quota but get 403
SECONDARY_RATE_LIMIT_FLOOR = 60
NETWORK_RETRY_FLOOR = 120
NETWORK_MAX_RETRIES = 5             # reduced from 8
CONSECUTIVE_NETWORK_FAIL_LIMIT = 5
CIRCUIT_BREAKER_PAUSE = 300
CONSECUTIVE_RATE_LIMIT_PAUSE = 3    # global pause after this many 403s in a row
GLOBAL_RATE_LIMIT_PAUSE = 300       # 5 min global cooldown

# Temporal windows
PRE_START = datetime(2022, 1, 1)
PRE_CUTOFF = datetime(2024, 1, 1)
POST_START = datetime(2024, 1, 1)
MIN_PRE_COMMITS = 5
MIN_POST_COMMITS = 5

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

AI_MARKER_RE = re.compile(
    r"(noreply@anthropic\.com|claude\.ai/code|noreply@aider\.chat|aider@aider\.chat"
    r"|copilot\[bot\]|kiro.agent|noreply@github\.com.*copilot)",
    re.IGNORECASE,
)

random.seed(2026)

# ---------------------------------------------------------------------------
# Global rate-limit state (shared across REST and GraphQL)
# ---------------------------------------------------------------------------

_rate_limit_state = {"remaining": None, "reset_at": None, "last_check": 0}
_consecutive_rate_limits = 0


def _jitter(base: float) -> float:
    """Add +/- 25% jitter to a delay so request cadence isn't robotic."""
    return base * (0.75 + 0.5 * random.random())


def _update_rate_limit(headers):
    """Update global rate limit state from response headers."""
    global _rate_limit_state
    try:
        rem = headers.get("X-RateLimit-Remaining")
        reset = headers.get("X-RateLimit-Reset")
        if rem is not None:
            _rate_limit_state["remaining"] = int(rem)
        if reset is not None:
            _rate_limit_state["reset_at"] = int(reset)
        _rate_limit_state["last_check"] = time.time()
    except (ValueError, TypeError):
        pass


def _check_global_rate_limit():
    """
    Proactive check: if we're close to the primary rate limit,
    sleep until reset (capped at MAX_SLEEP).
    """
    rem = _rate_limit_state.get("remaining")
    reset_at = _rate_limit_state.get("reset_at")
    if rem is None or reset_at is None:
        return
    if rem <= 10:
        now = time.time()
        if reset_at > now:
            sleep_for = min(reset_at - now + 5, MAX_SLEEP)
            print(f"  Proactive rate-limit pause ({rem} remaining). Sleeping {sleep_for:.0f}s...")
            time.sleep(_jitter(sleep_for))
            # Refresh state after sleep
            _rate_limit_state["remaining"] = None


def _refresh_rate_limit_state():
    """Hit /rate_limit to get ground-truth remaining quota."""
    global _rate_limit_state
    try:
        req = urllib.request.Request(
            "https://api.github.com/rate_limit",
            headers={
                "Authorization": f"Bearer {GITHUB_TOKEN}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "population-scraper-v3/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())
            core = data.get("resources", {}).get("core", {})
            _rate_limit_state["remaining"] = core.get("remaining")
            _rate_limit_state["reset_at"] = core.get("reset")
            _rate_limit_state["last_check"] = time.time()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Country map
# ---------------------------------------------------------------------------

COUNTRY_NAME_MAP = {
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
    "ca": "US", "ny": "US", "tx": "US", "wa": "US", "or": "US",
    "ma": "US", "il": "US", "co": "US", "ga": "US", "fl": "US",
    "tn": "US", "nc": "US", "va": "US", "pa": "US", "oh": "US",
    "united kingdom": "GB", "uk": "GB", "england": "GB", "london": "GB",
    "great britain": "GB", "scotland": "GB", "wales": "GB",
    "manchester": "GB", "birmingham": "GB", "leeds": "GB",
    "cambridge": "GB", "oxford": "GB", "bristol": "GB",
    "edinburgh": "GB", "glasgow": "GB", "cornwall": "GB", "hull": "GB",
    "canada": "CA", "toronto": "CA", "vancouver": "CA", "montreal": "CA",
    "ottawa": "CA", "calgary": "CA", "edmonton": "CA",
    "nanaimo, bc": "CA", "nanaimo": "CA", "bc": "CA",
    "ontario": "CA", "cornwall/kingston ontario": "CA",
    "germany": "DE", "berlin": "DE", "munich": "DE", "hamburg": "DE",
    "frankfurt": "DE", "cologne": "DE", "düsseldorf": "DE",
    "stuttgart": "DE", "dortmund": "DE", "essen": "DE",
    "dresden": "DE", "heidelberg": "DE",
    "france": "FR", "paris": "FR", "lyon": "FR", "marseille": "FR",
    "toulouse": "FR", "nice": "FR", "nantes": "FR", "strasbourg": "FR",
    "bordeaux": "FR",
    "singapore": "SG", "sg": "SG",
    "finland": "FI", "helsinki": "FI",
    "republic of korea": "KR", "south korea": "KR", "korea": "KR",
    "seoul": "KR", "s. korea": "KR", "busan": "KR", "대한민국": "KR",
    "japan": "JP", "tokyo": "JP", "osaka": "JP", "kyoto": "JP",
    "tochigi": "JP", "tochigi,tochigi": "JP",
    "australia": "AU", "sydney": "AU", "melbourne": "AU",
    "brisbane": "AU", "perth": "AU", "adelaide": "AU",
    "sweden": "SE", "stockholm": "SE", "gothenburg": "SE",
    "netherlands": "NL", "the netherlands": "NL", "amsterdam": "NL",
    "rotterdam": "NL", "the hague": "NL", "venlo": "NL",
    "venlo (the netherlands)": "NL",
    "denmark": "DK", "copenhagen": "DK",
    "new zealand": "NZ", "auckland": "NZ", "wellington": "NZ",
    "norway": "NO", "oslo": "NO",
    "austria": "AT", "vienna": "AT",
    "switzerland": "CH", "zurich": "CH", "zürich": "CH", "geneva": "CH",
    "bern": "CH",
    "israel": "IL", "tel aviv": "IL", "jerusalem": "IL",
    "china": "CN", "beijing": "CN", "shanghai": "CN", "shenzhen": "CN",
    "hangzhou": "CN", "chengdu": "CN", "chengdu,sichuan": "CN",
    "guangzhou": "CN", "wuhan": "CN", "nanjing": "CN",
    "estonia": "EE", "tallinn": "EE",
    "ireland": "IE", "dublin": "IE",
    "spain": "ES", "madrid": "ES", "barcelona": "ES", "seville": "ES",
    "canary islands": "ES", "sevilla": "ES", "tarragona": "ES",
    "belgium": "BE", "brussels": "BE",
    "portugal": "PT", "lisbon": "PT", "porto": "PT",
    "lisboa": "PT", "lisboa - portugal": "PT",
    "czech republic": "CZ", "czechia": "CZ", "prague": "CZ",
    "brno": "CZ", "brno, cz": "CZ",
    "italy": "IT", "milan": "IT", "rome": "IT", "naples": "IT",
    "turin": "IT",
    "taiwan": "TW", "taipei": "TW",
    "russian federation": "RU", "russia": "RU", "moscow": "RU",
    "saint-petersburg": "RU", "st. petersburg": "RU",
    "saint petersburg": "RU", "novosibirsk": "RU",
    "brazil": "BR", "brasil": "BR", "são paulo": "BR", "sao paulo": "BR",
    "rio de janeiro": "BR", "rio": "BR", "florianópolis": "BR",
    "brasil, mg": "BR", "americana-sp, brasil": "BR",
    "florianópolis / sc / brazil": "BR", "jundiaí, sp": "BR",
    "são paulo - sp": "BR", "quixadá": "BR", "fortaleza": "BR",
    "fortaleza,ce": "BR", "salvador": "BR", "salvador,ba": "BR",
    "piauí, teresina": "BR",
    "india": "IN", "bangalore": "IN", "bengaluru": "IN", "mumbai": "IN",
    "delhi": "IN", "new delhi": "IN", "new elhi": "IN",
    "hyderabad": "IN", "chennai": "IN", "pune": "IN",
    "kolkata": "IN", "ahmedabad": "IN", "jaipur": "IN",
    "nashik": "IN", "kochi": "IN", "banglore": "IN",
    "panchkula, haryana": "IN", "maharashtra": "IN",
    "panchkula": "IN", "haryana": "IN", "indore": "IN",
    "patna": "IN", "patna, bihar india": "IN",
    "zirakpur, punjab": "IN", "zirakpur": "IN", "bharat": "IN",
    "ukraine": "UA", "kyiv": "UA", "cherkassy": "UA",
    "kharkiv": "UA", "odessa": "UA", "lviv": "UA",
    "bangladesh": "BD", "dhaka": "BD",
    "poland": "PL", "warsaw": "PL", "krakow": "PL", "kraków": "PL",
    "wroclaw": "PL", "wrocław": "PL", "gdansk": "PL",
    "gdańsk": "PL", "gdansk": "PL",
    "pakistan": "PK", "karachi": "PK", "lahore": "PK",
    "pakistan punjab lahore": "PK", "lahore, punjab pakistan": "PK",
    "islamabad": "PK", "karachi pakistan": "PK", "e11/3 islamabad": "PK",
    "kenya": "KE", "nairobi": "KE",
    "egypt": "EG", "cairo": "EG",
    "turkey": "TR", "türkiye": "TR", "istanbul": "TR", "ankara": "TR",
    "i̇stanbul": "TR", "ankara/türkiye": "TR",
    "hungary": "HU", "budapest": "HU",
    "latvia": "LV", "riga": "LV",
    "lithuania": "LT", "vilnius": "LT",
    "croatia": "HR", "zagreb": "HR",
    "slovakia": "SK", "bratislava": "SK",
    "slovenia": "SI", "ljubljana": "SI",
    "romania": "RO", "bucharest": "RO", "cluj": "RO",
    "bulgaria": "BG", "sofia": "BG",
    "greece": "GR", "athens": "GR", "thessaloniki": "GR",
    "moldova": "MD", "chisinau": "MD", "chișinău": "MD",
    "chisinau, moldova": "MD",
    "united arab emirates": "AE", "uae": "AE", "dubai": "AE",
    "abu dhabi": "AE",
    "saudi arabia": "SA", "riyadh": "SA", "jeddah": "SA",
    "south africa": "ZA", "cape town": "ZA", "johannesburg": "ZA",
    "durban": "ZA",
    "nigeria": "NG", "lagos": "NG", "abuja": "NG", "lagos state": "NG",
    "ethiopia": "ET", "addis ababa": "ET", "addis ababa, ethiopia": "ET",
    "madagascar": "MG",
    "mexico": "MX", "mexico city": "MX", "ciudad de méxico": "MX",
    "guadalajara": "MX", "monterrey": "MX",
    "merida, yucatan": "MX", "merida": "MX",
    "argentina": "AR", "buenos aires": "AR", "córdoba": "AR",
    "colombia": "CO", "bogota": "CO", "bogotá": "CO", "medellín": "CO",
    "chile": "CL", "santiago": "CL",
    "malaysia": "MY", "kuala lumpur": "MY",
    "thailand": "TH", "bangkok": "TH",
    "indonesia": "ID", "jakarta": "ID", "pekanbaru": "ID",
    "philippines": "PH", "manila": "PH",
    "vietnam": "VN", "viet nam": "VN", "ho chi minh city": "VN",
    "hanoi": "VN", "hcmc": "VN",
    "sri lanka": "LK", "colombo": "LK",
    "nepal": "NP", "kathmandu": "NP",
    "iraq": "IQ", "baghdad": "IQ", "iraq - sulaimaiyah": "IQ",
    "ecuador": "EC", "quito": "EC", "quito, ecuador.": "EC",
    "hong kong": "HK",
    "iran": "IR", "tehran": "IR",
    "uzbekistan": "UZ", "tashkent": "UZ", "tashkent, uzbekistan": "UZ",
    "serbia": "RS", "belgrade": "RS",
    "florida": "US", "gainesville, florida": "US", "gainesville": "US",
    "iowa": "US", "des moines": "US", "des moines,, ia": "US",
}

PANEL_COUNTRIES = {
    # 38 kept countries (trimmed from 54 on 2026-04-22)
    # Dropped 16: LK, SA, GR, IE, MY, RO, TW, ZA, UA, NZ, CO, BE, TH, HU, NP, AR
    # Rationale: <=4 accounts each + geographic redundancy. See country_trim_analysis.md
    "US", "GB", "DE", "FR", "IN", "CN", "BR", "CA", "AU", "RU",
    "JP", "KR", "NL", "SE", "CH", "PL", "NG", "BD", "KE", "TR",
    "ES", "IT", "NO", "DK", "FI", "PT", "AT", "CZ",
    "PK", "EG", "MX", "CL",
    "SG", "ID", "PH", "VN", "IL", "AE",
}


def parse_location(location_str):
    if not location_str:
        return None
    key = location_str.strip().lower()
    key = re.sub(r"\s*\(.*?\)", "", key).strip()
    if key in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[key]
    parts = [p.strip() for p in key.split(",")]
    for part in reversed(parts):
        if part in COUNTRY_NAME_MAP:
            return COUNTRY_NAME_MAP[part]
    if parts[0] in COUNTRY_NAME_MAP:
        return COUNTRY_NAME_MAP[parts[0]]
    return None


# ---------------------------------------------------------------------------
# HTTP helpers — v3: capped, jittered, proactive, abuse-aware
# ---------------------------------------------------------------------------

def _rest_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "population-scraper-v3/1.0",
    }


def _graphql_headers():
    return {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Content-Type": "application/json",
        "User-Agent": "population-scraper-v3/1.0",
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


def _global_rate_limit_pause():
    """Pause the whole script when we've hit too many 403s in a row."""
    global _consecutive_rate_limits
    print(f"  GLOBAL PAUSE: {_consecutive_rate_limits} consecutive rate-limit hits. "
          f"Cooling off for {GLOBAL_RATE_LIMIT_PAUSE}s...")
    time.sleep(_jitter(GLOBAL_RATE_LIMIT_PAUSE))
    _refresh_rate_limit_state()
    _consecutive_rate_limits = 0
    print("  Global pause complete. Resuming.")


def gh_get(url):
    """
    REST GET with capped, jittered, proactive rate-limit handling.
    Returns parsed JSON or None.
    """
    global _consecutive_rate_limits
    _check_global_rate_limit()
    headers = _rest_headers()
    network_attempts = 0
    rate_limit_attempts = 0

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                _update_rate_limit(resp.headers)
                _consecutive_rate_limits = 0
                return json.loads(resp.read().decode())

        except urllib.error.HTTPError as e:
            _update_rate_limit(e.headers)
            body = e.read().decode(errors="replace")

            if e.code == 403 and "secondary" in body.lower():
                wait = min(SECONDARY_RATE_LIMIT_FLOOR * (attempt + 1), MAX_SLEEP)
                print(f"    Secondary rate limit, sleeping {wait:.0f}s...")
                time.sleep(_jitter(wait))
                _consecutive_rate_limits += 1

            elif e.code == 403:
                remaining = _rate_limit_state.get("remaining")
                if remaining is not None and remaining > 0:
                    # Abuse detection or other non-quota 403
                    wait = min(ABUSE_BACKOFF_BASE * (attempt + 1), MAX_SLEEP)
                    print(f"    403 with {remaining} quota remaining (abuse?), sleeping {wait:.0f}s...")
                    time.sleep(_jitter(wait))
                    _consecutive_rate_limits += 1
                else:
                    # Primary rate limit — use reset timestamp, capped
                    reset_ts = _rate_limit_state.get("reset_at")
                    now = time.time()
                    if reset_ts and reset_ts > now:
                        wait = min(reset_ts - now + 5, MAX_SLEEP)
                    else:
                        wait = min(60, MAX_SLEEP)
                    print(f"    Rate limit (403), sleeping {wait:.0f}s...")
                    time.sleep(_jitter(wait))
                    _consecutive_rate_limits += 1
                rate_limit_attempts += 1
                if rate_limit_attempts >= RATE_LIMIT_MAX_ATTEMPTS:
                    print(f"    Rate-limit retries exhausted. Returning None.")
                    return None
                if _consecutive_rate_limits >= CONSECUTIVE_RATE_LIMIT_PAUSE:
                    _global_rate_limit_pause()

            elif e.code in (404, 409, 451):
                _consecutive_rate_limits = 0
                return None
            elif e.code >= 500:
                wait = min(API_DELAY * (2 ** attempt), MAX_SLEEP)
                time.sleep(_jitter(wait))
            else:
                _consecutive_rate_limits = 0
                return None

        except Exception as exc:
            if _is_network_error(exc):
                network_attempts += 1
                wait = min(NETWORK_RETRY_FLOOR * network_attempts, MAX_SLEEP)
                print(f"    Network error ({exc}), sleeping {wait:.0f}s...")
                if network_attempts >= NETWORK_MAX_RETRIES:
                    raise NetworkError(f"Network failed after {NETWORK_MAX_RETRIES} attempts") from exc
                time.sleep(_jitter(wait))
            else:
                _consecutive_rate_limits = 0
                return None

    return None


def graphql_post(query):
    """
    GraphQL POST with the same capped, jittered rate-limit discipline.
    """
    global _consecutive_rate_limits
    _check_global_rate_limit()
    payload = json.dumps({"query": query}).encode()
    headers = _graphql_headers()
    network_attempts = 0
    rate_limit_attempts = 0

    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(
                "https://api.github.com/graphql",
                data=payload,
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                _update_rate_limit(resp.headers)
                _consecutive_rate_limits = 0
                return json.loads(resp.read().decode())

        except urllib.error.HTTPError as e:
            _update_rate_limit(e.headers)
            body = e.read().decode(errors="replace")

            if e.code in (403, 429):
                remaining = _rate_limit_state.get("remaining")
                if remaining is not None and remaining > 0:
                    wait = min(ABUSE_BACKOFF_BASE * (attempt + 1), MAX_SLEEP)
                    print(f"    GraphQL abuse? ({remaining} remaining), sleeping {wait:.0f}s...")
                else:
                    wait = min(SECONDARY_RATE_LIMIT_FLOOR * (attempt + 1), MAX_SLEEP)
                    print(f"    GraphQL rate limit ({e.code}), sleeping {wait:.0f}s...")
                time.sleep(_jitter(wait))
                _consecutive_rate_limits += 1
                rate_limit_attempts += 1
                if rate_limit_attempts >= RATE_LIMIT_MAX_ATTEMPTS:
                    print(f"    GraphQL rate-limit retries exhausted. Returning None.")
                    return None
                if _consecutive_rate_limits >= CONSECUTIVE_RATE_LIMIT_PAUSE:
                    _global_rate_limit_pause()

            elif e.code >= 500:
                wait = min(GRAPHQL_BATCH_DELAY * (2 ** attempt), MAX_SLEEP)
                time.sleep(_jitter(wait))
            else:
                _consecutive_rate_limits = 0
                return None

        except Exception as exc:
            if _is_network_error(exc):
                network_attempts += 1
                wait = min(NETWORK_RETRY_FLOOR * network_attempts, MAX_SLEEP)
                print(f"    GraphQL network error ({exc}), sleeping {wait:.0f}s...")
                if network_attempts >= NETWORK_MAX_RETRIES:
                    raise NetworkError(f"GraphQL network failed") from exc
                time.sleep(_jitter(wait))
            else:
                _consecutive_rate_limits = 0
                return None

    return None


def _sleep():
    time.sleep(_jitter(API_DELAY))


# ---------------------------------------------------------------------------
# Stage 1: GH Archive candidate collection
# ---------------------------------------------------------------------------

def _gh_archive_path(date_str, hour):
    return DATA_DIR / "gh_archive_cache" / f"{date_str}-{hour}.jsonl"


def _download_gh_archive(date_str, hour):
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
    print("\n=== STAGE 1: Collecting candidates from GH Archive ===")
    all_logins = set()
    for date_str, hour in GH_ARCHIVE_HOURS:
        cached_path = DATA_DIR / "gh_archive_cache" / f"{date_str}-{hour}.jsonl"
        alt_path = DATA_DIR / f"gharchive_{date_str}-{hour}.jsonl"

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
# Stage 1.5: GraphQL batch location pre-filter
# ---------------------------------------------------------------------------

def load_location_cache():
    cache = {}
    if not LOCATION_CACHE_PATH.exists():
        return cache
    with open(LOCATION_CACHE_PATH, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 2:
                cache[row[0]] = row[1] if row[1] else None
    return cache


def _save_location_batch(results):
    with open(LOCATION_CACHE_PATH, "a", newline="") as f:
        w = csv.writer(f)
        for login, country in results.items():
            w.writerow([login, country or ""])


def _build_location_query(logins):
    aliases = "\n    ".join(
        f'u{i}: user(login: {json.dumps(login)}) {{ login location }}'
        for i, login in enumerate(logins)
    )
    return f"{{\n    {aliases}\n}}"


def batch_fetch_locations(logins):
    if not logins:
        return {}
    query = _build_location_query(logins)
    response = graphql_post(query)
    if response is None:
        return {}
    data = response.get("data") or {}
    out = {}
    for i, login in enumerate(logins):
        node = data.get(f"u{i}")
        if node:
            out[login] = node.get("location")
        else:
            out[login] = None
    return out


def prefilter_candidates(candidates, done_logins):
    print("\n=== STAGE 1.5: GraphQL batch location pre-filter ===")
    location_cache = load_location_cache()
    all_done = done_logins

    uncached = [l for l in candidates if l not in location_cache and l not in all_done]
    print(f"  Total candidates: {len(candidates)}")
    print(f"  Already location-cached: {len(location_cache)}")
    print(f"  To fetch via GraphQL: {len(uncached)}")
    print(f"  Batch size: {LOCATION_BATCH_SIZE}, delay: {GRAPHQL_BATCH_DELAY}s/batch")

    batches_done = 0
    for i in range(0, len(uncached), LOCATION_BATCH_SIZE):
        batch = uncached[i : i + LOCATION_BATCH_SIZE]
        raw_locations = batch_fetch_locations(batch)
        country_results = {
            login: parse_location(loc)
            for login, loc in raw_locations.items()
        }
        _save_location_batch(country_results)
        location_cache.update(country_results)
        batches_done += 1

        if batches_done % 100 == 0:
            panel_found = sum(1 for c in location_cache.values() if c in PANEL_COUNTRIES)
            pct_done = 100 * (i + len(batch)) / max(len(uncached), 1)
            print(f"  [{pct_done:.0f}%] {i + len(batch)}/{len(uncached)} fetched, "
                  f"{panel_found} panel-country accounts found so far")

        time.sleep(_jitter(GRAPHQL_BATCH_DELAY))

    panel_candidates = [
        (login, location_cache[login])
        for login in candidates
        if login in location_cache
        and location_cache[login] in PANEL_COUNTRIES
        and login not in done_logins
    ]

    panel_count = sum(1 for c in location_cache.values() if c in PANEL_COUNTRIES)
    print(f"\n  Location cache: {len(location_cache)} entries, "
          f"{panel_count} in panel countries ({100*panel_count/max(len(location_cache),1):.1f}%)")
    print(f"  Panel-country candidates ready for scrape: {len(panel_candidates)}")
    return panel_candidates


# ---------------------------------------------------------------------------
# Resume state
# ---------------------------------------------------------------------------

def load_status():
    status = {}
    if not STATUS_PATH.exists():
        return status
    with open(STATUS_PATH, newline="") as f:
        for row in csv.DictReader(f):
            status[row["login"]] = row
    return status


def append_status(row):
    fieldnames = ["login", "status", "country", "classifier_score",
                  "pre_commits", "post_commits", "timestamp"]
    write_header = not STATUS_PATH.exists()
    with open(STATUS_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def append_features(row):
    write_header = not FEATURES_PATH.exists()
    with open(FEATURES_PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


# ---------------------------------------------------------------------------
# Light scrape
# ---------------------------------------------------------------------------

def scrape_account_light(login, known_country=None):
    cache_path = POP_CACHE_DIR / f"{login}.json"
    if cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        if not cached.get("error"):
            return cached

    if known_country is None:
        profile = gh_get(f"https://api.github.com/users/{login}")
        _sleep()
        if not profile:
            return {"error": "profile fetch failed", "commits": [], "prs": [], "location": None}
        location = profile.get("location")
    else:
        location = known_country

    repos_data = gh_get(
        f"https://api.github.com/users/{login}/repos"
        f"?type=owner&sort=created&direction=asc&per_page={MAX_REPOS_PER_ACCOUNT}"
    )
    _sleep()
    if not repos_data or not isinstance(repos_data, list):
        return {"error": "repos fetch failed", "commits": [], "prs": [], "location": location}

    repos = [r["name"] for r in repos_data if not r.get("fork", False)][:MAX_REPOS_PER_ACCOUNT]

    all_commits = []
    all_prs = []
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
                    "sha": item.get("sha", ""),
                    "message": c.get("message", ""),
                    "created_at": (c.get("committer") or c.get("author") or {}).get("date", ""),
                    "repo": repo_name,
                    "file_sampled": False,
                    "has_test_file": None,
                    "has_impl_file": None,
                })
            commits_remaining -= len(data)

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
                    "created_at": pr.get("created_at", ""),
                    "body_length": len(body),
                })

    result = {
        "login": login,
        "location": location,
        "commits": all_commits,
        "prs": all_prs,
    }
    with open(cache_path, "w") as f:
        json.dump(result, f)
    return result


# ---------------------------------------------------------------------------
# Feature extraction
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

    multiline_n = sum(1 for m in cleaned if "\n" in m)
    conventional_n = sum(1 for m in cleaned if conv_re.match(m))
    test_n = sum(1 for m in cleaned if test_re.search(m))
    bullets_n = sum(1 for m in cleaned if "- " in m or "* " in m)

    sorted_w = sorted(window_c, key=lambda c: _parse_dt(c.get("created_at")) or datetime.min)
    inter = []
    for i in range(1, len(sorted_w)):
        dt1 = _parse_dt(sorted_w[i-1].get("created_at"))
        dt2 = _parse_dt(sorted_w[i].get("created_at"))
        if dt1 and dt2:
            inter.append((dt2 - dt1).total_seconds() / 3600.0)

    mean_inter = sum(inter) / len(inter) if inter else 0.0
    frac_burst = sum(1 for h in inter if h <= 2.0) / len(inter) if inter else 0.0

    window_p = [
        pr for pr in prs
        if _parse_dt(pr.get("created_at")) is not None
        and _parse_dt(pr.get("created_at")) >= after
        and (not before or _parse_dt(pr.get("created_at")) < before)
    ]
    if window_p:
        bl = [pr.get("body_length", 0) for pr in window_p]
        mean_body = sum(bl) / len(bl)
        frac_has_body = sum(1 for b in bl if b > 50) / len(bl)
    else:
        mean_body = frac_has_body = 0.0

    n = len(window_c)
    return {
        "commit_count": n,
        "mean_message_length": round(sum(msg_lengths) / n, 2),
        "active_weeks": active_weeks,
        "repos_touched": repos,
        "mean_commits_per_active_week": round(n / max(active_weeks, 1), 2),
        "frac_multiline": round(multiline_n / n, 3),
        "frac_conventional": round(conventional_n / n, 3),
        "frac_mentions_test": round(test_n / n, 3),
        "frac_has_bullets": round(bullets_n / n, 3),
        "mean_inter_commit_hours": round(mean_inter, 2),
        "frac_burst_commits": round(frac_burst, 3),
        "sampled_test_cowrite_rate": 0.0,
        "file_sample_count": 0,
        "mean_pr_body_length": round(mean_body, 2),
        "frac_pr_has_body": round(frac_has_body, 3),
    }


def extract_features_for_account(login, data):
    commits = _deduplicate(data.get("commits", []))
    prs = data.get("prs", [])

    pre_count = _count_in_window(commits, PRE_START, PRE_CUTOFF)
    post_count = _count_in_window(commits, POST_START)

    if pre_count < MIN_PRE_COMMITS or post_count < MIN_POST_COMMITS:
        return None, pre_count, post_count

    pre_f = _window_features(commits, prs, PRE_START, PRE_CUTOFF)
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
    import numpy as np
    vals = [row.get(fc, 0.0) for fc in feature_cols]
    X = imputer.transform([vals])
    return float(model.predict_proba(X)[0, 1])


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def build_adoption_table(scores_path):
    import pandas as pd

    df = pd.read_csv(scores_path)
    print(f"\nScored accounts: {len(df)}")
    print(f"Countries: {df['country'].nunique()}")

    records = []
    for _, row in df.iterrows():
        country = row["country"]
        if country not in PANEL_COUNTRIES:
            continue
        records.append({"country": country, "year": 2022, "ai_prob": row["pre_classifier_score"]})
        records.append({"country": country, "year": 2023, "ai_prob": row["pre_classifier_score"]})
        records.append({"country": country, "year": 2024, "ai_prob": row["post_classifier_score"]})

    agg = pd.DataFrame(records)
    summary = (
        agg.groupby(["country", "year"])
           .agg(pct_ai_users=("ai_prob", "mean"),
                n_accounts=("ai_prob", "count"))
           .reset_index()
    )

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
    print("POPULATION SCRAPE v3")
    print(f"Target: {MAX_TOTAL_ACCOUNTS} accounts across {len(PANEL_COUNTRIES)} panel countries")
    print(f"API delay: {API_DELAY}s (REST), {GRAPHQL_BATCH_DELAY}s (GraphQL batch)")
    print(f"Max sleep per attempt: {MAX_SLEEP}s")
    print(f"GraphQL batch size: {LOCATION_BATCH_SIZE} users/request")
    print("=" * 65)

    print("\nLoading classifier model...")
    model, imputer, feature_cols = load_model()
    print(f"  Model loaded. Feature cols: {len(feature_cols)}")

    status = load_status()
    done_logins = {l for l, s in status.items() if s["status"] in ("scored", "skipped", "error")}
    scored_logins = {l for l, s in status.items() if s["status"] == "scored"}
    country_counts = defaultdict(int)
    for l, s in status.items():
        if s["status"] == "scored" and s.get("country") in PANEL_COUNTRIES:
            country_counts[s["country"]] += 1

    print(f"\nResume state: {len(scored_logins)} scored, {len(done_logins)} total processed")
    total_scored = len(scored_logins)

    if total_scored >= MAX_TOTAL_ACCOUNTS:
        print(f"\nTarget reached ({total_scored} accounts). Building adoption table.")
        build_adoption_table(SCORES_PATH)
        return

    candidates = collect_candidates()
    random.shuffle(candidates)

    panel_candidates = prefilter_candidates(candidates, done_logins)
    print(f"\nFresh panel-country candidates to scrape: {len(panel_candidates)}")

    scores_write_header = not SCORES_PATH.exists()
    scores_fieldnames = ["login", "country", "pre_classifier_score",
                         "post_classifier_score", "pre_commits", "post_commits"]

    consecutive_failures = 0

    for login, country in panel_candidates:
        if total_scored >= MAX_TOTAL_ACCOUNTS:
            print(f"\nTarget reached ({MAX_TOTAL_ACCOUNTS}). Stopping.")
            break

        try:
            data = scrape_account_light(login, known_country=country)
        except NetworkError:
            consecutive_failures += 1
            print(f"  {login}: NetworkError (skip)")
            append_status({"login": login, "status": "network_error", "country": country,
                           "classifier_score": "", "pre_commits": "", "post_commits": "",
                           "timestamp": datetime.now().isoformat()})
            if consecutive_failures >= CONSECUTIVE_NETWORK_FAIL_LIMIT:
                print(f"  Circuit breaker — pausing {CIRCUIT_BREAKER_PAUSE}s")
                time.sleep(_jitter(CIRCUIT_BREAKER_PAUSE))
                consecutive_failures = 0
            continue

        consecutive_failures = 0

        if data.get("error"):
            append_status({"login": login, "status": "error", "country": country,
                           "classifier_score": "", "pre_commits": "", "post_commits": "",
                           "timestamp": datetime.now().isoformat()})
            continue

        feat_row, pre_count, post_count = extract_features_for_account(login, data)
        if feat_row is None:
            append_status({"login": login, "status": "skipped", "country": country,
                           "classifier_score": "", "pre_commits": pre_count,
                           "post_commits": post_count,
                           "timestamp": datetime.now().isoformat()})
            continue

        pre_feat_row = {k: v for k, v in feat_row.items()}
        for k in list(pre_feat_row.keys()):
            if k.startswith("post_"):
                base = k[5:]
                pre_feat_row[k] = pre_feat_row.get(f"pre_{base}", 0.0)
        for k in list(pre_feat_row.keys()):
            if k.startswith("delta_"):
                pre_feat_row[k] = 0.0

        pre_score = score_row(model, imputer, feature_cols, pre_feat_row)
        post_score = score_row(model, imputer, feature_cols, feat_row)

        score_row_dict = {
            "login": login,
            "country": country,
            "pre_classifier_score": round(pre_score, 4),
            "post_classifier_score": round(post_score, 4),
            "pre_commits": pre_count,
            "post_commits": post_count,
        }
        with open(SCORES_PATH, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scores_fieldnames)
            if scores_write_header:
                w.writeheader()
                scores_write_header = False
            w.writerow(score_row_dict)

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

        if total_scored % 100 == 0:
            covered = sum(1 for c in PANEL_COUNTRIES if country_counts[c] >= TARGET_PER_COUNTRY)
            print(f"\n  === Progress: {total_scored} scored, "
                  f"{covered}/{len(PANEL_COUNTRIES)} countries at target ===\n")

    print("\n=== Building country-quarter adoption table ===")
    if SCORES_PATH.exists():
        adoption = build_adoption_table(SCORES_PATH)
        import pandas as pd
        print("\nTop countries by post-2024 AI adoption rate:")
        post = adoption[adoption["year"] == 2024].sort_values("pct_ai_users", ascending=False)
        print(post.head(20).to_string(index=False))
    else:
        print("No scores file found — nothing to aggregate.")

    print("\n" + "=" * 65)
    print("POPULATION SCRAPE v3 COMPLETE")
    print(f"  Scored: {total_scored} accounts")
    print(f"  Status: {STATUS_PATH}")
    print(f"  Scores: {SCORES_PATH}")
    print(f"  Adoption table: {ADOPTION_PATH}")
    print("=" * 65)


if __name__ == "__main__":
    main()
