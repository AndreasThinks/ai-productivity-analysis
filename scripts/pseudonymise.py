#!/usr/bin/env python
"""
Pseudonymise GitHub logins in everything committed under data/.

Logins are replaced by salted SHA-256 ids ("acct_" + 12 hex characters) and
the column is renamed from ``login`` to ``account_id``. The same salt always
gives the same id, so files still join to each other. The salt and the
login -> id key live in the private directory (scripts/paths.py) and are never
committed; without them the ids cannot be reversed by hashing guessed logins.

Before pseudonymising, the script writes private/accounts_v4.csv, the account
list scripts/rescrape_v4.py needs (it must have real logins).

  uv run scripts/pseudonymise.py            # rewrite data/ in place
  uv run scripts/pseudonymise.py --dry-run  # list what would change
"""

from __future__ import annotations

import argparse
import hashlib
import os
import secrets
import shutil
from pathlib import Path

import pandas as pd

try:
    from scripts.paths import DATA_DIR, PRIVATE_DIR
except ImportError:
    from paths import DATA_DIR, PRIVATE_DIR

SALT_PATH = PRIVATE_DIR / "pseudonym_salt.txt"
KEY_PATH = PRIVATE_DIR / "login_key.csv"

# Raw per-account material that cannot be pseudonymised in place (commit
# messages, profile locations, login lists inside JSON). Moved out of data/.
MOVE_TO_PRIVATE = [
    "population_cache_v2",
    "aider_cache",
    "full_positive_progress.json",
    "expansion_progress.json",
    "test_positive_progress.json",
    "population_scrape_status.csv.bak",
    "population_scrape_status.csv.bak2",
]


def load_salt() -> str:
    """Salt from AI_PRODUCTIVITY_SALT, else private/pseudonym_salt.txt (created if absent)."""
    env = os.environ.get("AI_PRODUCTIVITY_SALT")
    if env:
        return env
    if not SALT_PATH.exists():
        SALT_PATH.parent.mkdir(parents=True, exist_ok=True)
        SALT_PATH.write_text(secrets.token_hex(32))
    return SALT_PATH.read_text().strip()


def pseudonym(login: str, salt: str) -> str:
    """Stable, non-reversible id for a GitHub login (case-insensitive, as GitHub is)."""
    digest = hashlib.sha256(f"{salt}:{str(login).lower()}".encode()).hexdigest()
    return f"acct_{digest[:12]}"


def build_account_list(data_dir: Path) -> pd.DataFrame:
    """Every account the v4 re-scrape covers, with its group and tool."""
    feats = pd.read_csv(data_dir / "classifier_full_features.csv")
    expansion = pd.read_csv(data_dir / "expansion_features.csv")
    aider = pd.read_csv(data_dir / "aider_validation_results.csv")
    rows = (
        [{"login": l, "group": "adopter", "tool": "claude", "source": "v2.7 main"} for l in feats.loc[feats.label == 1, "login"]]
        + [{"login": l, "group": "adopter", "tool": "claude", "source": "v2.7 expansion"} for l in expansion["login"]]
        + [{"login": l, "group": "adopter", "tool": "aider", "source": "aider validation"} for l in aider["login"]]
        + [{"login": l, "group": "control", "tool": "", "source": "v2.7 controls"} for l in feats.loc[feats.label == 0, "login"]]
    )
    return pd.DataFrame(rows).drop_duplicates("login")


def pseudonymise_csvs(data_dir: Path, salt: str, dry_run: bool = False) -> dict[str, str]:
    key: dict[str, str] = {}
    for path in sorted(data_dir.glob("*.csv")):
        df = pd.read_csv(path)
        if "repo" in df.columns:
            # "owner/name" carries the login too
            print(f"  {path.name}: hashing repo names")
            if not dry_run:
                df["repo"] = df["repo"].map(lambda r: "repo_" + pseudonym(r, salt)[5:] if pd.notna(r) and not str(r).startswith("repo_") else r)
                df.to_csv(path, index=False)
        if "login" not in df.columns:
            continue
        ids = df["login"].map(lambda l: pseudonym(l, salt) if pd.notna(l) else l)
        key.update({l: i for l, i in zip(df["login"].dropna(), ids.dropna())})
        print(f"  {path.name}: {len(df)} rows")
        if not dry_run:
            df.insert(list(df.columns).index("login"), "account_id", ids)
            df.drop(columns="login").to_csv(path, index=False)
    return key


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    salt = load_salt()
    accounts_path = PRIVATE_DIR / "accounts_v4.csv"
    if not accounts_path.exists() and (DATA_DIR / "classifier_full_features.csv").exists():
        if "login" in pd.read_csv(DATA_DIR / "classifier_full_features.csv", nrows=1).columns:
            build_account_list(DATA_DIR).to_csv(accounts_path, index=False)
            print(f"wrote {accounts_path}")

    key = pseudonymise_csvs(DATA_DIR, salt, dry_run=args.dry_run)
    for name in MOVE_TO_PRIVATE:
        src = DATA_DIR / name
        if src.exists():
            print(f"  move {name} -> private/")
            if not args.dry_run:
                shutil.move(str(src), str(PRIVATE_DIR / name))
    if key and not args.dry_run:
        old = pd.read_csv(KEY_PATH) if KEY_PATH.exists() else pd.DataFrame(columns=["login", "account_id"])
        new = pd.DataFrame({"login": list(key), "account_id": list(key.values())})
        pd.concat([old, new]).drop_duplicates("login").to_csv(KEY_PATH, index=False)
        print(f"wrote {KEY_PATH} ({len(key)} logins)")


if __name__ == "__main__":
    main()
