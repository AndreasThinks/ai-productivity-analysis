#!/bin/bash
# Launcher for scrape_expanded_positives.py
# Sources .env, activates uv, writes logs to data/expansion_run.log

set -a
[ -f "${ENV_FILE:-.env}" ] && source "${ENV_FILE:-.env}"
set +a

export PYTHONUNBUFFERED=1

cd "$(dirname "${BASH_SOURCE[0]}")/.."

LOG=data/expansion_run.log

echo "=== Expansion scraper started: $(date) ===" >> "$LOG"
uv run python -u scripts/scrape_expanded_positives.py >> "$LOG" 2>&1
echo "=== Expansion scraper finished: $(date) ===" >> "$LOG"
