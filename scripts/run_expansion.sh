#!/bin/bash
# Launcher for scrape_expanded_positives.py
# Sources .env, activates uv, writes logs to data/expansion_run.log

set -a
source /home/avery/.hermes/.env
set +a

export PYTHONUNBUFFERED=1

cd /home/avery/projects/ai_productivity_analysis

LOG=/home/avery/projects/ai_productivity_analysis/data/expansion_run.log

echo "=== Expansion scraper started: $(date) ===" >> "$LOG"
uv run python -u scripts/scrape_expanded_positives.py >> "$LOG" 2>&1
echo "=== Expansion scraper finished: $(date) ===" >> "$LOG"
