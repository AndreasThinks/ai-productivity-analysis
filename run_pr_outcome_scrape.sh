#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/avery/projects/ai_productivity_analysis
LOCK=/tmp/ai-productivity-pr-outcome-scrape.lock
LOG=/tmp/pr_outcome_scrape.log

cd "$PROJECT"
mkdir -p data/pr_outcome_cache

exec 9>"$LOCK"
if ! flock -n 9; then
  echo "$(date -Is) another PR outcome scrape is already running" | tee -a "$LOG"
  exit 1
fi

export PYTHONUNBUFFERED=1

if [ -z "${GITHUB_TOKEN:-}" ]; then
  if [ -f "$HOME/.git-credentials" ]; then
    GITHUB_TOKEN=$(uv run python - <<'PY'
from pathlib import Path
from urllib.parse import urlparse

path = Path.home() / ".git-credentials"
for line in path.read_text().splitlines():
    if "github.com" in line:
        parsed = urlparse(line.strip())
        if parsed.password:
            print(parsed.password)
            break
PY
)
    export GITHUB_TOKEN
  fi
fi

if [ -z "${GITHUB_TOKEN:-}" ]; then
  echo "$(date -Is) GITHUB_TOKEN not available" | tee -a "$LOG"
  exit 1
fi

echo "$(date -Is) starting PR outcome scrape" | tee -a "$LOG"
uv run --with pandas --with statsmodels \
  scripts/pr_outcome_metrics.py --scrape --analyse \
  >> "$LOG" 2>&1
echo "$(date -Is) PR outcome scrape finished" | tee -a "$LOG"
