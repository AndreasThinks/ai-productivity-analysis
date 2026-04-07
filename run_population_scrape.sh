#!/bin/bash
# Launch population scrape with GitHub token from git credential store
# Resume-safe: status CSV tracks progress, restart picks up where it left off

set -euo pipefail

LOCKFILE="/tmp/population_scrape.lock"

# Prevent duplicate launches
if [ -f "$LOCKFILE" ]; then
    LOCKED_PID=$(cat "$LOCKFILE")
    if kill -0 "$LOCKED_PID" 2>/dev/null; then
        echo "ERROR: scrape already running (PID $LOCKED_PID). Exiting." >&2
        exit 1
    else
        echo "Stale lockfile found, removing."
        rm -f "$LOCKFILE"
    fi
fi

cd /home/andreasclaw/projects/ai_productivity_analysis

# Pull token from git credential store (works with clean HTTPS remotes)
GITHUB_TOKEN=$(printf 'protocol=https\nhost=github.com\n' | git credential fill | grep '^password=' | cut -d= -f2-)

if [ -z "$GITHUB_TOKEN" ] || [ ${#GITHUB_TOKEN} -lt 20 ]; then
    echo "ERROR: Could not retrieve a valid GITHUB_TOKEN from git credentials" >&2
    exit 1
fi

echo "Token length: ${#GITHUB_TOKEN} — looks valid"

export GITHUB_TOKEN

# Write lockfile, clean up on exit
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

exec /home/andreasclaw/.local/bin/uv run --with scikit-learn --with joblib \
    python3 -u scripts/scrape_population.py
