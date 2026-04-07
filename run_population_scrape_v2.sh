#!/bin/bash
set -euo pipefail

cd /home/andreasclaw/projects/ai_productivity_analysis

GITHUB_TOKEN=$(git remote get-url origin | sed 's|.*AndreasThinks:\(.*\)@github.*|\1|')

if [ -z "$GITHUB_TOKEN" ]; then
    echo "ERROR: Could not extract GITHUB_TOKEN from git remote" >&2
    exit 1
fi

echo "Token extracted, length: ${#GITHUB_TOKEN}"

exec env GITHUB_TOKEN="$GITHUB_TOKEN" uv run python scripts/scrape_population_v2.py
