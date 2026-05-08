#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/avery/projects/ai_productivity_analysis
UNIT=ai-productivity-pr-outcome-scrape.service
LOG=/tmp/pr_outcome_scrape.log
CACHE_DIR="$PROJECT/data/pr_outcome_cache"
STATUS="$PROJECT/data/pr_outcome_status.csv"
FEATURES="$PROJECT/data/classifier_full_features.csv"

cd "$PROJECT"

cached=0
if [ -d "$CACHE_DIR" ]; then
  cached=$(find "$CACHE_DIR" -name '*.json' | wc -l)
fi

total=0
if [ -f "$FEATURES" ]; then
  total=$(($(wc -l < "$FEATURES") - 1))
fi

active=$(systemctl --user is-active ai-productivity-pr-outcome-scrape 2>/dev/null || true)
substate=$(systemctl --user show ai-productivity-pr-outcome-scrape -p SubState --value 2>/dev/null || true)
last_status="none"
if [ -f "$STATUS" ]; then
  last_status=$(tail -1 "$STATUS")
fi

last_cache="none"
if [ -d "$CACHE_DIR" ]; then
  last_cache=$(find "$CACHE_DIR" -name '*.json' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 || true)
  [ -n "$last_cache" ] || last_cache="none"
fi

latest_epoch=0
if [ -d "$CACHE_DIR" ]; then
  latest_epoch=$(find "$CACHE_DIR" -name '*.json' -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1 || echo 0)
fi
if [ -f "$STATUS" ]; then
  status_epoch=$(stat -c %Y "$STATUS")
  if [ "$status_epoch" -gt "$latest_epoch" ]; then
    latest_epoch="$status_epoch"
  fi
fi

now=$(date +%s)
age=$(( now - latest_epoch ))

echo "unit_active=$active"
echo "unit_substate=$substate"
echo "cached=$cached/$total"
echo "last_status=$last_status"
echo "last_cache=$last_cache"
echo "latest_write_age_seconds=$age"
echo "log=$LOG"

if [ "$active" = "failed" ]; then
  echo "ALERT: unit failed"
  exit 2
fi

if [ "$active" = "active" ] && [ "$total" -gt 0 ] && [ "$cached" -lt "$total" ] && [ "$latest_epoch" -gt 0 ] && [ "$age" -gt 7200 ]; then
  echo "ALERT: scrape appears stalled, no cache/status write for >2h"
  exit 3
fi

if [ -f "$LOG" ]; then
  echo "--- log tail ---"
  tail -20 "$LOG"
fi
