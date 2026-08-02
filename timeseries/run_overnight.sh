#!/usr/bin/env bash
# run_overnight.sh — one command to launch the full overnight trial + analysis.
#
#   cd /Users/rubygrambauer/RA_Raji/AMW-mining-detector
#   bash timeseries/run_overnight.sh
#
# Detaches with nohup AND starts its own `caffeinate -w` watcher, so the Mac
# won't idle-sleep while the job runs. Keep it plugged in with the lid OPEN
# (lid close still sleeps regardless of caffeinate).
# Progress: tail -f data/timeseries/nohup_overnight.log
#
# Resume after any crash: run the same command again — finished periods are
# skipped, cached tiles are reused.
set -euo pipefail
cd "$(dirname "$0")/.."          # repo root

# --- environment -------------------------------------------------------------
source venv/bin/activate
# Pull EE_PROJECT out of .env without sourcing it (the .env here is not valid
# shell: it has spaces around '='). Tolerates KEY=v, KEY = v, and quotes.
if [ -z "${EE_PROJECT:-}" ] && [ -f .env ]; then
  EE_PROJECT=$(sed -n 's/^EE_PROJECT[[:space:]]*=[[:space:]]*//p' .env \
               | head -1 | tr -d '"'"'"' \t\r')
  export EE_PROJECT
fi
: "${EE_PROJECT:?EE_PROJECT not set — put it in .env or export it}"

LOG=data/timeseries/nohup_overnight.log
mkdir -p data/timeseries

echo "=== overnight run started $(date) (EE_PROJECT=$EE_PROJECT) ===" | tee -a "$LOG"

# --- 1. the 16 period panels (12 monthly + 4 quarterly, calendar 2025) -------
nohup bash -c '
  python timeseries/run_trials.py --year 2025 \
    && python timeseries/analysis.py \
    && echo "=== ALL DONE $(date) ==="
' >> "$LOG" 2>&1 &
JOB_PID=$!

# keep the Mac awake (idle-sleep) exactly as long as the job runs; -i = no idle
# sleep, -m = no disk sleep. Dies automatically when the job finishes.
if command -v caffeinate >/dev/null; then
  nohup caffeinate -im -w "$JOB_PID" >/dev/null 2>&1 &
fi

echo "PID $JOB_PID — follow with: tail -f $LOG"
echo "Keep the laptop plugged in, lid open (caffeinate is watching the job)."
