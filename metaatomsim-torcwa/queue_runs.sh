#!/usr/bin/env bash
# Run notebooks back to back: each waits for the previous runner process to exit.
# usage: queue_runs.sh <wait-for-pattern> <notebook> <log> [<notebook> <log> ...]
cd "$(dirname "$0")"
PY=~/anaconda3/envs/Metasurface-Design/bin/python
wait_for=$1; shift
while pgrep -f "run_notebook.py $wait_for" >/dev/null; do sleep 30; done
while [ $# -ge 2 ]; do
  nb=$1; log=$2; shift 2
  mkdir -p "$(dirname "$log")"
  echo "[$(date +%T)] queue: starting $nb" >> "$log"
  $PY run_notebook.py "$nb" --log "$log" >> "${log%.log}.stderr" 2>&1
done
