#!/usr/bin/env bash
# Launch optimize_prompt.py detached in the background with a unique --run-dir,
# so the run can be monitored (tail -f run_log.txt) and stopped gracefully
# (touch gepa.stop) even after this terminal is closed.
#
# Usage: forward any optimize_prompt.py args as-is, e.g.:
#   ./run_optimize.sh --domain corporate-ma \
#     --sampling-strategy pxn --sampling-p 2 --sampling-n 3 \
#     --selection-strategy top-k --selection-k 2 \
#     --max-workers 4 --max-metric-calls 1000
#
# Do not pass --run-dir yourself; this script assigns one automatically.

set -euo pipefail

for arg in "$@"; do
  if [[ "$arg" == "--run-dir" || "$arg" == --run-dir=* ]]; then
    echo "Don't pass --run-dir yourself; this script assigns one automatically." >&2
    exit 1
  fi
done

# Pull --domain out of the args (if present) just to make the run dir readable.
domain="run"
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  if [[ "${args[$i]}" == "--domain" && $((i + 1)) -lt ${#args[@]} ]]; then
    domain="${args[$((i + 1))]}"
  fi
done

# Resolve paths relative to this script's own directory (the repo root), not the
# caller's cwd, and make run_dir absolute so a long background run can't be broken
# by any later cwd change or transient relative-path resolution hiccup.
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$repo_root/gepa_runs"
run_dir="$repo_root/gepa_runs/${domain}-$(date +%Y%m%d-%H%M%S)"
log_file="${run_dir}.nohup.log"

nohup uv run python -u optimize_prompt.py "$@" --run-dir "$run_dir" > "$log_file" 2>&1 &
pid=$!
disown

echo "$run_dir" > "$repo_root/gepa_runs/last_run_dir.txt"

echo "Started PID $pid"
echo "Run dir:     $run_dir"
echo "Nohup log:   $log_file"
echo
echo "Monitor:     tail -f $run_dir/run_log.txt"
echo "Stop now:    touch $run_dir/gepa.stop"
echo "Check alive: ps -p $pid"
