#!/bin/bash
# Take a cell's lock for this Slurm job, unless another live job holds it.
#
#   cell_lock.sh <lock_file> <job_id>     exit 0: taken; exit 3: held elsewhere
#
# The retry job and the sweep watcher can both resubmit a failed cell; two
# jobs working in one cell (rebuilding its agent tree, writing its
# experiments) would corrupt it. A lock left by a job that is no longer in the
# queue is stale and is taken over.
set -uo pipefail
[[ $# -eq 2 ]] || { echo "usage: $0 <lock_file> <job_id>" >&2; exit 2; }
lock="$1"; job="$2"
SQUEUE="${SQUEUE:-squeue}"
if [[ -f "$lock" ]]; then
  holder=$(cat "$lock")
  # Live only if squeue prints exactly the holder's id (%A: each array task's
  # own id). For a job no longer in the queue squeue prints "Invalid job id
  # specified" on stdout, so non-empty output alone would keep a stale lock —
  # left by a job killed at its time limit — forever.
  if [[ "$holder" != "$job" ]] && "$SQUEUE" -h -j "$holder" -o %A 2>/dev/null | grep -qx "$holder"; then
    echo "cell is already running as job $holder"
    exit 3
  fi
fi
echo "$job" > "$lock"
