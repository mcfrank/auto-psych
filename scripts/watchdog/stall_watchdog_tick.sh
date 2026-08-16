#!/bin/bash
# Stall watchdog for holdout-recovery array tasks.
#
# Flags (and optionally cancels) any RUNNING task named "holdout_recovery" whose
# Slurm .out log has not been written to for >= STALL_MIN minutes. With the NUTS
# progress bar off, a single pm.sample() call produces no log output while it
# runs, so a live-but-slow fit also looks "stale" — the threshold is set well
# above the longest legitimate single fit (motif_stack worst-case ~20 min), so
# only a genuine hang (e.g. the NUTS max-treedepth pathology that idled a 64eig
# task for 22 h with no output) trips it.
#
# Actions (STALL_ACTION env):
#   detect  (default) append an alert line only; never cancel
#   cancel            additionally `scancel` the stalled task — frees the node
#                     and unblocks an afterany analysis job, turning a 22 h hang
#                     into ~STALL_MIN-then-killed. The threshold guards legit
#                     fits; raise STALL_MIN if a real fit ever exceeds it.
#
# Usage: stall_watchdog_tick.sh <logdir> [<logdir> ...]
#   <logdir>...   one or more run slurm_logs dirs to search for each task's
#                 holdout_recovery_<A>_<a>.out (shell globs are fine; missing
#                 dirs are skipped). A task's log may live in any of them, so
#                 one tick can watch several concurrent runs.
# Env:
#   STALL_MIN     idle threshold in minutes before flagging   (default 90)
#   STALL_ACTION  detect | cancel                             (default detect)
#   STALL_ALERTS  alert-log path (default under the first watched dir's parent)
set -uo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: stall_watchdog_tick.sh <logdir> [<logdir> ...]" >&2
  exit 2
fi
LOGDIRS=("$@")
STALL_MIN="${STALL_MIN:-90}"
STALL_ACTION="${STALL_ACTION:-detect}"
ALERTS="${STALL_ALERTS:-$(dirname "${LOGDIRS[0]}")/stall_alerts.log}"
mkdir -p "$(dirname "$ALERTS")" 2>/dev/null || true
now=$(date +%s)
stamp=$(date '+%Y-%m-%d %H:%M:%S')

# Running array tasks named holdout_recovery, as "<arrayjobid>_<taskid>".
mapfile -t running < <(
  squeue --me -h -t RUNNING -o '%i %j' 2>/dev/null | awk '$2=="holdout_recovery"{print $1}'
)
if [[ ${#running[@]} -eq 0 ]]; then
  echo "[$stamp] no running holdout_recovery tasks"
  exit 0
fi

flagged=0
cancelled=0
for jid in "${running[@]}"; do
  # Locate this task's log across the candidate dirs (a task's log lives in the
  # slurm_logs of whichever run it belongs to).
  f=""
  for d in "${LOGDIRS[@]}"; do
    if [[ -f "$d/holdout_recovery_${jid}.out" ]]; then
      f="$d/holdout_recovery_${jid}.out"
      break
    fi
  done
  [[ -n "$f" ]] || continue            # not in any watched dir (or no log yet)
  m=$(stat -c '%Y' "$f" 2>/dev/null) || continue
  stale_min=$(( (now - m) / 60 ))
  (( stale_min >= STALL_MIN )) || continue

  last=$(tail -1 "$f" 2>/dev/null | tr -d '\r' | cut -c1-90)
  msg="[$stamp] STALL: task $jid idle ${stale_min}m (>= ${STALL_MIN}m) | log: $f | last: $last"
  echo "$msg"
  echo "$msg" >> "$ALERTS"
  flagged=$((flagged + 1))

  if [[ "$STALL_ACTION" == "cancel" ]]; then
    if scancel "$jid" 2>/dev/null; then
      cmsg="[$stamp] CANCELLED stalled task $jid (idle ${stale_min}m >= ${STALL_MIN}m)"
      echo "$cmsg"
      echo "$cmsg" >> "$ALERTS"
      cancelled=$((cancelled + 1))
    else
      wmsg="[$stamp] WARNING: scancel $jid failed"
      echo "$wmsg" >&2
      echo "$wmsg" >> "$ALERTS"
    fi
  fi
done
echo "[$stamp] checked ${#running[@]} running task(s), flagged $flagged, cancelled $cancelled (threshold ${STALL_MIN}m, action ${STALL_ACTION})"
