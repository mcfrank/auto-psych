#!/bin/bash
# Stall watchdog for holdout-recovery array tasks.
#
# Flags any RUNNING task named "holdout_recovery" whose Slurm .out log has not
# been written to for >= STALL_MIN minutes. With the NUTS progress bar off, a
# single pm.sample() call produces no log output while it runs, so a live-but-
# slow fit also looks "stale" — the threshold is deliberately set well above the
# longest legitimate single fit, so only a genuine hang (e.g. the target_accept
# pathology that idled tasks for many hours) trips it.
#
# Detection ONLY: appends an alert line, does not cancel or requeue anything.
#
# Usage: stall_watchdog_tick.sh <run_slurm_logdir> [stall_minutes]
#   run_slurm_logdir  the run's slurm_logs dir (holds holdout_recovery_<A>_<a>.out)
#   stall_minutes     idle threshold before flagging (default 90)
set -uo pipefail

LOGDIR="${1:?usage: stall_watchdog_tick.sh <run_slurm_logdir> [stall_minutes]}"
STALL_MIN="${2:-90}"
ALERTS="$(dirname "$LOGDIR")/stall_alerts.log"
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
for jid in "${running[@]}"; do
  f="$LOGDIR/holdout_recovery_${jid}.out"
  [[ -f "$f" ]] || continue            # task just started, no log yet
  m=$(stat -c '%Y' "$f" 2>/dev/null) || continue
  stale_min=$(( (now - m) / 60 ))
  if (( stale_min >= STALL_MIN )); then
    last=$(tail -1 "$f" 2>/dev/null | tr -d '\r' | cut -c1-90)
    msg="[$stamp] STALL: task $jid idle ${stale_min}m (>= ${STALL_MIN}m) | last: $last"
    echo "$msg"
    echo "$msg" >> "$ALERTS"
    flagged=$((flagged + 1))
  fi
done
echo "[$stamp] checked ${#running[@]} running task(s), flagged $flagged (threshold ${STALL_MIN}m)"
