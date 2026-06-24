#!/bin/bash
# install_watchdog.sh -- register the OOM watchdog as a scheduled (scrontab) job.
#
# Sherlock blocks long-lived "sleeper" jobs, so the watchdog runs as a short tick
# every few minutes via Slurm cron instead of one looping job. This script adds
# (or refreshes) a single scrontab entry; it never clobbers your other entries.
#
# Usage:
#   bash scripts/watchdog/install_watchdog.sh
#
# Overrides (env vars):
#   WD_STATE_DIR=$SCRATCH/auto-psych/watchdog/run2   # recipes/ledger/logs
#   WD_PARTITION=hns        # partition for the tick job (default normal)
#   WD_POLL_MINUTES=3       # minutes between ticks (default 5)
#   WD_TICK_WALLTIME=00:30:00   # max time for one tick (default 00:20:00)
#   WD_MODEL=claude-opus-4-8  WD_MEM_CAP_GB=512  WD_MAX_RETRIES=5  ...
set -euo pipefail

WD_TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# State (recipes, ledger, logs) must live off $HOME -- default to $SCRATCH.
export WD_STATE_DIR="${WD_STATE_DIR:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/watchdog/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$WD_STATE_DIR"

PARTITION="${WD_PARTITION:-normal}"
TICK_WALLTIME="${WD_TICK_WALLTIME:-00:20:00}"
POLL_MINUTES="${WD_POLL_MINUTES:-5}"

# Persist WD_* overrides for the tick to source (scrontab jobs don't reliably
# inherit this shell's environment).
: > "$WD_STATE_DIR/tick_env.sh"
for v in WD_MODEL WD_MAX_TURNS WD_EXIT_AFTER_IDLE_TICKS WD_MEM_CAP_GB WD_MAX_RETRIES \
         WD_BIGMEM_THRESHOLD_GB WD_NORMAL_PERCORE_GB WD_BIGMEM_PERCORE_GB; do
  [[ -n "${!v:-}" ]] && printf 'export %s=%q\n' "$v" "${!v}" >> "$WD_STATE_DIR/tick_env.sh"
done

# One watchdog at a time: a generic marker so re-installing replaces the old entry.
MARKER_BEGIN="# >>> oom_watchdog"
MARKER_END="# <<< oom_watchdog"
block="$MARKER_BEGIN
#SCRON --partition=$PARTITION --time=$TICK_WALLTIME --cpus-per-task=1 --mem=4G --job-name=oom_watchdog
#SCRON --output=$WD_STATE_DIR/tick.out --open-mode=append
*/$POLL_MINUTES * * * * $WD_TOOLKIT_DIR/watchdog_tick.sh $WD_STATE_DIR
$MARKER_END"

# Merge into any existing scrontab, dropping a previous watchdog block first.
# NB: `scrontab -l` prints "no crontab for <user>" to STDOUT and exits non-zero
# when you have no crontab yet -- so gate on its exit code, never pipe its output
# blindly (that message would otherwise land in the file and fail to parse).
tmp="$(mktemp)"; cur="$(mktemp)"
if scrontab -l > "$cur" 2>/dev/null; then
  awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0==b {skip=1} skip {if ($0==e) skip=0; next} {print}' "$cur" > "$tmp"
else
  : > "$tmp"                      # no existing crontab
fi
printf '%s\n' "$block" >> "$tmp"
scrontab "$tmp"
rm -f "$tmp" "$cur"

echo "OOM watchdog scheduled via scrontab"
echo "  toolkit dir : $WD_TOOLKIT_DIR"
echo "  state dir   : $WD_STATE_DIR"
echo "  partition   : $PARTITION    tick walltime: $TICK_WALLTIME    every: ${POLL_MINUTES} min"
echo "  watching    : all jobs owned by $USER as of the first tick"
echo
echo "verify:    scrontab -l"
echo "follow:    tail -f $WD_STATE_DIR/tick.out"
echo "did what:  WD_STATE_DIR=$WD_STATE_DIR $WD_TOOLKIT_DIR/wd.sh status"
echo "stop:      bash $WD_TOOLKIT_DIR/uninstall_watchdog.sh"
