#!/bin/bash
# watchdog_tick.sh -- ONE watchdog tick, run on a schedule by scrontab (Slurm cron).
#
# Sherlock forbids long-lived "sleeper" jobs (a job that holds an allocation while
# idle between polls). So instead of one looping job, we run a short job every few
# minutes: capture recipes, scan for OOMs, and -- only when an OOM actually appears
# -- invoke Claude to confirm and resubmit. Each tick exits promptly.
#
# Usage (normally via scrontab; install_watchdog.sh sets that up):
#   watchdog_tick.sh <WD_STATE_DIR>

set -uo pipefail

export WD_STATE_DIR="${1:?usage: watchdog_tick.sh <WD_STATE_DIR>}"
WD_TOOLKIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PATH="$WD_TOOLKIT_DIR:$PATH"          # so Claude can call `wd.sh` by name
WD="$WD_TOOLKIT_DIR/wd.sh"
PROMPT_FILE="$WD_TOOLKIT_DIR/watchdog_prompt.md"

# Per-run overrides written by install_watchdog.sh (scrontab jobs don't reliably
# inherit the shell that created the crontab).
[[ -f "$WD_STATE_DIR/tick_env.sh" ]] && source "$WD_STATE_DIR/tick_env.sh"

MODEL="${WD_MODEL:-claude-sonnet-4-6}"
MAX_TURNS="${WD_MAX_TURNS:-30}"
EXIT_AFTER_IDLE_TICKS="${WD_EXIT_AFTER_IDLE_TICKS:-2}"

note() { printf '%s [tick %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "${SLURM_JOB_ID:-?}" "$*"; }

# Once retired (nothing left to watch), ticks are a near-instant no-op until the
# scrontab entry is removed.
if [[ -f "$WD_STATE_DIR/retired" ]]; then
  note "retired; nothing to watch -- remove the schedule with uninstall_watchdog.sh"
  exit 0
fi

# Make `ml`/`module` available even if this scheduled job started with a bare
# environment, then load Claude Code.
if ! command -v module >/dev/null 2>&1; then
  for f in "${MODULESHOME:-}/init/bash" /share/software/user/open/lmod/lmod/init/bash /etc/profile.d/modules.sh; do
    [[ -r "$f" ]] && { source "$f"; break; }
  done
fi
ml load claude-code 2>/dev/null || module load claude-code 2>/dev/null || true
command -v claude >/dev/null || { note "FATAL: claude not on PATH after module load"; exit 1; }
command -v "$WD"  >/dev/null || { note "FATAL: wd.sh not found at $WD"; exit 1; }

# Claude's tools: the wd.sh helper (the ONLY state-changing command) plus
# read-only inspection. It cannot cancel, hold, or alter jobs.
CLAUDE_ALLOWED=(
  --allowedTools
  "Bash(wd.sh:*)"
  "Bash(sacct:*)" "Bash(squeue:*)" "Bash(seff:*)" "Bash(scontrol show:*)"
  "Bash(tail:*)" "Bash(head:*)" "Bash(grep:*)" "Bash(sed:*)" "Bash(cat:*)" "Bash(awk:*)" "Bash(wc:*)" "Bash(ls:*)"
  "Read"
)

run_claude() {
  local candidates="$1" full_prompt
  full_prompt="$(cat "$PROMPT_FILE")"$'\n\n## OOM candidates this tick (one JSON object per line)\n'"$candidates"
  note "invoking Claude ($MODEL) on $(printf '%s\n' "$candidates" | grep -c .) candidate(s)"
  claude -p "$full_prompt" --model "$MODEL" --max-turns "$MAX_TURNS" "${CLAUDE_ALLOWED[@]}" \
    2>&1 | sed 's/^/    claude| /'
  note "Claude finished (exit ${PIPESTATUS[0]})"
}

# The FIRST tick is the startup snapshot of "all my jobs".
if [[ ! -s "$WD_STATE_DIR/managed_arrays.txt" ]]; then
  note "first tick: snapshotting current jobs"
  "$WD" init
fi

# Capture a resubmission recipe for any managed array now running (no-op once stored).
while read -r id; do [[ -n "$id" ]] && "$WD" capture "$id" >/dev/null 2>&1 || true; done < "$WD_STATE_DIR/managed_arrays.txt"

candidates="$("$WD" scan 2>>"$WD_STATE_DIR/watchdog.log" || true)"
if [[ -n "$candidates" ]]; then
  run_claude "$candidates"
else
  note "no OOM candidates"
fi

# Retire once nothing is left to watch, with a couple of grace ticks to catch a
# final-moment OOM.
active="$("$WD" active 2>/dev/null || echo 1)"
idle_file="$WD_STATE_DIR/idle_ticks"
if [[ "$active" == "0" ]]; then
  idle=$(( $(cat "$idle_file" 2>/dev/null || echo 0) + 1 ))
  echo "$idle" > "$idle_file"
  note "no managed jobs still queued/running (idle $idle/$EXIT_AFTER_IDLE_TICKS)"
  if (( idle >= EXIT_AFTER_IDLE_TICKS )); then
    candidates="$("$WD" scan 2>/dev/null || true)"
    [[ -n "$candidates" ]] && run_claude "$candidates"
    touch "$WD_STATE_DIR/retired"
    note "retiring -- run uninstall_watchdog.sh to remove the schedule"
  fi
else
  echo 0 > "$idle_file"
fi
