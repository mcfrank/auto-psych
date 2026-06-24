#!/bin/bash
# uninstall_watchdog.sh -- remove the watchdog's scrontab entry and cancel any
# currently-running tick. Leaves your other scrontab entries untouched.
set -uo pipefail

MARKER_BEGIN="# >>> oom_watchdog"
MARKER_END="# <<< oom_watchdog"

# Gate on exit code: with no crontab, `scrontab -l` prints a message to STDOUT
# and exits non-zero -- there is then nothing to remove.
tmp="$(mktemp)"; cur="$(mktemp)"
if scrontab -l > "$cur" 2>/dev/null; then
  awk -v b="$MARKER_BEGIN" -v e="$MARKER_END" '
    $0==b {skip=1} skip {if ($0==e) skip=0; next} {print}' "$cur" > "$tmp"
  if [[ -s "$tmp" ]]; then
    scrontab "$tmp"                   # keep the other entries
  else
    scrontab -r 2>/dev/null || true   # our block was the only entry
  fi
fi
rm -f "$tmp" "$cur"

# scron is in explicit_scancel mode, so a running tick must be cancelled by hand.
scancel --me --name=oom_watchdog 2>/dev/null || true

echo "removed oom_watchdog from scrontab and cancelled any running tick."
echo "current scrontab:"
scrontab -l 2>/dev/null || echo "(empty)"
