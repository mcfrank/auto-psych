#!/bin/bash
# Copy the final recovery runs' result files from scratch into the repo, at
# data/results/final_recovery, so they can be synced off the cluster.
#
# The five literature repeats ran as three sweeps; here they are pooled into one
# run root, numbered by repeat, which is the layout the holdout aggregation and
# plotting code reads (run<r>/<gt>/holdout.json):
#
#   final_1/run1  -> run1        final_23/run1 -> run2    final_23/run2 -> run3
#   final_45/run1 -> run4        final_45/run2 -> run5
#
# The impossible-model control is copied to impossible/run1 (final_impossible_1).
#
# Only light files are copied (holdout.*, trajectory.json, agent_activity.md,
# eval_exclusions.jsonl, code_commit; ~200 KB a cell): never the MCMC caches or
# the agent trees. A cell with no holdout.json is unfinished and is skipped and
# listed. Re-run it when the remaining cells finish; it replaces the
# destination.
#
# Usage:
#   bash scripts/subjective_randomness/collect_final_results.sh
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SRC="${SCRATCH:?SCRATCH is unset}/auto-psych"
DEST="$REPO/data/results/final_recovery"

CELL_FILES=(holdout.json holdout.csv holdout.png trajectory.json
            agent_activity.md eval_exclusions.jsonl code_commit)
SWEEP_FILES=(LAUNCH.txt test_retest.json test_retest.csv test_retest.png
             MISSING_CELLS.txt OUTSIDE_PATH_WARNINGS.txt code_commit)

# "<sweep>/<run dir> <destination run dir>"
RUN_MAP=(
  "final_1/run1 run1"
  "final_23/run1 run2"
  "final_23/run2 run3"
  "final_45/run1 run4"
  "final_45/run2 run5"
  "final_impossible_1/run1 impossible/run1"
)

rm -rf "$DEST"
mkdir -p "$DEST"
unfinished=()

for entry in "${RUN_MAP[@]}"; do
  read -r src_run dest_run <<< "$entry"
  [[ -d "$SRC/$src_run" ]] || { unfinished+=("$src_run (no run directory yet)"); continue; }
  for cell in "$SRC/$src_run"/*/; do
    gt="$(basename "$cell")"
    if [[ ! -f "$cell/holdout.json" ]]; then
      unfinished+=("$src_run/$gt -> $dest_run/$gt")
      continue
    fi
    mkdir -p "$DEST/$dest_run/$gt"
    for f in "${CELL_FILES[@]}"; do
      [[ -f "$cell/$f" ]] && cp --preserve=timestamps "$cell/$f" "$DEST/$dest_run/$gt/"
    done
  done
done

# Each sweep's own launch record and summary, as the sweep wrote them.
for sweep in final_1 final_23 final_45 final_impossible_1; do
  [[ -d "$SRC/$sweep" ]] || continue
  mkdir -p "$DEST/sweeps/$sweep"
  for f in "${SWEEP_FILES[@]}"; do
    [[ -f "$SRC/$sweep/$f" ]] && cp --preserve=timestamps "$SRC/$sweep/$f" "$DEST/sweeps/$sweep/"
  done
done

{
  echo "# Final recovery runs"
  echo
  echo "Collected $(date '+%Y-%m-%d %H:%M') by scripts/subjective_randomness/collect_final_results.sh."
  echo
  echo "Claude Opus 5.5 agents (API billing), faithful config. Repeats, by sweep:"
  echo
  for entry in "${RUN_MAP[@]}"; do
    read -r src_run dest_run <<< "$entry"
    echo "- \`$dest_run/\` <- \`\$SCRATCH/auto-psych/$src_run\`"
  done
  echo
  echo "\`sweeps/<sweep>/\` holds each sweep's launch record and its own test_retest summary"
  echo "(over that sweep's repeats only)."
  echo
  echo "## Cells without a result at collection time"
  echo
  if (( ${#unfinished[@]} == 0 )); then
    echo "None."
  else
    for u in "${unfinished[@]}"; do echo "- $u"; done
  fi
} > "$DEST/README.md"

echo "collected into $DEST"
echo "cells: $(find "$DEST" -name holdout.json | wc -l) with results, ${#unfinished[@]} without"
printf '  unfinished: %s\n' "${unfinished[@]}"
