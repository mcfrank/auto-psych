#!/bin/bash
# Assemble data/results/number_game_dataset-nov26/ from the staged dataset-run work roots.
# Every data file is a byte copy (checked with cmp) or a program's output.
#   assemble.sh
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/ngds-nov26-collect
REPO=/sailhome/wurgaft/auto-psych
D=$REPO/data/results/number_game_dataset-nov26
A=$S/assembled
declare -A NODE=([run1]=tiger3 [run2]=tiger1 [run3]=tiger4)
RECORDS="holdout.json trajectory.json agent_activity.md code_commit"

rm -rf "$A" "$D"
mkdir -p "$D"
for run in run1 run2 run3; do
  src=$S/staged/${NODE[$run]}/$run/dataset
  mkdir -p "$D/harness/$run"
  for f in $RECORDS; do cp "$src/$f" "$D/harness/$run/$f" && cmp "$src/$f" "$D/harness/$run/$f"; done
  # The collector's input: <source>/<run>/data/number_game/experiment*.
  dest=$A/collect_src/$run/data/number_game
  mkdir -p "$dest" "$A/untar/$run"
  tar -xzf "$src/agent_runs.tar.gz" -C "$A/untar/$run"
  mv "$A/untar/$run/_runs/cell_1/"* "$dest/"
done
cp "$S/staged/tiger3/code_commit" "$D/harness/sweep_code_commit"
for n in tiger1 tiger4; do cmp "$S/staged/tiger3/code_commit" "$S/staged/$n/code_commit"; done

cd "$REPO"
uv run python scripts/outer_loop_live/collect_human_results.py --source "$A/collect_src" \
  --dest "$D/collected" --project number_game --runs run1 run2 run3
echo "assembled $D"
