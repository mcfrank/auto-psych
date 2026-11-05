#!/bin/bash
# Assemble data/results/number_game_recovery-nov26/ from the staged nov26b work roots.
# Every data file is a byte copy (checked with cmp) or a program's output.
#   assemble.sh
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
REPO=/sailhome/wurgaft/auto-psych
D=$REPO/data/results/number_game_recovery-nov26
A=$S/assembled
GTS="tenenbaum_rules_similarity bigelow_piantadosi_mixture bigelow_piantadosi_independent"

# Each cell's final staged copy: <staged root>/<run>/<gt>.
declare -A ROOT=(
  [run1/tenenbaum_rules_similarity]=final/c1/cocoflops1
  [run1/bigelow_piantadosi_mixture]=final/c1/cocoflops1
  [run1/bigelow_piantadosi_independent]=c2/cocoflops2
  [run2/tenenbaum_rules_similarity]=final/jag28/tiger5
  [run2/bigelow_piantadosi_mixture]=final/jag28/tiger5
  [run2/bigelow_piantadosi_independent]=c2/cocoflops2
  [run3/tenenbaum_rules_similarity]=jag28/jagupard28
  [run3/bigelow_piantadosi_mixture]=final/jag28/tiger5
  [run3/bigelow_piantadosi_independent]=tiger4/tiger4
)
# Each sweep root's patch log(s), as code_patches/<root>[_<node>]/.
declare -A PATCHES=(
  [c1]=final/c1/cocoflops1/code_patches
  [c2]=c2/cocoflops2/code_patches
  [jag28_jagupard28]=final/jag28_jagupard28_code_patches
  [jag28_tiger5]=final/jag28/tiger5/code_patches
  [tiger4]=tiger4/tiger4/code_patches
)
RECORDS="holdout.json holdout.csv holdout.png trajectory.json agent_activity.md code_commit"

rm -rf "$A" "$D"
mkdir -p "$A/runs_root" "$D"
for cell in "${!ROOT[@]}"; do
  src=$S/${ROOT[$cell]}/$cell
  run=${cell%%/*} gt=${cell#*/}
  # The test-retest script and cell_status read <runs root>/<run>/<gt>/.
  mkdir -p "$A/runs_root/$run" && ln -s "$src" "$A/runs_root/$cell"
  # Per-cell records, byte for byte.
  mkdir -p "$D/$cell"
  for f in $RECORDS; do cp "$src/$f" "$D/$cell/$f" && cmp "$src/$f" "$D/$cell/$f"; done
  # A redone cell's earlier finished records, without its agent archive.
  for old in "$src"/superseded/*_finished_cell; do
    [[ -d $old ]] || continue
    mkdir -p "$D/$cell/superseded/$(basename "$old")"
    for f in README holdout.json holdout.csv holdout.png trajectory.json agent_activity.md; do
      cp "$old/$f" "$D/$cell/superseded/$(basename "$old")/$f" && cmp "$old/$f" "$D/$cell/superseded/$(basename "$old")/$f"
    done
  done
  # The collector's input: <source>/<run label>/data/number_game/experiment*.
  dest=$A/collect_src/$run/$gt/data/number_game
  mkdir -p "$dest" "$A/untar/$cell"
  tar -xzf "$src/agent_runs.tar.gz" -C "$A/untar/$cell"
  mv "$A/untar/$cell/_runs/cell_1/"* "$dest/"
done
for name in "${!PATCHES[@]}"; do
  mkdir -p "$D/code_patches/$name"
  cp "$S/${PATCHES[$name]}/"* "$D/code_patches/$name/"
  for f in "$S/${PATCHES[$name]}/"*; do cmp "$f" "$D/code_patches/$name/$(basename "$f")"; done
done

cd "$REPO"
uv run python scripts/subjective_randomness/slurm/cell_status.py missing \
  --work-root "$A/runs_root" --n-repeats 3 --gt-models "$GTS" > "$D/MISSING_CELLS.txt"
uv run python scripts/subjective_randomness/holdout_test_retest.py --runs-root "$A/runs_root" \
  --out "$D/test_retest.json" --csv "$D/test_retest.csv" --figure "$D/test_retest.png" \
  --n-repeats 3 --gt-models "$GTS"
for run in run1 run2 run3; do
  uv run python scripts/outer_loop_live/collect_human_results.py --source "$A/collect_src/$run" \
    --dest "$D/collected/$run" --project number_game --runs $GTS
done
# The sweep roots' own outside-path warnings, concatenated in root order.
for r in c1/cocoflops1 c2/cocoflops2 final/jag28/tiger5 jag28/jagupard28 tiger4/tiger4; do
  f=$S/$r/OUTSIDE_PATH_WARNINGS.txt
  [[ -s $f ]] && { echo "## $r"; cat "$f"; }
done > "$D/OUTSIDE_PATH_WARNINGS.txt"
cp "$S/readme_tables.py" "$D/readme_tables.py"
echo "assembled $D"
