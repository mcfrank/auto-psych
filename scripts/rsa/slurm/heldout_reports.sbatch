#!/bin/bash
#SBATCH --job-name=rsa_heldout_reports
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# The held-out condition-by-condition page (src.rsa.heldout_report) of every
# finished cell, from each loop's own fit cache: the held-out evaluation
# already fitted every model at these settings, so this only predicts.
# Run after the cells are brought back into the checkout (it writes
# <cell>/heldout/report.html, report.bundle.json and unit_lpd.csv beside
# the brought-back heldout.json), from the checkout:
#
#   sbatch --chdir="$HOME/auto-psych" scripts/rsa/slurm/heldout_reports.sbatch
#
# BRING_BACK: the brought-back sweep (default data/rsa/sherlock_<run> for
# WORK_ROOT .../rsa_<run>). CELLS: a space-separated subset (default: every
# finished cell). A cell not yet finished or not brought back is skipped with
# a note; any cell whose page fails makes the job fail.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
BRING_BACK="${BRING_BACK:-$REPO/data/rsa/sherlock_$(basename "$WORK_ROOT" | sed 's/^rsa_//')}"
[[ -d "$BRING_BACK" ]] || { echo "ERROR: no brought-back sweep at $BRING_BACK (set BRING_BACK)" >&2; exit 1; }

failed=""
for (( task = 0; task < RSA_N_CELLS; task++ )); do
  rsa_cell "$task"
  if [[ -n "${CELLS:-}" && " $CELLS " != *" $CELL "* ]]; then continue; fi
  CELL_DIR="$CELLS_ROOT/$CELL"
  DEST="$BRING_BACK/$CELL"
  if [[ ! -f "$CELL_DIR/DONE" ]]; then echo "[$CELL] not finished: skipped"; continue; fi
  if [[ ! -f "$DEST/heldout/heldout.json" ]]; then echo "[$CELL] not brought back to $DEST: skipped"; continue; fi
  RESULTS="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")/repo/_runs/loop"
  DATA_DIR="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_dir"])' "$CELL_DIR/cell.json")"
  echo "[$CELL] held-out page from $RESULTS"
  if ! (cd "$REPO" && "$VENV_PY" -m src.rsa.heldout_report --cell "$DEST" --loop-dir "$RESULTS" \
          --test "$DATA_DIR/test.csv" --work "$CELL_DIR/heldout_report_work"); then
    echo "[$CELL] FAILED" >&2
    failed="$failed $CELL"
  fi
done
if [[ -n "$failed" ]]; then
  echo "ERROR: the held-out page failed for:$failed" >&2
  exit 1
fi
echo "done: $BRING_BACK/<cell>/heldout/report.html"
