#!/bin/bash
#SBATCH --job-name=rsa_heldout_reports
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#
# The held-out condition-by-condition page (src.rsa.heldout_report) of every
# finished cell, from each loop's own fit cache: the held-out evaluation
# already fitted every model at these settings, so this only predicts.
# It writes heldout/report.html, report.bundle.json and unit_lpd.csv into the
# cell's outputs/ ($CELLS_ROOT/<cell>/outputs), the directory the bring-back
# copies from, so the Mac's usual rsync brings the pages back too (run 2 had to
# build a scratch copy of the brought-back cells). From the checkout:
#
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/heldout_reports.sbatch
#
# (without -o the log lands in the checkout). CELLS: a space-separated subset
# (default: every finished cell). A cell not yet finished is skipped with a
# note; any cell whose page fails makes the job fail.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"

failed=""
for (( task = 0; task < RSA_N_CELLS; task++ )); do
  rsa_cell "$task"
  if [[ -n "${CELLS:-}" && " $CELLS " != *" $CELL "* ]]; then continue; fi
  CELL_DIR="$CELLS_ROOT/$CELL"
  DEST="$CELL_DIR/outputs"
  if [[ ! -f "$CELL_DIR/DONE" ]]; then echo "[$CELL] not finished: skipped"; continue; fi
  [[ -f "$DEST/heldout/heldout.json" ]] || { echo "[$CELL] FAILED: finished, but $DEST has no heldout/heldout.json" >&2; failed="$failed $CELL"; continue; }
  RESULTS="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")/repo/_runs/loop"
  DATA_DIR="$("$VENV_PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["data_dir"])' "$CELL_DIR/cell.json")"
  echo "[$CELL] held-out page from $RESULTS"
  if ! (cd "$REPO" && "$VENV_PY" -m src.rsa.heldout_report --cell "$DEST" --loop-dir "$RESULTS" \
          --data-sha256 "$CELL_DIR/data.sha256" --test "$DATA_DIR/test.csv" --work "$CELL_DIR/heldout_report_work"); then
    echo "[$CELL] FAILED" >&2
    failed="$failed $CELL"
  fi
done
if [[ -n "$failed" ]]; then
  echo "ERROR: the held-out page failed for:$failed" >&2
  exit 1
fi
echo "done: $CELLS_ROOT/<cell>/outputs/heldout/report.html (bring them back with the cells' outputs)"
