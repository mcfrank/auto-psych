#!/bin/bash
# run_live.sbatch's resources (its #SBATCH lines do not apply when it is run
# from this job); the time limit and QOS are given at submission.
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
# ============================================================================
#  Run 1, experiments 2-3, on a pinned code snapshot (2026-09-30).
#
#  Submitted by queue_run1_exps23.sh behind job A (experiment 1's model loop),
#  which runs from run 1's copy $WORK_ROOT/runs/run1/repo. The copy cannot be
#  updated while A runs (its fit processes import from it), so the new code
#  waits in a pinned snapshot beside it (runs/run1/repo_next, made by the
#  queue script from a clean checkout at EXPECTED_COMMIT). This job starts
#  after A, syncs the snapshot into the copy (public/, with experiment 1's
#  live page, and the other launcher excludes are left alone), then runs
#  run_live.sbatch from the copy: design, implement, deploy and PUBLISH a
#  Prolific study for each of experiments 2 and 3.
# ============================================================================
set -euo pipefail

: "${EXPECTED_COMMIT:?}" "${WORK_ROOT:?}" "${RUN_WORKTREE:?}" "${AUTO_PSYCH_OUTPUT_DIR:?}" "${VENV_PY:?}"
SNAP="$WORK_ROOT/runs/run1/repo_next"
EXP="$AUTO_PSYCH_OUTPUT_DIR/subjective_randomness"
die() { echo "STOP: $*" >&2; exit 1; }

prov() { "$VENV_PY" -c 'import json,sys; p=json.load(open(sys.argv[1])); print(p["git_commit"], p["git_dirty"])' "$1"; }
[[ "$(prov "$SNAP/code_provenance.json")" == "$EXPECTED_COMMIT False" ]] \
  || die "snapshot $SNAP is not a clean $EXPECTED_COMMIT"
[[ -f "$EXP/experiment1/model_loop/export_complete.json" ]] \
  || die "experiment 1's model loop has no export record: it did not finish"
[[ ! -e "$EXP/experiment2" ]] || die "$EXP/experiment2 already exists"
[[ -d "$RUN_WORKTREE/public/e1-run1" ]] || die "run copy has no public/e1-run1"

echo "[sync] $SNAP -> $RUN_WORKTREE"
rsync -a --delete --itemize-changes \
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees' \
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  "$SNAP"/ "$RUN_WORKTREE"/
touch "$RUN_WORKTREE/.here"
[[ "$(prov "$RUN_WORKTREE/code_provenance.json")" == "$EXPECTED_COMMIT False" ]] \
  || die "the run copy is not at $EXPECTED_COMMIT after the sync"
[[ -d "$RUN_WORKTREE/public/e1-run1" ]] || die "the sync removed public/e1-run1"
echo "[sync] run copy at $EXPECTED_COMMIT"

# The job script and its _env.sh from the synced copy, so a later change to
# the checkout cannot reach this run (REPO stays the checkout: .secrets).
export OUTER_LIVE_SLURM_DIR="$RUN_WORKTREE/scripts/outer_loop_live"
exec bash "$OUTER_LIVE_SLURM_DIR/run_live.sbatch"
