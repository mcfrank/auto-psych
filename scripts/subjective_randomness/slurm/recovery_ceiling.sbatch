#!/bin/bash
#SBATCH --job-name=recovery_ceiling
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --mail-type=END,FAIL,TIMEOUT
#
# The recovery ceiling: refit each cell's held-out ground-truth model on that
# cell's own training responses and score it on the held-out pool. That is the
# best RMSE the right functional form can reach on this data, so it separates
# what the loop failed to find from what the data cannot support.
#
# The end-to-end test runs FIRST and gates the analysis. Four analysis tools in
# this campaign shipped without ever being run against a real archived cell and
# all four failed inside a Slurm job; this one does not get to repeat that.
#
# Env (via --export): WORK_ROOT (holds repo/ and the sweeps), SWEEPS (space-
# separated sweep directory names under WORK_ROOT).

set -euo pipefail
: "${WORK_ROOT:?WORK_ROOT must be set}"
SWEEPS="${SWEEPS:-sweep sweep_rerun}"
REPO="${REPO:-$WORK_ROOT/repo}"
VENV_PY="${VENV_PY:-$WORK_ROOT/venv/bin/python}"
OUT_ROOT="$WORK_ROOT/ceiling"

ml purge
ml load devel
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true

# Caches off $HOME (15 GB, NFS), compile dir on node-local scratch.
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/matplotlib"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-$WORK_ROOT/.cache}/pytensor"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p "$XDG_CACHE_HOME" "$MPLCONFIGDIR" "$OUT_ROOT"

cd "$REPO"
echo "[ceiling] repo $REPO @ $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "[ceiling] sweeps: $SWEEPS"

# --- gate: the tool must work on a real archived cell before it runs on all of them
echo "[ceiling] === gate: end-to-end test against a real archived cell ==="
CEILING_TEST_SWEEP="$WORK_ROOT/$(echo "$SWEEPS" | awk '{print $NF}')" \
  "$VENV_PY" -m pytest -q -p no:cacheprovider tests/test_recovery_ceiling.py
echo "[ceiling] gate passed"

# --- the analysis
for sweep in $SWEEPS; do
  root="$WORK_ROOT/$sweep"
  if [[ ! -d "$root" ]]; then
    echo "[ceiling] ERROR: no sweep at $root" >&2
    exit 1
  fi
  echo "[ceiling] === $sweep ==="
  "$VENV_PY" scripts/subjective_randomness/recovery_ceiling.py \
    --sweep "$root" --out "$OUT_ROOT/$sweep"
done

echo "[ceiling] done -> $OUT_ROOT"
