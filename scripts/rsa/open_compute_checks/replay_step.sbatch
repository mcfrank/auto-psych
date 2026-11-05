#!/bin/bash
# One step of a step-mode rehearsal with no agents (replay_step.py). Set up like open_rehearsal.sbatch's
# step mode: _step_mode.sh writes the continuation and step settings, and fit_array.sbatch runs the fits.
# Never sources _env.sh or .secrets: nothing here needs a login. First submission, with STEP_MODE=1, CHAIN,
# WORK_ROOT, HARNESS_REPO, REPLAY_INPUT, REPLAY_SOURCE, ROUNDS, CANDIDATES and optionally SEEDS exported:
#   sbatch --job-name=<chain's name> --partition=... --cpus-per-task=8 --mem=32G --time=... --no-requeue \
#     --chdir=$HARNESS_REPO -o $WORK_ROOT/logs/%x_%j.out $WORK_ROOT/replay_step.sbatch
set -euo pipefail
if [[ -n "${STEP_SETTINGS:-}" ]]; then source "$STEP_SETTINGS"; fi
: "${WORK_ROOT:?}" "${HARNESS_REPO:?}" "${CHAIN:?}" "${REPLAY_INPUT:?}" "${REPLAY_SOURCE:?}" "${ROUNDS:?}" "${CANDIDATES:?}"
export REPO="$HARNESS_REPO" STEP_MODE=1
export VENV_PY=/home/groups/mcfrank/venvs/auto-psych/bin/python
export JAX_COMPILATION_CACHE_DIR="$WORK_ROOT/jax_cache"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"
CELL="replay_chain_$CHAIN"
CELL_DIR="$WORK_ROOT/cells/$CELL"
mkdir -p "$CELL_DIR" "$JAX_COMPILATION_CACHE_DIR"
[[ ! -f "$CELL_DIR/STOP" ]] || { echo "[replay] $CELL is stopped: $(cat "$CELL_DIR/STOP")"; exit 0; }
type ml >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true
ml load devel 2>/dev/null || true
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="${L_SCRATCH_JOB:-${L_SCRATCH:-/tmp}}/replay_step_$SLURM_JOB_ID"
export PYTENSOR_FLAGS="base_compiledir=$XDG_CACHE_HOME/pytensor"
mkdir -p "$XDG_CACHE_HOME"
source "$SLURM_DIR/_step_mode.sh"
rsa_step_mode "$WORK_ROOT/replay_step.sbatch" CHAIN HARNESS_REPO REPLAY_INPUT REPLAY_SOURCE ROUNDS CANDIDATES SEEDS
cd "$HARNESS_REPO"
echo "[replay] $CELL step job ${SLURM_JOB_ID} on $(hostname) start $(date -Is) commit $(git rev-parse HEAD)"
set +e
env ${STEP_ENV[@]+"${STEP_ENV[@]}"} PYTHONPATH="$HARNESS_REPO" "$VENV_PY" "$WORK_ROOT/replay_step.py" \
  "$CELL_DIR/run" "$REPLAY_INPUT" "$REPLAY_SOURCE" "$ROUNDS" "$CANDIDATES" "${SEEDS:-}"
rc=$?
set -e
echo "[replay] $CELL step job ${SLURM_JOB_ID} python exit $rc at $(date -Is)"
case "$rc" in
  0) date -Is > "$CELL_DIR/DONE" ;;
  5|6|7) rsa_step_exit "$rc" || exit $? ;;
  *) exit "$rc" ;;
esac
