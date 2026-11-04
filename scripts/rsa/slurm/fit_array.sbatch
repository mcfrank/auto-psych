#!/bin/bash
#SBATCH --job-name=rsa_open_fits
#SBATCH --cpus-per-task=1
#SBATCH --output=%x_%A_%a.out
#
# One fit of a step's array (src.rsa_open.loop.fit_array; DISTRIBUTED_DESIGN.md v3 §0b). Submitted by
# src.rsa_open.loop.slurm.submit_step, held until the step's continuation is in, with:
#   $1                      the array's manifest (one fit per task, all of one memory size)
#   --export=NONE plus VENV_PY, HARNESS_REPO, JAX_COMPILATION_CACHE_DIR: nothing else. In particular no
#   .secrets: a fit runs agent-written theory code, so this script never sources _env.sh.
# The task fits exactly as the local backend's child does (its time limit, its peak memory) and writes
# a result beside the fit. A theory's own failure is a result and exits 0 (no mail per task); only the
# task's own trouble exits nonzero.
set -euo pipefail
: "${VENV_PY:?}" "${HARNESS_REPO:?}" "${JAX_COMPILATION_CACHE_DIR:?}" "${SLURM_ARRAY_TASK_ID:?}" "${1:?a manifest}"
type ml >/dev/null 2>&1 || source /etc/profile >/dev/null 2>&1 || true
ml load devel 2>/dev/null || true
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
# One core per fit, as in every RSA job: XLA and OpenBLAS would otherwise size their pools by the node.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
export OPENBLAS_NUM_THREADS=1
export XDG_CACHE_HOME="${L_SCRATCH:-${TMPDIR:-/tmp}}/rsa_open_fit_${SLURM_JOB_ID}"
export PYTENSOR_FLAGS="base_compiledir=$XDG_CACHE_HOME/pytensor"
mkdir -p "$XDG_CACHE_HOME"
cd "$HARNESS_REPO"
exec "$VENV_PY" -m src.rsa_open.loop.fit_array --manifest "$1" --index "$SLURM_ARRAY_TASK_ID"
