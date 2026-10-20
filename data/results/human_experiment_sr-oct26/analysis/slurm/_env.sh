#!/bin/bash
# Shared environment for the sr-oct26 generalization jobs (sourced by every sbatch here).
#
# Honors, if exported:
#   REPO       the auto-psych checkout, with this analysis folder (default $HOME/auto-psych)
#   WORK_ROOT  scratch for the uv cache and the MCMC fit cache, reconstructible
#              (default $SCRATCH/auto-psych/sr_oct26_generalization)
# Python environments live in $GROUP_HOME (Sherlock: not $HOME, not the purged $SCRATCH).
set -euo pipefail

export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/sr_oct26_generalization}"
mkdir -p "$WORK_ROOT"

ml purge
ml load devel
# PyTensor JIT-compiles every model with a C compiler; el7's gcc 4.8.5 is too old.
ml load gcc/14.2.0 2>/dev/null || ml load gcc
ml load system uv 2>/dev/null || ml load uv

# The repo's Python is 3.11 (.python-version; 3.12+ resolves arviz 1.x, which
# breaks the comparison code) and Sherlock has no 3.11 module, so uv installs
# it, beside the venv in $GROUP_HOME.
export UV_PYTHON=3.11
export UV_PYTHON_INSTALL_DIR="$GROUP_HOME/benpry/uv-python"
export UV_PROJECT_ENVIRONMENT="$GROUP_HOME/benpry/venvs/sr_oct26_generalization"
export UV_CACHE_DIR="$WORK_ROOT/.uv_cache"
export UV_LINK_MODE=copy  # cache and venv are on different filesystems
export FIT_CACHE="$WORK_ROOT/mcmc_cache"
mkdir -p "$UV_CACHE_DIR" "$FIT_CACHE" "$(dirname "$UV_PROJECT_ENVIRONMENT")"

# One chain per core; keep BLAS from oversubscribing.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

# Per-task PyTensor compile dir and XDG cache on node-local disk: a shared
# compile lock stalls concurrent tasks, and arviz writes a once-a-day marker
# under XDG_CACHE_HOME through a fixed-name temporary file (tasks starting
# together collide on it).
TASK_TAG="${SLURM_JOB_ID:-0}_${SLURM_ARRAY_TASK_ID:-0}"
LOCAL="${L_SCRATCH:-/tmp}/sr_oct26_gen_$TASK_TAG"
mkdir -p "$LOCAL/pytensor" "$LOCAL/xdg"
export PYTENSOR_FLAGS="base_compiledir=$LOCAL/pytensor"
export XDG_CACHE_HOME="$LOCAL/xdg"

ANALYSIS="$REPO/data/results/human_experiment_sr-oct26/analysis"
export ANALYSIS
cd "$REPO"
echo "[env] host=$(hostname) REPO=$REPO commit=$(git rev-parse --short HEAD) FIT_CACHE=$FIT_CACHE"
git status --short -- src | sed 's/^/[env] modified: /'  # the analysis imports src/
