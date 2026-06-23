#!/bin/bash
#SBATCH --job-name=mega_model_cmp
#SBATCH --output=mega_model_cmp_%j.out
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#
# Cost note: the held-out scheme means 24 separate fits (4 seeds refit per
# held-out run + the run winners). At smoke settings (200 draws) that took
# ~33 min; the full run (2000 draws, 4 chains) is ~1.5-2 h. 4 h gives headroom.
#
# Mega-analytic model comparison on Sherlock: fit the 8 theories (4 seeds + 4
# run winners) on the pooled human data and write metric + prediction CSVs into
# the repo (analysis/behavioral/data/). The R notebook 02 reads those.
#
# Env setup mirrors scripts/subjective_randomness/slurm/_env.sh (the repo's
# blessed Sherlock conventions): gcc for PyTensor's JIT C compiler, uv-managed
# Python 3.12, and a venv + caches kept OFF $HOME. We don't need the coding
# agent here, so this is self-contained rather than sourcing that file.
#
# Usage (from the repo root on a login node, after committing your changes):
#   sbatch analysis/behavioral/slurm/run_mega_comparison.sbatch
#   # fast pre-flight (tiny MCMC, proves the chain runs):
#   QUICK=1 sbatch analysis/behavioral/slurm/run_mega_comparison.sbatch
set -euo pipefail

export REPO="${REPO:-$(pwd)}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/mega_comparison}"
mkdir -p "$WORK_ROOT"

# --- modules: a modern gcc is REQUIRED at run time (pytensor JIT-compiles each
# PyMC model with a C compiler), plus uv to drive the Python env.
ml purge
ml load devel
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
# arviz -> h5py needs HDF5. The capped (glibc-2.17) deps have no h5py wheel, so
# it builds from source: load HDF5 (also puts libhdf5.so on LD_LIBRARY_PATH for
# import at run time) and point HDF5_DIR at it so the build finds headers/libs.
ml load hdf5/1.14.4 2>/dev/null || ml load hdf5 2>/dev/null || true
export HDF5_DIR="${HDF5_DIR:-/share/software/user/open/hdf5/1.14.4}"
ml load system uv 2>/dev/null || ml load uv

# --- Python 3.11 is REQUIRED here (not 3.12). The model-comparison code uses the
# arviz 0.x / pymc 5.x API (loo.elpd_loo, az.compare(ic="loo"), positive
# elpd_diff); on >=3.12 the lock resolves arviz 1.x / pymc 6.x and breaks it.
# See the NOTE in pyproject.toml [tool.uv] and .python-version (pinned 3.11).
export UV_PYTHON="${UV_PYTHON:-3.11}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}"
export UV_CACHE_DIR="$WORK_ROOT/.uv_cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
mkdir -p "$UV_CACHE_DIR" "$XDG_CACHE_HOME"

# --- one chain per core; keep BLAS from oversubscribing within a chain.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$REPO"

# Resolve the env from the lock (default groups include `pymc`; the lock pins
# arviz 0.x / pymc 5.x, matching the code's loo.elpd_loo API).
uv sync --frozen 2>/dev/null || uv sync
VENV_PY="$UV_PROJECT_ENVIRONMENT/bin/python"

EXTRA=()
[[ -n "${QUICK:-}" ]] && EXTRA+=(--quick)

echo "[mega] REPO=$REPO WORK_ROOT=$WORK_ROOT python=$($VENV_PY --version)"
exec "$VENV_PY" analysis/behavioral/fit_mega_models.py \
  --out-dir analysis/behavioral/data \
  --cache-dir "$WORK_ROOT/mega_cache" \
  ${EXTRA[@]+"${EXTRA[@]}"}
