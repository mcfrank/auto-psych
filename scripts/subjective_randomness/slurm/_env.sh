#!/bin/bash
# Shared environment for the holdout-recovery test-retest Slurm jobs.
# Sourced by the setup, array, and analysis sbatch scripts.
#
# Honors these env vars if already exported (otherwise uses the defaults):
#   REPO        - path to the auto-psych checkout
#   WORK_ROOT   - where all run output / caches / venv live (must NOT be $HOME)
#   UV_PROJECT_ENVIRONMENT - shared venv location
set -euo pipefail

# --- repo + work locations -------------------------------------------------
export REPO="${REPO:-$HOME/auto-psych}"
# Heavy I/O and large files must stay off $HOME (15 GB, NFS-backed). Default to
# $SCRATCH; fall back to $GROUP_SCRATCH if $SCRATCH is unset.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_test_retest}"
mkdir -p "$WORK_ROOT"

# --- modules ---------------------------------------------------------------
# uv drives the Python env; opencode is the coding-agent backend.
ml purge
ml load devel
# A modern compiler is REQUIRED at RUN TIME: pytensor JIT-compiles every PyMC
# model with a C compiler, and el7's system gcc is 4.8.5. The module also sets
# CC/CXX and puts the matching libstdc++ on LD_LIBRARY_PATH. (gcc/14.2.0 is the
# login default, but `ml purge` strips it.)
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
ml load system uv 2>/dev/null || ml load uv
ml load opencode 2>/dev/null || true

# --- python version --------------------------------------------------------
# Pin a STABLE Python. uv otherwise grabs the newest (3.14), for which almost no
# binary package ships wheels yet -> everything (numpy/scipy/pandas/matplotlib/
# greenlet) builds from source on el7, which is slow and fails (no compiler /
# freetype download). 3.12 has full manylinux2014 (glibc 2.17) wheel coverage,
# so the sync is all wheels and fast.
export UV_PYTHON="${UV_PYTHON:-3.12}"

# --- caches off $HOME ------------------------------------------------------
# Per-run venv under $WORK_ROOT (not a shared dir): a single shared venv couples
# independent runs — one run's `uv pip install` can corrupt the packages another
# run is importing. One venv per WORK_ROOT keeps them isolated. It's all wheels,
# so rebuilding is ~25s; override UV_PROJECT_ENVIRONMENT to reuse one elsewhere.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}"
# Jobs run the venv's interpreter directly (the env is built with `uv pip`, not
# `uv sync`, so there's no project lock for `uv run` to reconcile).
export VENV_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export UV_CACHE_DIR="$WORK_ROOT/.uv_cache"
export PIP_CACHE_DIR="$WORK_ROOT/.pip_cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export HF_HOME="$WORK_ROOT/.hf"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"

# --- coding-agent backend --------------------------------------------------
# The config selects opencode + google/gemini-3.1-pro-preview. opencode needs
# its provider credentials. We export GOOGLE_API_KEY from .secrets (if present)
# and also expose it under the names common Gemini SDKs/opencode look for.
if [[ -f "$REPO/.secrets" ]]; then
  # .secrets as a KEY=value file.
  while IFS='=' read -r k v || [[ -n "$k" ]]; do
    k="${k// /}"
    [[ -z "$k" || "$k" == \#* ]] && continue
    export "$k=$(echo "$v" | xargs)" || true
  done < "$REPO/.secrets"
elif [[ -d "$REPO/.secrets" ]]; then
  # .secrets as a directory, one file per key.
  for f in "$REPO/.secrets"/*; do
    [[ -f "$f" ]] || continue
    export "$(basename "$f")=$(tr -d '\n' < "$f")" || true
  done
fi
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
export GOOGLE_GENERATIVE_AI_API_KEY="${GOOGLE_GENERATIVE_AI_API_KEY:-${GOOGLE_API_KEY:-}}"

# --- threading -------------------------------------------------------------
# PyMC samples one chain per core; keep BLAS from oversubscribing within a chain.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

cd "$REPO"
echo "[_env] REPO=$REPO"
echo "[_env] WORK_ROOT=$WORK_ROOT"
echo "[_env] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
