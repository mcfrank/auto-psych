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
# Loop agents run inside a bubblewrap filesystem sandbox (src/runtime/agent_sandbox.py).
ml load system bubblewrap
command -v bwrap >/dev/null || { echo "FATAL: bwrap not on PATH after ml load system bubblewrap" >&2; exit 1; }
# The codex backend ships as its own module (needs `devel`, loaded above) and
# authenticates from ~/.codex/auth.json (ChatGPT subscription, no API key).
# Load it only when it is the selected backend, and fail loudly if it is
# selected but unusable — a missing CLI would otherwise surface as every
# candidate slot silently writing no file.
if [[ "${AGENT_BACKEND:-}" == "codex" ]]; then
  # Prefer the self-installed CLI over the module: the newest module is
  # codex/0.151.0, which rejects GPT-6 Astra with "requires a newer version of
  # Codex" (HTTP 400). 0.155.1 in the $GROUP_HOME npm prefix accepts it.
  CODEX_BIN_DIR="${CODEX_BIN_DIR:-/home/groups/ngoodman/benpry/software/npm-global/bin}"
  if [[ -x "$CODEX_BIN_DIR/codex" ]]; then
    # The npm-installed CLI is a node script (`#!/usr/bin/env node`) and brings
    # no interpreter with it — unlike the codex module, which pulls its nodejs
    # dependency in automatically. Without this the agents run but every call
    # dies with "/usr/bin/env: node: No such file or directory", which surfaces
    # as every candidate slot writing no file.
    ml load "${NODEJS_MODULE:-nodejs/25.3.0}"
    export PATH="$CODEX_BIN_DIR:$PATH"
  else
    ml load "${CODEX_MODULE:-codex/0.151.0}"
  fi
  command -v codex >/dev/null || { echo "FATAL: AGENT_BACKEND=codex but no codex on PATH (looked in $CODEX_BIN_DIR, then module ${CODEX_MODULE:-codex/0.151.0})" >&2; exit 1; }
  [[ -f "${CODEX_HOME:-$HOME/.codex}/auth.json" ]] || { echo "FATAL: AGENT_BACKEND=codex but codex is not logged in (codex login --device-auth)" >&2; exit 1; }
  # Assert the CLI actually RUNS. `command -v` only proves the file exists; a
  # missing interpreter still resolves. An empty version string is the exact
  # symptom that preceded a whole task of empty candidate slots.
  CODEX_VERSION="$(codex --version 2>&1 || true)"
  [[ -n "${CODEX_VERSION// }" ]] || { echo "FATAL: '$(command -v codex) --version' produced no output — the CLI cannot execute (missing node?)" >&2; exit 1; }
  echo "[env] codex backend: $CODEX_VERSION at $(command -v codex), node $(node --version 2>/dev/null || echo MISSING)"
fi

# --- python version --------------------------------------------------------
# Pin a STABLE Python. uv otherwise grabs the newest (3.14), for which almost no
# binary package ships wheels yet -> everything (numpy/scipy/pandas/matplotlib/
# greenlet) builds from source on el7, which is slow and fails (no compiler /
# freetype download). 3.12 has full manylinux2014 (glibc 2.17) wheel coverage,
# so the sync is all wheels and fast.
export UV_PYTHON="${UV_PYTHON:-3.12}"

# --- caches off $HOME ------------------------------------------------------
# Per-run venv (not a shared dir): a single shared venv couples independent
# runs — one run's `uv pip install` can corrupt the packages another run is
# importing. One venv per WORK_ROOT keeps them isolated. It's all wheels, so
# rebuilding is ~25s; override UV_PROJECT_ENVIRONMENT to reuse one elsewhere.
# Sherlock's rule puts Python environments in $GROUP_HOME, not on $SCRATCH:
# $GROUP_HOME/venvs/auto-psych_<WORK_ROOT's name> when GROUP_HOME is set.
if [[ -z "${UV_PROJECT_ENVIRONMENT:-}" && -n "${GROUP_HOME:-}" ]]; then
  export UV_PROJECT_ENVIRONMENT="$GROUP_HOME/venvs/auto-psych_$(basename "$WORK_ROOT")"
fi
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}"
# uv's standalone Python looks for /etc/ssl/cert.pem, which el7 lacks: every
# HTTPS fetch failed with CERTIFICATE_VERIFY_FAILED (RSA Sherlock run 1).
# Verification stays on; only the bundle's path is given.
if [[ -z "${SSL_CERT_FILE:-}" && -f /etc/pki/tls/certs/ca-bundle.crt ]]; then
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
fi
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
