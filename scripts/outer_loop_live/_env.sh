#!/bin/bash
# Shared environment for the LIVE outer-loop Slurm harness — real Prolific
# participants, jsPsych experiment deployed to Firebase. Sourced by
# setup.sbatch, run_live.sbatch, and submit_parallel.sh.
#
# Honors these env vars if already exported (else uses the defaults):
#   REPO        - path to the auto-psych checkout (the main one)
#   WORK_ROOT   - where all run output / caches / venv live (must NOT be $HOME)
#   UV_PROJECT_ENVIRONMENT - shared venv location
set -euo pipefail

# --- repo + work locations -------------------------------------------------
export REPO="${REPO:-$HOME/auto-psych}"
# Heavy I/O and large files must stay off $HOME (15 GB, NFS-backed). Default to
# $SCRATCH; fall back to $GROUP_SCRATCH if $SCRATCH is unset.
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/outer_loop_live}"
mkdir -p "$WORK_ROOT"

# --- modules ---------------------------------------------------------------
ml purge 2>/dev/null || true
ml load devel 2>/dev/null || true
# gcc 14 is REQUIRED at RUN TIME: pytensor JIT-compiles every PyMC model with a
# C compiler (el7's system gcc is 4.8.5). The module also sets CC/CXX and puts
# the matching libstdc++ on LD_LIBRARY_PATH.
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
# nodejs: `firebase deploy`, the functions npm install, and `npx firebase-tools`.
# Pin Node 24: firebase-tools 15.x supports Node 20/22/24, NOT 25 (the bare
# `nodejs` module is 25.3.0 -> EBADENGINE). Fall back if 24 is unavailable.
ml load nodejs/24.13.0 2>/dev/null || ml load nodejs/24 2>/dev/null || ml load nodejs 2>/dev/null || true
# Coding-agent backends for theory/design/implement + inner loop. opencode is
# the default (opencode + Gemini); claude-code is available via coding_agent: claude.
ml load opencode 2>/dev/null || true
ml load claude-code 2>/dev/null || true
# uv drives the Python env.
ml load system uv 2>/dev/null || ml load uv 2>/dev/null || true

# --- python version --------------------------------------------------------
# Pin a STABLE Python with full el7 (glibc 2.17 / manylinux2014) wheel coverage.
# uv otherwise grabs the newest, for which few binary packages ship el7 wheels
# yet -> the scientific stack builds from source on el7 (slow / fails).
export UV_PYTHON="${UV_PYTHON:-3.12}"

# --- caches + state off $HOME ----------------------------------------------
# Shared venv (built once by setup.sbatch): no run does `uv pip install` at run
# time, so all runs can import from one venv. VENV_PY is an explicit interpreter
# path so the pipeline never reconciles a project lock via `uv run`.
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}"
export VENV_PY="$UV_PROJECT_ENVIRONMENT/bin/python"
export UV_CACHE_DIR="$WORK_ROOT/.uv_cache"
export UV_PYTHON_INSTALL_DIR="$WORK_ROOT/.uv_python"
export PIP_CACHE_DIR="$WORK_ROOT/.pip_cache"
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export HF_HOME="$WORK_ROOT/.hf"
export npm_config_cache="$WORK_ROOT/.npm"
mkdir -p "$(dirname "$UV_PROJECT_ENVIRONMENT")" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" \
         "$PIP_CACHE_DIR" "$XDG_CACHE_HOME" "$HF_HOME" "$npm_config_cache"

# --- secrets ---------------------------------------------------------------
# .secrets as a KEY=value file (or a directory, one file per key). Exports
# PROLIFIC_API_TOKEN, GOOGLE_API_KEY, FIREBASE_TOKEN when present.
if [[ -f "$REPO/.secrets" ]]; then
  while IFS='=' read -r k v || [[ -n "$k" ]]; do
    k="${k// /}"
    [[ -z "$k" || "$k" == \#* ]] && continue
    export "$k=$(echo "$v" | xargs)" || true
  done < "$REPO/.secrets"
elif [[ -d "$REPO/.secrets" ]]; then
  for f in "$REPO/.secrets"/*; do
    [[ -f "$f" ]] || continue
    export "$(basename "$f")=$(tr -d '\n' < "$f")" || true
  done
fi
# Gemini aliases — only used if a closed/Gemini participant or LLM browser
# steering path is exercised. A pure live HUMAN run does not need them.
export GEMINI_API_KEY="${GEMINI_API_KEY:-${GOOGLE_API_KEY:-}}"
export GOOGLE_GENERATIVE_AI_API_KEY="${GOOGLE_GENERATIVE_AI_API_KEY:-${GOOGLE_API_KEY:-}}"

# --- firebase deploy serialization -----------------------------------------
# Parallel runs share ONE Firebase site; this lockfile serializes only the brief
# `firebase deploy` step (run_firebase_deploy honors AUTO_PSYCH_DEPLOY_LOCK).
# Data, URLs, and Prolific studies are already isolated per run by --run-label.
export AUTO_PSYCH_DEPLOY_LOCK="${AUTO_PSYCH_DEPLOY_LOCK:-$WORK_ROOT/firebase-deploy.lock}"

# --- TLS / CA certificates -------------------------------------------------
# The uv-managed Python (python-build-standalone) does not find el7's system CA
# store on its own, so its stdlib `ssl` (urllib -> /results) and `requests`
# (Prolific API) fail with CERTIFICATE_VERIFY_FAILED. Point both at the system
# bundle (the same one curl trusts). Pick the first that exists.
for _ca in /etc/pki/tls/certs/ca-bundle.crt /etc/ssl/certs/ca-bundle.crt /etc/ssl/certs/ca-certificates.crt; do
  if [[ -f "$_ca" ]]; then
    export SSL_CERT_FILE="$_ca"
    export REQUESTS_CA_BUNDLE="$_ca"
    break
  fi
done
[[ -d /etc/pki/tls/certs ]] && export SSL_CERT_DIR="/etc/pki/tls/certs"

# --- threading -------------------------------------------------------------
# PyMC samples one chain per core; keep BLAS from oversubscribing within a chain.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

echo "[_env] REPO=$REPO"
echo "[_env] WORK_ROOT=$WORK_ROOT"
echo "[_env] UV_PROJECT_ENVIRONMENT=$UV_PROJECT_ENVIRONMENT"
echo "[_env] AUTO_PSYCH_DEPLOY_LOCK=$AUTO_PSYCH_DEPLOY_LOCK"
