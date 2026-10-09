#!/bin/bash
# Shared environment for the RSA run-1 Slurm jobs (prepare_data.sh,
# rsa_setup.sbatch, rsa_loop_array.sbatch). Sources the subjective-randomness
# _env.sh (modules: gcc, uv, opencode, bubblewrap; .secrets; the per-WORK_ROOT
# venv and caches; GOOGLE_GENERATIVE_AI_API_KEY; UV_PYTHON=3.12) with the RSA
# sweep's defaults, then checks what the RSA loop needs on top.
#
# Honors, if already exported:
#   REPO            the auto-psych checkout on branch auto-rsa ($HOME/auto-psych)
#   WORK_ROOT       the sweep root ($SCRATCH/auto-psych/$RSA_SWEEP_NAME, _cells.sh)
#   UV_PROJECT_ENVIRONMENT  the venv ($GROUP_HOME/venvs/auto-psych_<sweep>; rsa_default_venv)
#   SSL_CERT_FILE   CA bundle (el7's /etc/pki/tls/certs/ca-bundle.crt when it exists)
#   OPENCODE_BIN_DIR  bin/ of an npm-installed opencode-ai (see below; default
#                   $GROUP_HOME/software/npm-global/bin when it holds opencode)
#   NODEJS_MODULE   the nodejs module an npm-installed opencode runs with
set -euo pipefail

RSA_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$RSA_SLURM_DIR/_cells.sh"
export WORK_ROOT="${WORK_ROOT:-$(rsa_default_work_root)}"
[[ -n "$WORK_ROOT" ]] || exit 1
# Sherlock run 1 (2026-10-07) needed these exported by hand:
# - the venv in $GROUP_HOME (Sherlock's rule), not $WORK_ROOT/venv on $SCRATCH;
export UV_PROJECT_ENVIRONMENT="$(rsa_default_venv "$WORK_ROOT")"
# - uv's standalone Python looks for /etc/ssl/cert.pem, which el7 lacks, and
#   every HTTPS fetch failed (verification stays on; only the bundle's path);
if [[ -z "${SSL_CERT_FILE:-}" && -f /etc/pki/tls/certs/ca-bundle.crt ]]; then
  export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
fi
# - the opencode module is too old; the npm install in $GROUP_HOME is the standard one.
if [[ -z "${OPENCODE_BIN_DIR:-}" && -n "${GROUP_HOME:-}" && -x "$GROUP_HOME/software/npm-global/bin/opencode" ]]; then
  export OPENCODE_BIN_DIR="$GROUP_HOME/software/npm-global/bin"
fi
# Shared with subjective randomness: VENV_PY, caches off $HOME, modules,
# .secrets, BLAS threads.
source "$RSA_SLURM_DIR/../../subjective_randomness/slurm/_env.sh"
# JAX on one CPU per process (Research Computing, 2026-10-09): XLA's CPU
# runtime sizes its thread pool by the node's cores, not the job's allocation,
# and oversubscribed the cores. Fit children already ran single-threaded
# (src.rsa.loop.fitting.SINGLE_THREAD_XLA_FLAGS); now every process does
# (design, simulation, scoring). Parallelism comes from one process per CPU.
export XLA_FLAGS="${XLA_FLAGS:+$XLA_FLAGS }--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

# Everything the sweep reads or writes, outside every agent tree. Agents see
# only their own tree (built from agent_src) and the loop's results dir in it.
export DATA_ROOT="$WORK_ROOT/data"        # combined trials, splits, simulations, provenance
export CELLS_ROOT="$WORK_ROOT/cells"      # per-cell records, links, outputs
export HARNESS_REPO="$WORK_ROOT/harness_repo"
export AGENT_SRC="$WORK_ROOT/agent_src"
# Agent trees live under opaque ids, never under a condition-named path: every
# path in a tree reaches an agent's prompt.
export AGENT_TREES_ROOT="${AGENT_TREES_ROOT:-$(dirname "$WORK_ROOT")/agent_trees}"  # $SCRATCH/auto-psych/agent_trees by default

# --- opencode ------------------------------------------------------------------
# The RSA agents need opencode >= 1.18.35 (the shell tool's timeout parameter
# and "snapshot": false). If the module is older, install it once into
# $GROUP_HOME and point OPENCODE_BIN_DIR at its bin/:
#   ml load nodejs && npm install -g --prefix "$GROUP_HOME/software/npm-global" opencode-ai@1.18.35
#   export OPENCODE_BIN_DIR="$GROUP_HOME/software/npm-global/bin"
export OPENCODE_MIN_VERSION="${OPENCODE_MIN_VERSION:-1.18.35}"
if [[ -n "${OPENCODE_BIN_DIR:-}" ]]; then
  [[ -x "$OPENCODE_BIN_DIR/opencode" ]] || { echo "FATAL: OPENCODE_BIN_DIR=$OPENCODE_BIN_DIR has no opencode" >&2; exit 1; }
  # The npm package's launcher is a node script; node comes from a module
  # (mounted read-only into the sandbox with the rest of /share/software).
  ml load "${NODEJS_MODULE:-nodejs/25.3.0}"
  export PATH="$OPENCODE_BIN_DIR:$PATH"
fi

rsa_require_opencode() {
  command -v opencode >/dev/null || { echo "FATAL: no opencode on PATH (ml load opencode, or set OPENCODE_BIN_DIR)" >&2; exit 1; }
  local version
  version="$(opencode --version 2>&1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | head -1 || true)"
  [[ -n "$version" ]] || { echo "FATAL: '$(command -v opencode) --version' printed no version: the CLI cannot run here" >&2; exit 1; }
  if [[ "$(printf '%s\n%s\n' "$OPENCODE_MIN_VERSION" "$version" | sort -V | head -1)" != "$OPENCODE_MIN_VERSION" ]]; then
    echo "FATAL: opencode $version at $(command -v opencode) is older than $OPENCODE_MIN_VERSION; set OPENCODE_BIN_DIR (see _env.sh)" >&2
    exit 1
  fi
  echo "[rsa_env] opencode $version at $(command -v opencode)"
}

rsa_require_gemini_key() {
  [[ -n "${GOOGLE_API_KEY:-}" ]] || { echo "FATAL: GOOGLE_API_KEY is not set (add it to $REPO/.secrets)" >&2; exit 1; }
  echo "[rsa_env] GOOGLE_API_KEY is set"
}

echo "[rsa_env] WORK_ROOT=$WORK_ROOT DATA_ROOT=$DATA_ROOT AGENT_TREES_ROOT=$AGENT_TREES_ROOT"
