#!/bin/bash
#SBATCH --job-name=recovery_review
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# One review iteration of a recovery-improvement campaign (or its finalize
# step). Submitted by review.sh. With AUTO_LAUNCH=1 each review job also
# launches the declared sweep and submits the next review `afterany` its
# analysis job; by default it stops after writing its prescription and the
# user launches the sweep with launch_next.sh.
#
# Env (set via --export): CAMPAIGN_ROOT, ITERATION, MODE (review|finalize).
# Everything else comes from $CAMPAIGN_ROOT/campaign.env.
#
# What runs here: a Claude Code session (headless, `-p`) that reads the sweep
# digest, edits the pipeline in its own git clone, writes a prescription and a
# next_run.env; then run_iteration.py validates that and records the sweep
# (launching it only with AUTO_LAUNCH=1). Claude's stream is summarised into
# this log.

set -euo pipefail
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT must be set}"
: "${ITERATION:?ITERATION must be set}"
MODE="${MODE:-review}"

if [[ -f "$CAMPAIGN_ROOT/STOP" ]]; then
  echo "[review] $CAMPAIGN_ROOT/STOP present — campaign stopped by the user; exiting."
  exit 0
fi
[[ -f "$CAMPAIGN_ROOT/campaign.env" ]] || { echo "FATAL: no campaign.env under $CAMPAIGN_ROOT" >&2; exit 1; }
set -a; source "$CAMPAIGN_ROOT/campaign.env"; set +a
echo "[review] campaign=$CAMPAIGN_NAME iteration=$ITERATION mode=$MODE"
echo "[review] source repo=$SOURCE_REPO base branch=$BASE_BRANCH"
cd "$SOURCE_REPO"

# --- modules (mirror scripts/subjective_randomness/slurm/_env.sh) ------------
ml purge
ml load devel
# pytensor JIT-compiles PyMC models at run time; el7's gcc 4.8.5 can't.
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
ml load system uv 2>/dev/null || ml load uv
# A modern git for the agent (system git is 1.8: no worktree, no -C).
ml load system git 2>/dev/null || true

# --- Claude Code -------------------------------------------------------------
# review.sh pins the claude binary it found on the user's PATH; fall
# back to the cluster module. Auth is ~/.claude/.credentials.json over NFS —
# i.e. the user's Claude subscription. An API key in the environment would
# silently switch billing to pay-per-token API usage, so drop any that
# --export=ALL carried in from the submitting shell.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL \
      CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY 2>/dev/null || true
[[ -f "$HOME/.claude/.credentials.json" ]] \
  || { echo "FATAL: ~/.claude/.credentials.json missing — claude is not logged in (run 'claude' once on the login node)" >&2; exit 1; }
if [[ -n "${CLAUDE_BIN_DIR:-}" && -x "$CLAUDE_BIN_DIR/claude" ]]; then
  export PATH="$CLAUDE_BIN_DIR:$PATH"
else
  ml load claude-code
fi
command -v claude >/dev/null || { echo "FATAL: claude not on PATH" >&2; exit 1; }
echo "[review] claude $(claude --version) at $(command -v claude)"

# --- caches off $HOME --------------------------------------------------------
export XDG_CACHE_HOME="$CAMPAIGN_ROOT/.cache"
export UV_CACHE_DIR="$CAMPAIGN_ROOT/.uv_cache"
export PIP_CACHE_DIR="$CAMPAIGN_ROOT/.pip_cache"
export MPLCONFIGDIR="$CAMPAIGN_ROOT/.cache/matplotlib"
export HF_HOME="$CAMPAIGN_ROOT/.hf"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-$CAMPAIGN_ROOT/.cache}/pytensor"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- venv (campaign-level, built once, reused by every iteration) -----------
# Same el7 binary-only recipe as holdout_setup.sbatch, plus pandas/pytest/
# h5netcdf so the agent can run the fast test suite and read .nc fit caches.
VENV="$CAMPAIGN_ROOT/venv"
export VENV_PY="$VENV/bin/python"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import pymc, pandas, pytest, yaml, tyro, pyprojroot" >/dev/null 2>&1; then
  echo "[review] reusing venv $VENV"
else
  echo "[review] building venv $VENV"
  uv venv --clear --python "${UV_PYTHON:-3.12}" "$VENV"
  uv pip install --python "$VENV" --only-binary :all: \
    "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
    numpy scipy pyyaml tyro pyprojroot matplotlib pandas pytest h5netcdf h5py
fi

# AUTO_LAUNCH=1 in campaign.env opts into launching the declared sweep and
# chaining the next review; the default leaves that to the user (launch_next.sh).
AUTO_FLAG=""
if [[ "${AUTO_LAUNCH:-0}" == "1" ]]; then AUTO_FLAG="--auto-launch"; fi
exec "$VENV_PY" scripts/recovery_improvement/run_iteration.py \
  --campaign-root "$CAMPAIGN_ROOT" --iteration "$ITERATION" --mode "$MODE" $AUTO_FLAG
