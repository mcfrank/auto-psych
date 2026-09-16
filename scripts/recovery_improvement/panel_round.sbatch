#!/bin/bash
#SBATCH --job-name=panel_round
#SBATCH --partition=normal
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# One stage of a review panel (a discussion round, or the synthesis). Submitted
# by panel.sh as a chain (round1 -> round2 -> ... -> synthesis, afterany), and
# by panel_round.py when a stage requeues itself (inputs not ready yet, or a
# member hit its subscription's session limit).
#
# Env (set via --export): PANEL_ROOT, STAGE. Everything else is in panel.env.

set -euo pipefail
: "${PANEL_ROOT:?PANEL_ROOT must be set}"
: "${STAGE:?STAGE must be set}"
[[ -f "$PANEL_ROOT/panel.env" ]] || { echo "FATAL: no panel.env under $PANEL_ROOT" >&2; exit 1; }
if [[ -f "$PANEL_ROOT/STOP" ]]; then echo "[panel] STOP present; exiting"; exit 0; fi
set -a; source "$PANEL_ROOT/panel.env"; set +a
echo "[panel] panel=$PANEL_NAME stage=$STAGE"
cd "$SOURCE_REPO"

# --- modules -------------------------------------------------------------------
ml purge
ml load devel
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
ml load system uv 2>/dev/null || ml load uv
ml load system git 2>/dev/null || true

# --- agents: subscription logins only ------------------------------------------
# Claude: ~/.claude/.credentials.json. Codex: ~/.codex/auth.json (codex login
# --device-auth). An API key in the environment would switch either to
# pay-per-token billing, so drop any that --export=ALL carried in.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL OPENAI_API_KEY \
      CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY 2>/dev/null || true
if [[ "$MEMBERS" == *":claude:"* ]]; then
  if [[ -n "${CLAUDE_BIN_DIR:-}" && -x "$CLAUDE_BIN_DIR/claude" ]]; then export PATH="$CLAUDE_BIN_DIR:$PATH"; else ml load claude-code; fi
  command -v claude >/dev/null || { echo "FATAL: claude not on PATH" >&2; exit 1; }
  [[ -f "$HOME/.claude/.credentials.json" ]] || { echo "FATAL: claude is not logged in (~/.claude/.credentials.json missing)" >&2; exit 1; }
  echo "[panel] claude $(claude --version) at $(command -v claude)"
fi
if [[ "$MEMBERS" == *":codex:"* ]]; then
  ml load "${CODEX_MODULE:-codex/0.151.0}"
  command -v codex >/dev/null || { echo "FATAL: codex not on PATH after loading ${CODEX_MODULE:-codex/0.151.0}" >&2; exit 1; }
  [[ -f "${CODEX_HOME:-$HOME/.codex}/auth.json" ]] || { echo "FATAL: codex is not logged in (run: ml load devel ${CODEX_MODULE:-codex/0.151.0} && codex login --device-auth)" >&2; exit 1; }
  echo "[panel] $(codex --version) at $(command -v codex)"
fi
if [[ "$MEMBERS" == *":opencode:"* ]]; then
  ml load opencode 2>/dev/null || true
  command -v opencode >/dev/null || { echo "FATAL: opencode not on PATH" >&2; exit 1; }
fi

# --- caches off $HOME ------------------------------------------------------------
export XDG_CACHE_HOME="$PANEL_ROOT/.cache" UV_CACHE_DIR="$PANEL_ROOT/.uv_cache" PIP_CACHE_DIR="$PANEL_ROOT/.pip_cache"
export MPLCONFIGDIR="$PANEL_ROOT/.cache/matplotlib" HF_HOME="$PANEL_ROOT/.hf"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$PANEL_ROOT/slurm_logs"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-$PANEL_ROOT/.cache}/pytensor"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- venv (same recipe as the review job; reuse a campaign's if given) ---------
VENV="${PANEL_VENV:-$PANEL_ROOT/venv}"
export VENV_PY="$VENV/bin/python"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import pymc, pandas, pytest, yaml, tyro, pyprojroot" >/dev/null 2>&1; then
  echo "[panel] reusing venv $VENV"
else
  echo "[panel] building venv $VENV"
  uv venv --clear --python "${UV_PYTHON:-3.12}" "$VENV"
  uv pip install --python "$VENV" --only-binary :all: \
    "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
    numpy scipy pyyaml tyro pyprojroot matplotlib pandas pytest h5netcdf h5py
fi

exec "$VENV_PY" scripts/recovery_improvement/panel_round.py --panel-root "$PANEL_ROOT" --stage "$STAGE"
