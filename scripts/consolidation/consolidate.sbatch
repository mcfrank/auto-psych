#!/bin/bash
#SBATCH --job-name=consolidate_2026_09
#SBATCH --partition=normal
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#SBATCH --mail-type=END,FAIL,TIMEOUT
#
# The consolidation job: prepares the integration clone once (idempotent),
# builds the venv once, then hands over to run_consolidation.py, which runs
# one Claude Code session per plan phase and requeues this script when it has
# to wait (session limit, walltime, smoke jobs). Submitted by consolidate.sh;
# resubmitted by the driver with the same --export.
#
# Env (via --export): WORK_ROOT. Everything else is read from
# $WORK_ROOT/consolidation.env (written by consolidate.sh).

set -euo pipefail
: "${WORK_ROOT:?WORK_ROOT must be set}"
[[ -f "$WORK_ROOT/consolidation.env" ]] || { echo "FATAL: no consolidation.env under $WORK_ROOT" >&2; exit 1; }
set -a; source "$WORK_ROOT/consolidation.env"; set +a
echo "[consolidate] work root $WORK_ROOT"
echo "[consolidate] source repo $SOURCE_REPO @ $MAIN_SHA"
cd "$SOURCE_REPO"

# --- modules (mirror scripts/recovery_improvement/review_iteration.sbatch) ---
ml purge
ml load devel
ml load gcc/14.2.0 2>/dev/null || ml load gcc 2>/dev/null || true
ml load system uv 2>/dev/null || ml load uv
# A modern git for the merges (system git is 1.8: no worktree, no -C).
ml load system git
git --version

# --- Claude Code -------------------------------------------------------------
# Auth is ~/.claude/.credentials.json over NFS (the user's subscription). Drop
# any API key --export=ALL carried in: it would silently switch billing.
unset ANTHROPIC_API_KEY ANTHROPIC_AUTH_TOKEN ANTHROPIC_BASE_URL \
      CLAUDE_CODE_USE_BEDROCK CLAUDE_CODE_USE_VERTEX CLAUDE_CODE_USE_FOUNDRY 2>/dev/null || true
[[ -f "$HOME/.claude/.credentials.json" ]] \
  || { echo "FATAL: ~/.claude/.credentials.json missing — claude is not logged in" >&2; exit 1; }
if [[ -n "${CLAUDE_BIN_DIR:-}" && -x "$CLAUDE_BIN_DIR/claude" ]]; then
  export PATH="$CLAUDE_BIN_DIR:$PATH"
else
  ml load claude-code
fi
command -v claude >/dev/null || { echo "FATAL: claude not on PATH" >&2; exit 1; }
echo "[consolidate] claude $(claude --version) at $(command -v claude); model $MODEL"

# --- caches off $HOME --------------------------------------------------------
export XDG_CACHE_HOME="$WORK_ROOT/.cache"
export UV_CACHE_DIR="${UV_CACHE_DIR_SHARED:-$WORK_ROOT/.uv_cache}"
export PIP_CACHE_DIR="$WORK_ROOT/.pip_cache"
export MPLCONFIGDIR="$WORK_ROOT/.cache/matplotlib"
export HF_HOME="$WORK_ROOT/.hf"
mkdir -p "$XDG_CACHE_HOME" "$UV_CACHE_DIR" "$PIP_CACHE_DIR" "$MPLCONFIGDIR" "$WORK_ROOT/slurm_logs"
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-$WORK_ROOT/.cache}/pytensor"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# --- venv (built once, reused by every requeue) -------------------------------
# Same el7 binary-only recipe as holdout_setup.sbatch / review_iteration.sbatch,
# plus pandas/pytest/h5netcdf so the agent can run the fast suite and read .nc
# fit caches.
VENV="$WORK_ROOT/venv"
export VENV_PY="$VENV/bin/python"
if [[ -x "$VENV_PY" ]] && "$VENV_PY" -c "import pymc, pandas, pytest, yaml, tyro, pyprojroot" >/dev/null 2>&1; then
  echo "[consolidate] reusing venv $VENV"
else
  echo "[consolidate] building venv $VENV"
  uv venv --clear --python "${UV_PYTHON:-3.12}" "$VENV"
  uv pip install --python "$VENV" --only-binary :all: \
    "pymc==5.28.5" "pytensor==2.38.3" "arviz<1" \
    numpy scipy pyyaml tyro pyprojroot matplotlib pandas pytest h5netcdf h5py
fi

# --- the integration clone (idempotent) ------------------------------------------
# Fresh clone of the user's checkout, then the campaign branches and arm C are
# fetched under refs/consolidation/* and every SHA is verified against the
# frozen inputs. The branch starts at iteration 3. A requeue finds the clone
# already on the branch and leaves it alone.
REPO="$WORK_ROOT/repo"
BRANCH="consolidate/2026-09"
verify_ref() {  # verify_ref <ref> <expected sha>
  local got; got="$(git -C "$REPO" rev-parse --verify "$1^{commit}")"
  [[ "$got" == "$2" ]] || { echo "FATAL: $1 is $got, expected $2 (frozen input changed)" >&2; exit 1; }
  echo "[consolidate] $1 = $got (ok)"
}
if [[ -d "$REPO/.git" ]]; then
  current="$(git -C "$REPO" rev-parse --abbrev-ref HEAD)"
  [[ "$current" == "$BRANCH" ]] || { echo "FATAL: $REPO is on branch '$current', not '$BRANCH'" >&2; exit 1; }
  echo "[consolidate] reusing clone $REPO on $BRANCH @ $(git -C "$REPO" rev-parse --short HEAD)"
else
  echo "[consolidate] cloning $SOURCE_REPO -> $REPO"
  git clone --quiet --no-checkout -- "$SOURCE_REPO" "$REPO"
  git -C "$REPO" update-ref refs/consolidation/main "$MAIN_SHA"
  git -C "$REPO" fetch --quiet -- "$ITER3_REPO" "$ITER3_BRANCH:refs/consolidation/iter3"
  git -C "$REPO" fetch --quiet -- "$ITER4_REPO" "$ITER4_BRANCH:refs/consolidation/iter4"
  git -C "$REPO" fetch --quiet -- "$ITER5_REPO" "$ITER5_BRANCH:refs/consolidation/iter5"
  git -C "$REPO" fetch --quiet -- "$SOURCE_REPO" "$ARMC_SOURCE_REF:refs/consolidation/armc"
  verify_ref refs/consolidation/main "$MAIN_SHA"
  verify_ref refs/consolidation/iter3 "$ITER3_SHA"
  verify_ref refs/consolidation/iter4 "$ITER4_SHA"
  verify_ref refs/consolidation/iter5 "$ITER5_SHA"
  verify_ref refs/consolidation/armc "$ARMC_SHA"
  git -C "$REPO" checkout --quiet -b "$BRANCH" "$ITER3_SHA"
  git -C "$REPO" config user.name "$GIT_USER_NAME"
  git -C "$REPO" config user.email "$GIT_USER_EMAIL"
  echo "[consolidate] branch $BRANCH created at $(git -C "$REPO" rev-parse --short HEAD)"
fi
# The smoke chains (P7) read the coding-agent key from the clone's .secrets
# (gitignored, so the tree stays clean). Copy, never symlink, so the agent's
# repo copies rsynced by the array carry it too.
if [[ ! -f "$REPO/.secrets" && -f "$SOURCE_REPO/.secrets" ]]; then
  install -m 600 "$SOURCE_REPO/.secrets" "$REPO/.secrets"
fi
# The leakage patch is a fallback only; check it is what the plan says it is.
echo "$LEAKAGE_PATCH_SHA256  $LEAKAGE_PATCH" | sha256sum --check --quiet \
  || { echo "FATAL: leakage patch checksum mismatch: $LEAKAGE_PATCH" >&2; exit 1; }

exec "$VENV_PY" scripts/consolidation/run_consolidation.py --work-root "$WORK_ROOT"
