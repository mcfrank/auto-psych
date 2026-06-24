#!/bin/bash
#SBATCH --job-name=reanalyze_exhaustive
#SBATCH --partition=normal
#SBATCH --time=08:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.out
#
# Stage 2/2: one array task per finished holdout.json listed in
# $REANALYZE_MANIFEST. Each task re-scores its run on the common EXHAUSTIVE eval
# pool (every distinct unordered pair over all sequences of REANALYZE_LENGTHS,
# default 1..8) IN PLACE — overwriting that run's holdout.{json,csv,png}. No
# agents, no MCMC resampling: every per-step fit is a hit on the run's own
# `.nc` cache, so the only real work is predicting held-out p_left over the
# ~130k-pair pool (predict_max_draws thins the posterior to keep that bounded).
#
# Submitted by submit_reanalyze_exhaustive.sh, which builds the manifest and
# sizes the array to it.

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

# Stray `uv run` from nothing here, but keep parity with the run array: target
# the explicit interpreter, never re-sync the shared venv mid-task.
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON

MANIFEST="${REANALYZE_MANIFEST:?set REANALYZE_MANIFEST to the holdout.json manifest path}"
TASK="${SLURM_ARRAY_TASK_ID:-1}"
RESULT_JSON="$(sed -n "${TASK}p" "$MANIFEST")"
[[ -n "$RESULT_JSON" ]] || { echo "ERROR: no line $TASK in $MANIFEST" >&2; exit 1; }
[[ -f "$RESULT_JSON" ]] || { echo "ERROR: result JSON missing: $RESULT_JSON" >&2; exit 1; }

# Per-task PyTensor compile dir on node-local SSD (pytensor JIT-compiles the
# predict graph; a private compiledir avoids NFS $HOME filelock contention).
PYTENSOR_BASE="${L_SCRATCH:-/tmp}/pytensor_reanalyze_${SLURM_ARRAY_JOB_ID:-0}_${TASK}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

echo "[task $TASK] re-scoring $RESULT_JSON on the exhaustive pool"
# --lengths takes several ints; REANALYZE_LENGTHS is a space-separated list, so
# it must stay unquoted for word-splitting. cache-dir / seed-models-dir / the
# (normal vs impossible) gt-models-dir are all resolved by the CLI from the
# result JSON and the repo, so the task body needs no per-study knobs.
"$VENV_PY" scripts/subjective_randomness/reanalyze_holdout_exhaustive.py \
  --result "$RESULT_JSON" \
  --lengths ${REANALYZE_LENGTHS:-1 2 3 4 5 6 7 8} \
  --predict-max-draws "${REANALYZE_PREDICT_MAX_DRAWS:-500}" \
  --min-remaining "${REANALYZE_MIN_REMAINING:-100}"

echo "[task $TASK] done -> $RESULT_JSON"
