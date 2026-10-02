#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
# Re-prune run 1's experiment 1 at 4·dse (2026-09-30), after the identity check
# (job 46161164: at 2.0 every decision and record reproduced; model_posterior.json
# differed only in max_pareto_k's 16th digit). First at 4.0 on that check's copy,
# then: back up experiment 1's mutable files (a real copy; the fit cache is never
# rewritten), move the failed experiment2/ aside, re-prune the real experiment 1.
set -euo pipefail
: "${EXPECTED_COMMIT:?}" "${COPY:?the copy of experiment1 made by the identity check}"
source "$HOME/repos/live_env.sh"
source "$OUTER_LIVE_SLURM_DIR/_env.sh"
die() { echo "STOP: $*" >&2; exit 1; }
E="$WORK_ROOT/run1/data/subjective_randomness"
FIT=(--draws 3000 --tune 2000 --chains 4 --target-accept 0.8)   # run 1's fit_kwargs
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git status --porcelain)" ]] \
  || die "checkout is not a clean $EXPECTED_COMMIT"

echo "=== 1. at 4.0 on the copy $COPY"
"$VENV_PY" -m src.pipelines.outer_loop.reprune --experiment-dir "$COPY" \
  --dse-multiplier 4 "${FIT[@]}" --note "check on a copy"
echo "[copy] carries:"; grep "name:" "$COPY/cognitive_models/models_manifest.yaml"

echo "=== 2. back up experiment 1 and move the failed experiment2/ aside"
B="$WORK_ROOT/run1/failed_attempts/experiment1-before-reprune-20260930"
[[ ! -e "$B" ]] || die "$B exists"
mkdir -p "$WORK_ROOT/run1/failed_attempts"
rsync -a --exclude 'model_loop/.fit_cache' "$E/experiment1/" "$B/"
echo "[backup] $B (without the fit cache)"
if [[ -e "$E/experiment2" ]]; then
  [[ ! -e "$E/experiment2/deployment" && -z "$(ls -A "$E/experiment2/data")" ]] \
    || die "$E/experiment2 has a deployment or data: not moving it"
  mv "$E/experiment2" "$WORK_ROOT/run1/failed_attempts/experiment2-design-failed-20260930T1733"
  echo "[apply] moved the failed experiment2/ to $WORK_ROOT/run1/failed_attempts/"
fi

echo "=== 3. at 4.0 on run 1's experiment 1"
"$VENV_PY" -m src.pipelines.outer_loop.reprune --experiment-dir "$E/experiment1" \
  --dse-multiplier 4 "${FIT[@]}" \
  --note "user decision 2026-09-30: at 2 one model was carried; the series prunes at 4"
echo "[apply] experiment 1 now carries:"; grep "name:" "$E/experiment1/cognitive_models/models_manifest.yaml"
"$VENV_PY" -c 'import sys, pathlib; from src.pipelines.outer_loop.orchestrator_validators import validate_cc_output as v; ok, m = v("5_model_loop", pathlib.Path(sys.argv[1])); print("[apply] stage validates:", ok, m); sys.exit(0 if ok else 1)' "$E/experiment1"
