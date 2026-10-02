#!/bin/bash
#SBATCH --partition=normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64GB
#SBATCH --time=02:00:00
# ============================================================================
#  Re-prune full live run 1's experiment 1 at 4·dse_clustered (2026-09-30).
#
#    MODE=check  on a copy (fit cache hard-linked, read only): re-prune at 2.0
#                must leave it exactly as it was; then at 4.0, printing the set
#    MODE=apply  move the failed experiment2/ aside (it must hold no deployment
#                and no data), then re-prune the real experiment 1 at 4.0
#
#  Runs the checkout's code (EXPECTED_COMMIT, clean) with the live venv and the
#  loop's own sampler settings, so every fit comes from the cache.
# ============================================================================
set -euo pipefail
: "${MODE:?MODE=check or MODE=apply}" "${EXPECTED_COMMIT:?}"
source "$HOME/repos/live_env.sh"
source "$OUTER_LIVE_SLURM_DIR/_env.sh"
die() { echo "STOP: $*" >&2; exit 1; }

E="$WORK_ROOT/run1/data/subjective_randomness"
FIT=(--draws 3000 --tune 2000 --chains 4 --target-accept 0.8)   # run 1's fit_kwargs
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}"
cd "$REPO"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" && -z "$(git status --porcelain)" ]] \
  || die "checkout is not a clean $EXPECTED_COMMIT"

state() {  # the files a re-prune touches, by sha256
  local x="$1"
  (cd "$x" && sha256sum model_loop/models/models_manifest.yaml model_loop/attempted_hypotheses.jsonl \
     model_loop/history.json model_loop/export_complete.json model_loop/model_posterior.json \
     cognitive_models/* model_registry.yaml 2>/dev/null; \
   ls model_loop/models model_loop/models/pruned | sha256sum)
}

case "$MODE" in
  check)
    C="$SCRATCH/reprune_check/$(date +%Y%m%d_%H%M%S)/subjective_randomness"
    mkdir -p "$C"
    rsync -a --exclude 'model_loop/.fit_cache' "$E/experiment1/" "$C/experiment1/"
    cp -al "$E/experiment1/model_loop/.fit_cache" "$C/experiment1/model_loop/.fit_cache"
    cp "$E/starting_models.json" "$C/"
    before="$(state "$C/experiment1")"
    "$VENV_PY" -m src.pipelines.outer_loop.reprune --experiment-dir "$C/experiment1" \
      --dse-multiplier 2 "${FIT[@]}" --note "identity check on a copy"
    after="$(state "$C/experiment1")"
    if [[ "$before" == "$after" ]]; then
      echo "[check] re-pruning at 2.0 left the copy exactly as it was"
    else
      echo "[check] DIFFERENCES after re-pruning at 2.0:"; diff <(echo "$before") <(echo "$after") || true
      die "the identity check failed; nothing was changed in run 1"
    fi
    "$VENV_PY" -m src.pipelines.outer_loop.reprune --experiment-dir "$C/experiment1" \
      --dse-multiplier 4 "${FIT[@]}" --note "check on a copy"
    echo "[check] at 4.0 the copy carries:"; grep "name:" "$C/experiment1/cognitive_models/models_manifest.yaml"
    echo "[check] copy left at $C (its fit cache is hard links to run 1's)"
    ;;
  apply)
    if [[ -e "$E/experiment2" ]]; then
      [[ ! -e "$E/experiment2/deployment" && -z "$(ls -A "$E/experiment2/data")" ]] \
        || die "$E/experiment2 has a deployment or data: not moving it"
      mkdir -p "$WORK_ROOT/run1/failed_attempts"
      mv "$E/experiment2" "$WORK_ROOT/run1/failed_attempts/experiment2-design-failed-20260930T1733"
      echo "[apply] moved the failed experiment2/ to $WORK_ROOT/run1/failed_attempts/"
    fi
    "$VENV_PY" -m src.pipelines.outer_loop.reprune --experiment-dir "$E/experiment1" \
      --dse-multiplier 4 "${FIT[@]}" \
      --note "user decision 2026-09-30: at 2 one model was carried; the series prunes at 4"
    echo "[apply] experiment 1 now carries:"; grep "name:" "$E/experiment1/cognitive_models/models_manifest.yaml"
    ;;
  *) die "MODE must be check or apply" ;;
esac
