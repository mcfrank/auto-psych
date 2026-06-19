#!/bin/bash
# Submit ONE more holdout-recovery repeat — one run per seed/ground-truth model —
# into an existing test-retest study, then re-run the reliability analysis over
# all repeats. Reuses the full pipeline (submit_holdout_test_retest.sh) so it
# inherits every fix and per-task isolation; this just targets the next repeat
# index via that script's ARRAY_TASKS knob.
#
# Each new task runs the full config (3 experiments, inner loop, full MCMC) with
# a fresh seed (= the repeat index), exactly like the existing repeats.
#
# Usage:
#   bash scripts/subjective_randomness/slurm/submit_extra_repeat.sh
#       -> adds the next repeat to $SCRATCH/auto-psych/holdout_full5
#   WORK_ROOT=$SCRATCH/auto-psych/holdout_full5 bash .../submit_extra_repeat.sh
#   REPEAT=7 bash .../submit_extra_repeat.sh      # force a specific repeat index
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_full5}"
# Must match the config's gt_models (the setup job validates this).
export GT_MODELS="${GT_MODELS:-bayesian_diagnosticity encoding_compressibility prototype_similarity statistical_inference}"
read -r -a _GTS <<< "$GT_MODELS"; NG="${#_GTS[@]}"

[[ -d "$WORK_ROOT" ]] || { echo "ERROR: WORK_ROOT '$WORK_ROOT' does not exist — point it at your existing study dir." >&2; exit 1; }

# Next repeat index = highest existing run<N> + 1 (override with REPEAT=).
if [[ -z "${REPEAT:-}" ]]; then
  last=0
  for d in "$WORK_ROOT"/run[0-9]*; do
    [[ -d "$d" ]] || continue
    n="${d##*/run}"
    [[ "$n" =~ ^[0-9]+$ ]] && (( n > last )) && last="$n"
  done
  REPEAT=$(( last + 1 ))
fi

# Map repeat R -> array tasks [(R-1)*G+1 .. R*G]; the array script derives
# repeat=(T-1)/G+1 and seed=BASE_SEED+repeat (BASE_SEED defaults to 0, so
# seed = REPEAT — distinct from the existing repeats' seeds).
FIRST=$(( (REPEAT - 1) * NG + 1 ))
LAST=$(( REPEAT * NG ))

echo ">>> adding repeat $REPEAT (seed $REPEAT) for $NG GT models"
echo "    array tasks $FIRST-$LAST  ->  output run$REPEAT/<gt> under $WORK_ROOT"
echo "    (analysis re-runs over ALL repeats once these finish)"
echo

ARRAY_TASKS="${FIRST}-${LAST}" WORK_ROOT="$WORK_ROOT" GT_MODELS="$GT_MODELS" \
  bash "$DIR/submit_holdout_test_retest.sh"
