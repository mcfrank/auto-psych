#!/bin/bash
# Submit the impossible-theory holdout sweep: the literature sweep's setup ->
# array -> retry -> summary chain (submit_holdout_test_retest.sh) with the
# impossible config and ground truths. The ground truths (e.g. "more heads =>
# more random") live in src/subjective_randomness/impossible_models, outside the
# seed pool: the setup job stages its pristine GT snapshot from there
# (GT_MODELS_REL), and the loop keeps the full normal seed pool, so it is
# expected to FAIL to recover them. Every knob of the shared script still
# applies, e.g.
#   SMOKE=1 bash scripts/subjective_randomness/slurm/submit_impossible_holdout_test_retest.sh
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export GT_MODELS="${GT_MODELS:-more_heads_more_random fewer_heads_more_random longer_runs_more_random more_imbalance_more_random}"
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/impossible_holdout_recovery.yaml}"
export GT_MODELS_REL="${GT_MODELS_REL:-src/subjective_randomness/impossible_models}"
[[ "${CLUSTER:-sherlock}" == coco ]] && export SCRATCH="${SCRATCH:-/scr/$USER}"  # as in _env.sh
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/impossible_holdout_test_retest}"
exec bash "$here/submit_holdout_test_retest.sh"
