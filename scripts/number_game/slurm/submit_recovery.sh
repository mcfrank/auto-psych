#!/bin/bash
# Submit number-game holdout recovery cells: the subjective-randomness sweep's
# setup -> array -> retry -> summary chain (submit_holdout_test_retest.sh)
# with this project's config, ground truths and seeds, on the SC cluster's
# cocoflops1 by default. Each cell holds out one of GT_MODELS; the loop starts
# from the other seeds. Every knob of the shared script still applies, e.g.
#   N_REPEATS=3 bash scripts/number_game/slurm/submit_recovery.sh
#   SMOKE=1 bash scripts/number_game/slurm/submit_recovery.sh   # one cheap cell
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER="${CLUSTER:-coco}"
export CONFIG="${CONFIG:-scripts/number_game/configs/holdout_recovery.yaml}"
export GT_MODELS="${GT_MODELS:-tenenbaum_rules_similarity bigelow_piantadosi_mixture bigelow_piantadosi_independent}"
export SEED_MODELS_REL=src/pipelines/outer_loop/projects/number_game/seed_models
export POOL_MODELS_REL="$SEED_MODELS_REL"
export AGENT_BACKEND="${AGENT_BACKEND:-claude}" CLAUDE_AUTH="${CLAUDE_AUTH:-subscription}"
export N_REPEATS="${N_REPEATS:-3}"
export WORK_ROOT="${WORK_ROOT:-/scr/$USER/auto-psych/number_game_recovery}"
exec bash "$here/../../subjective_randomness/slurm/submit_holdout_test_retest.sh"
