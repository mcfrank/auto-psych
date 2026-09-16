#!/bin/bash
# Launch the raw-features arm (arm C): the agents receive only the H/T
# sequences, so every model must compute the features it uses. Paired with the
# featurized baseline so the difference isolates what the featurizer was
# contributing. See docs/raw_features_arm.md for the why and the caveats.
#
# Usage:
#   bash scripts/subjective_randomness/slurm/run_raw_features_arm.sh          # 3 x 4 = 12 tasks
#   SMOKE=1 bash scripts/subjective_randomness/slurm/run_raw_features_arm.sh  # one cheap task first
set -euo pipefail
SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Three replicates per ground truth; BASE_SEED=100 keeps repeats 1-3 on the same
# seeds (101-103) the featurized baseline's run1-run3 used, so cells pair.
export N_REPEATS="${N_REPEATS:-3}"
export BASE_SEED="${BASE_SEED:-100}"
export GT_MODELS="${GT_MODELS:-falk_konold_dp motif_stack finite_experience_occurrence local_representativeness}"
export CONFIG="${CONFIG:-scripts/subjective_randomness/configs/holdout_recovery_raw_features.yaml}"
# Both the GT/baseline registry and the live seed pool must be the RAW sets:
# their models compute their own columns. The array scrubs the held-out model
# from whichever dirs these name.
export SEED_MODELS_REL="${SEED_MODELS_REL:-src/subjective_randomness/pymc_model_families_raw}"
export POOL_MODELS_REL="${POOL_MODELS_REL:-src/pipelines/outer_loop/projects/subjective_randomness/seed_models_raw}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/holdout_raw_features}"
export MAX_PARALLEL="${MAX_PARALLEL:-4}"

echo ">>> raw-features arm (no harness-provided feature columns)"
echo "    N_REPEATS=$N_REPEATS  BASE_SEED=$BASE_SEED  GT_MODELS=$GT_MODELS"
echo "    CONFIG=$CONFIG"
echo "    SEED_MODELS_REL=$SEED_MODELS_REL"
echo "    POOL_MODELS_REL=$POOL_MODELS_REL"
echo "    WORK_ROOT=$WORK_ROOT"
[[ -n "${SMOKE:-}" ]] && echo "    SMOKE MODE (one cheap task)"
echo
exec bash "$SLURM_DIR/submit_holdout_test_retest.sh"
