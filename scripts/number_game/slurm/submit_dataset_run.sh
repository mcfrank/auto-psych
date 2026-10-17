#!/bin/bash
# Submit a number-game dataset run: the live pipeline's stages with every
# experiment collected from the Bigelow & Piantadosi dataset, no ground truth
# (scripts/number_game/configs/dataset_run.yaml). One job per repeat, through
# the same job chain and settings as submit_recovery.sh, e.g.
#   CELL_CPUS=64 CELL_MEM=128G CLAUDE_AUTH=subscription_then_api \
#     bash scripts/number_game/slurm/submit_dataset_run.sh
set -euo pipefail
export CONFIG="${CONFIG:-scripts/number_game/configs/dataset_run.yaml}"
export GT_MODELS=dataset  # the job chain's name for its one cell: no model is held out
export N_REPEATS="${N_REPEATS:-1}"
export TEST_RETEST_ANALYSIS=0  # no ground truth, so no recovery to summarise
export WORK_ROOT="${WORK_ROOT:-/scr/$USER/auto-psych/number_game_dataset}"
exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/submit_recovery.sh"
