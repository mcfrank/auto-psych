#!/bin/bash
# Patch jag28's staged code, file the failed cell's experiment-1 winner's fit
# under its export name, then resubmit the cell: it resumes at experiment 2's
# design.
#SBATCH --job-name=rerun_jag28_run3
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-jag28
bash $S/prep_jag28_run3_mixture.sh
unset $(env | grep -o '^SLURM_[A-Z_]*')
cd $W && env REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26-loprio WORK_ROOT=$W COCO_NODE=jagupard28 \
  COCO_PARTITION=sc-loprio CLAUDE_AUTH=subscription N_REPEATS=3 RETRY_ROUND=1 ARRAY_TASKS=6 \
  GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=4 CELL_CPUS=24 CELL_MEM=60G \
  bash $W/harness_repo/scripts/number_game/slurm/submit_recovery.sh
