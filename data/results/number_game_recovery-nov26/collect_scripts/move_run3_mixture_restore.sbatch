#!/bin/bash
# On tiger5: unpack the cell and agent tree packed on jagupard28, patch tiger5's staged harness
# with the export fit-alias fix, record the move, and resubmit task 6 here (it resumes from its
# last finished round, refitting what it needs).
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
D=/sailhome/wurgaft/auto-psych-runs/move_jag28_run3
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-jag28
A=/scr/wurgaft/auto-psych/agent_trees
id=f8fb9bba6e6fa0a9
[[ ! -e $W/run3/bigelow_piantadosi_mixture && ! -e $A/$id ]]
tar -C $W -xf $D/cell.tar
mkdir -p $A && tar -C $A -xf $D/tree.tar
mkdir -p $A/$id/mcmc_cache
bash $S/patch_harness.sh jag28
echo "a38c7f7b+export_fit_alias.src.patch was committed as e534c4c6 (branch export-files-renamed-fit), 2026-11-01." >> $W/code_patches/README
echo "$(date -Iseconds) moved run3/bigelow_piantadosi_mixture (task 6) from jagupard28 to $(hostname) without fit caches (preempted twice by jag-standard)" >> $W/code_patches/README
unset $(env | grep -o '^SLURM_[A-Z_]*')
cd $W && env REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26-loprio WORK_ROOT=$W COCO_NODE=tiger5 \
  COCO_PARTITION=sc-loprio CLAUDE_AUTH=subscription N_REPEATS=3 RETRY_ROUND=1 ARRAY_TASKS=6 \
  GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=4 CELL_CPUS=24 CELL_MEM=60G \
  bash $W/harness_repo/scripts/number_game/slurm/submit_recovery.sh
