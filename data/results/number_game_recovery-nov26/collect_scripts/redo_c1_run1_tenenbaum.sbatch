#!/bin/bash
# Redo Tenenbaum run1 from experiment 2's lost round 4 (2026-10-30 usage limit
# the old code did not recognise), so experiment 3 reruns on it too; then
# resubmit the cell as its sweep's retry.
#SBATCH --job-name=redo_c1_run1
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-c1
C=$W/run1/tenenbaum_rules_similarity
T=/scr/wurgaft/auto-psych/agent_trees/$(cat $C/agent_tree_id)
[[ ! -e $T ]] || { echo "ERROR: $T exists; expected the archived cell's tree to be gone" >&2; exit 1; }
[[ ! -e $C/repo ]] || { echo "ERROR: $C/repo exists" >&2; exit 1; }
bash $S/patch_harness.sh c1
mkdir -p $T/repo
tar -xzf $C/agent_runs.tar.gz -C $T/repo
ln -sfn $T/repo $C/repo
F=$C/superseded/$(date +%Y%m%dT%H%M%S)_finished_cell
mkdir -p $F
mv $C/holdout.json $C/holdout.csv $C/holdout.png $C/trajectory.json $C/agent_activity.md $C/agent_runs.tar.gz $F/
echo "the cell's finished records (17 of 18 steps: experiment 2 lost round 4), moved aside $(date -Iseconds) for the redo" > $F/README
cd $W/harness_repo
$W/venv/bin/python scripts/subjective_randomness/redo_abandoned_rounds.py --cell $C --experiment 2 --from-round 4
ls $T/repo/_runs/cell_1
unset $(env | grep -o '^SLURM_[A-Z_]*')
cd $W && env REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26 WORK_ROOT=$W COCO_NODE=cocoflops1 \
  CLAUDE_AUTH=subscription N_REPEATS=3 RETRY_ROUND=1 ARRAY_TASKS=1 \
  GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=6 CELL_CPUS=16 CELL_MEM=80G \
  bash $W/harness_repo/scripts/number_game/slurm/submit_recovery.sh
