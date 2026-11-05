#!/bin/bash
# Launch one number-game dataset run (repeat $1 = seed $1) on this node, in its own node-local
# WORK_ROOT; run it on the node itself (srun -w <node>). Billing: subscription, as dataset_run.yaml states.
#   launch_dataset_nov26.sh <repeat> <partition> <cell_mem>
set -euo pipefail
r=$1 partition=$2 mem=$3
node=$(hostname -s)
export PATH=$HOME/.local/share/uv-shim:$HOME/.local/bin:$PATH
W=/scr/wurgaft/auto-psych/number_game_dataset-nov26-$node
[[ ! -e $W ]] || { echo "ERROR: $W exists" >&2; exit 1; }
mkdir -p "$W"
cd "$W"
env REPO=/sailhome/wurgaft/auto-psych WORK_ROOT=$W COCO_NODE=$node COCO_PARTITION=$partition \
  CLAUDE_AUTH=subscription N_REPEATS=3 ARRAY_TASKS=$r MAX_PARALLEL=1 CELL_CPUS=48 CELL_MEM=$mem \
  bash /sailhome/wurgaft/auto-psych/scripts/number_game/slurm/submit_dataset_run.sh
