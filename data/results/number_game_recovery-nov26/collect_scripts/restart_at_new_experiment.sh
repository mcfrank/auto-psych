#!/bin/bash
# Cancel each nov26b cell still running code from before its work root's patches
# (5-hour usage limit, HDF5 lock off, per-round checkpoints) as soon as it begins a
# new experiment, and resubmit it as a retry on the patched harness_repo (resume
# skips the finished experiments). A cell already at round 0 of an experiment k >= 2
# when first seen is restarted at once. One line per cell below:
#   <array task job id> <node> <campaign> <cell rel dir> <array task> <srun partition> <knobs>
# If the array's own retry job also resumes a cell, the cell lock lets one run.
#SBATCH --job-name=nov26b_restart_watch
set -uo pipefail
D=/sailhome/wurgaft/auto-psych-runs/watchers
JAG="REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26-loprio COCO_PARTITION=sc-loprio GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=4 CELL_CPUS=24 CELL_MEM=60G"
CELLS=(
  "18189395_1|cocoflops2|c2|run1/bigelow_piantadosi_independent|1|cocoflops-interactive|REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26 GT_MODELS=bigelow_piantadosi_independent MAX_PARALLEL=3 CELL_CPUS=30 CELL_MEM=160G"
  "18196768_5|jagupard28|jag28|run3/tenenbaum_rules_similarity|5|sc-loprio|$JAG"
  "18196768_6|jagupard28|jag28|run3/bigelow_piantadosi_mixture|6|sc-loprio|$JAG"
)
log() { echo "$(date '+%F %T') $*"; }
unset $(env | grep -o '^SLURM_[A-Z_]*')
on_node() {  # on_node <array task id> <node> <cmd...>: run on the cell's node, inside its own job
  local job=$1 node=$2; shift 2
  if [[ "$(hostname -s)" == "$node" ]]; then "$@"
  else timeout 300 srun --overlap --jobid="$(squeue -h -j "$job" -o %A)" -n1 --cpu-bind=none --mem=0 "$@"; fi
}
declare -A done_cells=() base=()
while (( ${#done_cells[@]} < ${#CELLS[@]} )); do
  for c in "${CELLS[@]}"; do
    IFS='|' read -r job node camp rel task part knobs <<< "$c"
    [[ -n "${done_cells[$job]:-}" ]] && continue
    W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-$camp
    if [[ -z "$(squeue -h -j "$job" -o %T 2>/dev/null)" ]]; then
      log "$camp task $task ($job) left the queue on its own; its retry job resumes it"; done_cells[$job]=1; continue
    fi
    out=$(on_node "$job" "$node" bash "$D/probe_experiment.sh" "$W" "$rel" 2>&1 | tail -1)
    if [[ "$out" != STATE* ]]; then log "$camp task $task ($job): PROBE ERROR: $out"; continue; fi
    read -r _ n prev iters <<< "$out"
    if [[ -z "${base[$job]:-}" ]]; then
      base[$job]=$n; log "$camp task $task ($job): first seen in experiment $n (prev valid=$prev, $iters round(s))"
      (( n >= 2 && prev == 1 && iters == 0 )) || continue
    elif (( n <= base[$job] || prev != 1 )); then continue
    fi
    log "$camp task $task ($job): experiment $n begun, $iters round(s) -> scancel"
    scancel "$job"
    while [[ -n "$(squeue -h -j "$job" -o %T 2>/dev/null)" ]]; do sleep 10; done
    resub="cd $W && env WORK_ROOT=$W COCO_NODE=$node CLAUDE_AUTH=subscription N_REPEATS=3 RETRY_ROUND=1 ARRAY_TASKS=$task $knobs bash $W/harness_repo/scripts/number_game/slurm/submit_recovery.sh"
    if [[ "$(hostname -s)" == "$node" ]]; then r=$(bash -c "$resub" 2>&1)
    else r=$(timeout 3600 srun -A cocoflops -p "$part" -w "$node" -n1 -c1 --mem=1G -t 00:15:00 bash -c "$resub" 2>&1); fi
    log "$camp task $task resubmitted: $(echo "$r" | grep -E 'submitted|ERROR' | tr '\n' ';')"
    done_cells[$job]=1
  done
  sleep 120
done
log "all cells handled"
