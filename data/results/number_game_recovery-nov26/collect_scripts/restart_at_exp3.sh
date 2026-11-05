#!/bin/bash
# Cancel each nov26b cell that started on unpatched code once experiment 2's model
# loop has finished and experiment 3 has begun, and resubmit it as a retry on the
# patched harness_repo (resume skips experiments 1-2). One line per cell below:
#   <array task job id> <node> <campaign> <cell rel dir> <array task> <knobs>
#SBATCH --job-name=nov26b_restart_watch
set -uo pipefail
D=/sailhome/wurgaft/auto-psych-runs/watchers
REPO=/sailhome/wurgaft/auto-psych-runs/ng-recovery-nov26
CELLS=(
  "18194413_1|cocoflops1|c1|run1/tenenbaum_rules_similarity|1|GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=6 CELL_CPUS=16 CELL_MEM=80G"
  "18194413_2|cocoflops1|c1|run1/bigelow_piantadosi_mixture|2|GT_MODELS='tenenbaum_rules_similarity bigelow_piantadosi_mixture' MAX_PARALLEL=6 CELL_CPUS=16 CELL_MEM=80G"
  "18189395_1|cocoflops2|c2|run1/bigelow_piantadosi_independent|1|GT_MODELS=bigelow_piantadosi_independent MAX_PARALLEL=3 CELL_CPUS=30 CELL_MEM=160G"
)
log() { echo "$(date '+%F %T') $*"; }
unset $(env | grep -o '^SLURM_[A-Z_]*')
on_node() {  # on_node <job> <node> <cmd...>: run on the cell's node, inside its own job
  local job=$1 node=$2; shift 2
  if [[ "$(hostname -s)" == "$node" ]]; then "$@"; else timeout 300 srun --overlap --jobid="$job" -n1 --cpu-bind=none --mem=0 "$@"; fi
}
declare -A done_cells=()
while (( ${#done_cells[@]} < ${#CELLS[@]} )); do
  for c in "${CELLS[@]}"; do
    IFS='|' read -r job node camp rel task knobs <<< "$c"
    [[ -n "${done_cells[$job]:-}" ]] && continue
    W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-$camp
    if [[ -z "$(squeue -h -j "$job" -o %T 2>/dev/null)" ]]; then
      log "$camp task $task ($job) left the queue on its own; its retry job resumes it"; done_cells[$job]=1; continue
    fi
    out=$(on_node "$job" "$node" bash "$D/probe_cell.sh" "$W" "$rel" 2>&1 | tail -1)
    [[ "$out" == TRIGGER* ]] || continue
    log "$camp task $task ($job): $out -> scancel"
    scancel "$job"
    while [[ -n "$(squeue -h -j "$job" -o %T 2>/dev/null)" ]]; do sleep 10; done
    resub="cd $W && env REPO=$REPO WORK_ROOT=$W COCO_NODE=$node CLAUDE_AUTH=subscription N_REPEATS=3 RETRY_ROUND=1 ARRAY_TASKS=$task $knobs bash $W/harness_repo/scripts/number_game/slurm/submit_recovery.sh"
    if [[ "$(hostname -s)" == "$node" ]]; then r=$(bash -c "$resub" 2>&1)
    else r=$(timeout 3600 srun -A cocoflops -p cocoflops-interactive -w "$node" -n1 -c1 --mem=1G -t 00:15:00 bash -c "$resub" 2>&1); fi
    log "$camp task $task resubmitted: $(echo "$r" | grep -E 'submitted|ERROR' | tr '\n' ';')"
    done_cells[$job]=1
  done
  sleep 120
done
log "all cells handled"
