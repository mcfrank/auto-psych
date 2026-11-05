#!/bin/bash
# Stage a nov26b work root's results (no MCMC caches) to shared home.
#   stage.sh <work_root_suffix> <dest_dir>
set -euo pipefail
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-$1
D=$2/$1/$(hostname -s)
mkdir -p "$D"
cd "$W"
tar -cf - --exclude='*/mcmc_cache' --exclude=venv --exclude=harness_repo \
  --exclude=agent_src --exclude=gt_models_src --exclude='run*/*/repo' . \
  | tar -xf - -C "$D"
# Cells without holdout.json: snapshot their live agent tree for diagnosis.
for cell in run*/*/; do
  cell=${cell%/}
  [[ -f "$cell/holdout.json" ]] && continue
  id=$(cat "$cell/agent_tree_id")
  T=/scr/wurgaft/auto-psych/agent_trees/$id
  [[ -d "$T" ]] || continue
  mkdir -p "$D/$cell/diag"
  ls -la "$T/mcmc_cache" > "$D/$cell/diag/mcmc_cache_ls.txt" 2>&1 || true
  (cd "$T/repo/_runs" && find . -path '*/.cv_folds' -prune -o -print | sort) \
    > "$D/$cell/diag/runs_tree.txt"
  (cd "$T/repo" && tar -cf - --exclude='*.nc' --exclude='.cv_folds' _runs) \
    | tar -xf - -C "$D/$cell/diag"
done
echo "staged $W -> $D"
du -sh "$D"
