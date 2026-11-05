#!/bin/bash
# Stage this node's number_game_dataset-nov26 work root (no MCMC caches, venvs or code copies) to shared home.
#   stage.sh   (run on the node: srun -w <node>)
set -euo pipefail
W=$(ls -d /scr/wurgaft/auto-psych/number_game_dataset-nov26-$(hostname -s))
D=/sailhome/wurgaft/auto-psych-runs/ngds-nov26-collect/staged/$(hostname -s)
[[ ! -e $D ]] || { echo "ERROR: $D exists" >&2; exit 1; }
mkdir -p "$D"
cd "$W"
tar -cf - --exclude='*/mcmc_cache' --exclude=venv --exclude=harness_repo --exclude=agent_src \
  --exclude=gt_models_src --exclude=.uv_cache --exclude=.cache --exclude=.pip_cache . | tar -xf - -C "$D"
echo "staged $W -> $D"; du -sh "$D"
