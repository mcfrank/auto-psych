#!/bin/bash
# Score one dataset run's seeds and experiment winners on its unseen items, on the run's own node
# (its fit cache is there), with the code and venv the run ran on.
#   score_run.sh <run> [model.py ...]   (default models: the three seeds and each experiment's end winner)
set -euo pipefail
run=$1; shift
REPO=/sailhome/wurgaft/auto-psych
D=$REPO/data/results/number_game_dataset-nov26
W=$(ls -d /scr/wurgaft/auto-psych/number_game_dataset-nov26-$(hostname -s))
C=$D/collected/$run/number_game
if (( $# )); then models=("$@"); else
  models=()
  for s in tenenbaum_rules_similarity bigelow_piantadosi_mixture bigelow_piantadosi_independent; do
    models+=("$REPO/src/pipelines/outer_loop/projects/number_game/seed_models/$s.py"); done
  for e in 1 2 3; do
    best=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['best_model'])" "$C/experiment$e/model_loop/model_posterior.json")
    models+=("$C/experiment$e/cognitive_models/$best.py"); done
fi
cd "$W/harness_repo"
# Node-local PyTensor compile dir: the default one on AFS home fails its file lock (stale handle).
export PYTENSOR_FLAGS="base_compiledir=$W/pytensor_score" TMPDIR="$W/tmp_score"
mkdir -p "$TMPDIR"
PYTHONPATH=$W/harness_repo "$W/venv/bin/python" "$D/score_unseen_items.py" \
  --train "$C/experiment3/model_loop/responses.csv" --cache "$W/$run/dataset/mcmc_cache" \
  --out-dir "${OUT_DIR:-$D/unseen/$run}" --models "${models[@]}"
