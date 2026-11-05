#!/bin/bash
# inspect.sh <work root suffix> <cell>: read-only state of a cell and its staged code.
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-$1; C=$W/$2
echo "== harness_repo git status:"; (cd $W/harness_repo && git status --short | head -20 && git log --oneline -1)
echo "== agent_src vs harness_repo (src/models/pymc_inference.py, model_loop_runner.py):"
for f in src/models/pymc_inference.py src/pipelines/outer_loop/model_loop_runner.py src/pipelines/inner_loop/round_checkpoint.py; do cmp -s $W/agent_src/$f $W/harness_repo/$f && echo "same $f" || echo "DIFF $f"; done
ls $W/harness_repo/scripts/subjective_randomness/redo_abandoned_rounds.py
echo "== cell:"; ls -la $C
id=$(cat $C/agent_tree_id); T=/scr/wurgaft/auto-psych/agent_trees/$id
echo "== agent tree $T:"; ls $T 2>&1 | head; ls $T/repo/_runs 2>&1 | head -3
echo "== mcmc_cache: $(ls $C/mcmc_cache | wc -l) files, $(du -sh $C/mcmc_cache | cut -f1)"
ls $C/mcmc_cache | grep -E "iter4_candidate5|inner_loop_model" || true
E=$(ls -d $T/repo/_runs/*/experiment1 2>/dev/null) && [ -n "$E" ] && $W/venv/bin/python -c "import json,sys;print('exp1 history iterations', [h['iteration'] for h in json.load(open(sys.argv[1]))])" $E/model_loop/history.json
