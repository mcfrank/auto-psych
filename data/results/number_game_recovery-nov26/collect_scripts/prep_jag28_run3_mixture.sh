#!/bin/bash
# File the failed cell's experiment-1 winner's fit (iter4_candidate5) under its
# export name, as the fixed export would have, and check experiment 2's design
# can load it without sampling.
set -euo pipefail
S=/sailhome/wurgaft/auto-psych-runs/nov26b-collect
W=/scr/wurgaft/auto-psych/number_game_recovery-nov26b-jag28
C=$W/run3/bigelow_piantadosi_mixture
bash $S/patch_harness.sh jag28
cd $W/harness_repo
RUNS=$(readlink -f $C/repo)/_runs/cell_1
$W/venv/bin/python - "$RUNS" "$C/mcmc_cache" <<'PY'
import sys, yaml
from pathlib import Path
from src.models import pymc_inference
from src.pipelines.outer_loop.model_loop_runner import experiment_input_models_dir
runs, cache = Path(sys.argv[1]), Path(sys.argv[2])
fit_kwargs = yaml.safe_load(open("scripts/number_game/configs/holdout_recovery.yaml"))["fit"]
print("fit_kwargs", fit_kwargs)
loop = runs / "experiment1" / "model_loop"
pymc_inference.file_cached_fit_under(
    "iter4_candidate5", loop / "models", "inner_loop_model", loop / "responses.csv",
    cache_dir=cache, **fit_kwargs)
def refuse(*a, **k): raise AssertionError("sampled")
pymc_inference.fit_model = refuse
models_dir = experiment_input_models_dir(runs / "experiment2")
fitted = pymc_inference.load_cached_fit(
    "inner_loop_model", models_dir, loop / "responses.csv", cache_dir=cache, **fit_kwargs)
print("design lookup OK:", models_dir, fitted.fingerprint)
PY
ls -la $C/mcmc_cache | grep inner_loop_model
