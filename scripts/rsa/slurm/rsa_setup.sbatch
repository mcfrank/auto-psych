#!/bin/bash
#SBATCH --job-name=rsa_setup
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# RSA run 1, step 2 (compute node; submit.sh submits it, with resources from
# its SETUP_* variables, and the array after it with afterok):
#   1. checks the venv prepare_data.sh built works on a compute node (jax 0.7.0
#      on glibc 2.17), opencode >= 1.18.35 and the Gemini key;
#   2. stages the sweep's code once (stage_code.sh: harness_repo, agent_src,
#      code_commit) -- the same code that prepared the data, or it stops;
#   3. checks the loop CLI takes --seed (the replicates' loop seeds), unless
#      ALLOW_SHARED_LOOP_SEED=1 accepts replicates that differ only by the
#      agents' sampling;
#   4. simulates the two recovery datasets: each ground truth is fitted to the
#      REAL training split, choices are drawn for every included forced-choice
#      row of the combined data, and the result is split with the same seed
#      (the same held-out units as the real split, which is checked). The
#      provenance (it names the ground truth) stays in $DATA_ROOT, outside
#      every agent tree.
# Idempotent: a finished simulation ($DATA_ROOT/<condition>/DONE) is kept.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
source "$REPO/scripts/rsa/slurm/_env.sh"
SPLIT_SEED="${SPLIT_SEED:-0}"
SIM_SEED="${SIM_SEED:-0}"
NUM_WARMUP="${NUM_WARMUP:-1000}"; NUM_SAMPLES="${NUM_SAMPLES:-1000}"; NUM_CHAINS="${NUM_CHAINS:-4}"
CODE_COMMIT_SH="$REPO/scripts/subjective_randomness/slurm/code_commit.sh"

rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }

# --- 1. the venv on a compute node --------------------------------------------
[[ -x "$VENV_PY" ]] || { echo "ERROR: no venv at $VENV_PY; run prepare_data.sh on a login node first" >&2; exit 1; }
"$VENV_PY" - <<'PY'
import jax, jax.numpy as jnp, numpyro, memo, pymc  # noqa: F401
print(f"[setup] venv OK on {__import__('platform').node()}: jax {jax.__version__}, "
      f"{len(jax.devices())} device(s), check {float(jnp.ones(3).sum())}")
PY

# --- 2. data prepared by this code; stage it ----------------------------------
for f in prepared_code combined_trials.csv real/train.csv real/test.csv real/split.json; do
  [[ -f "$DATA_ROOT/$f" ]] || { echo "ERROR: $DATA_ROOT/$f is missing; run prepare_data.sh first" >&2; exit 1; }
done
prepared_code=$(cat "$DATA_ROOT/prepared_code")
repo_code=$(bash "$CODE_COMMIT_SH" "$REPO")
if [[ "$prepared_code" != "$repo_code" ]]; then
  echo "ERROR: the data in $DATA_ROOT were prepared by code $prepared_code, but $REPO is at $repo_code." >&2
  echo "       Re-run prepare_data.sh on the current checkout (or check out $prepared_code)." >&2
  exit 1
fi
bash "$REPO/scripts/rsa/slurm/stage_code.sh"
[[ "$(cat "$WORK_ROOT/code_commit")" == "$prepared_code" ]] \
  || { echo "ERROR: staged code $(cat "$WORK_ROOT/code_commit") != the code that prepared the data $prepared_code" >&2; exit 1; }

# Everything below runs on the staged copy, as the array will.
cd "$HARNESS_REPO"
"$VENV_PY" - <<'PY'
import sys
from pyprojroot import here
sys.path.insert(0, str(here()))
import src.rsa.loop.run, src.rsa.simulate, src.rsa.split, src.rsa.evaluate_heldout, src.rsa.recovery  # noqa: F401,E401
import src.rsa.loop.check_candidate  # noqa: F401
print("[setup] import chain OK")
PY

# --- 3. the replicates' loop seeds -------------------------------------------------
if rsa_loop_cli_has_seed "$VENV_PY"; then
  echo "[setup] the loop CLI takes --seed: replicate r runs with loop seed BASE_LOOP_SEED + r - 1 (BASE_LOOP_SEED=${BASE_LOOP_SEED:-0})"
elif [[ "${ALLOW_SHARED_LOOP_SEED:-}" == 1 ]]; then
  echo "[setup] WARNING: the loop CLI has no --seed; ALLOW_SHARED_LOOP_SEED=1, so the two replicates of a condition share every seed and differ only by the agents' own sampling."
else
  echo "ERROR: src/rsa/loop/run.py takes no --seed, so the two replicates of a condition would differ only by the agents' sampling." >&2
  echo "       Add --seed to the loop CLI (the PI's design), or accept that with ALLOW_SHARED_LOOP_SEED=1." >&2
  exit 1
fi

# --- 4. the recovery datasets --------------------------------------------------
simulate() {
  local cond="$1" dir gt gt_file
  dir="$(rsa_condition_data_dir "$DATA_ROOT" "$cond")"
  gt="$(rsa_condition_gt "$cond")"
  gt_file="$HARNESS_REPO/$RSA_SEED_MODELS_REL/$gt.py"
  [[ -f "$gt_file" ]] || { echo "ERROR: ground truth $gt_file is missing" >&2; return 1; }
  rm -rf "$dir"; mkdir -p "$dir"
  "$VENV_PY" -m src.rsa.simulate --gt "$gt_file" --fit-on "$DATA_ROOT/real/train.csv" \
    --displays "$DATA_ROOT/combined_trials.csv" --out "$dir/simulated.csv" \
    --provenance "$dir/provenance.json" --seed "$SIM_SEED" \
    --num-warmup "$NUM_WARMUP" --num-samples "$NUM_SAMPLES" --num-chains "$NUM_CHAINS"
  "$VENV_PY" -m src.rsa.split --trials "$dir/simulated.csv" --out-dir "$dir" --seed "$SPLIT_SEED"
  sha256sum "$gt_file" "$DATA_ROOT/real/train.csv" > "$dir/DONE"
}
pids=(); conds=()
for cond in "${RSA_CONDITIONS[@]}"; do
  [[ -n "$(rsa_condition_gt "$cond")" ]] || continue
  dir="$(rsa_condition_data_dir "$DATA_ROOT" "$cond")"
  if [[ -f "$dir/DONE" ]] && sha256sum --quiet -c "$dir/DONE"; then
    echo "[setup] $cond: already simulated ($dir)"
    continue
  fi
  echo "[setup] $cond: simulating from $(rsa_condition_gt "$cond") (log: $DATA_ROOT/simulate_$cond.log)"
  simulate "$cond" > "$DATA_ROOT/simulate_$cond.log" 2>&1 &
  pids+=($!); conds+=("$cond")
done
failed=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then echo "[setup] ${conds[$i]}: simulated"; else
    echo "ERROR: simulating ${conds[$i]} failed; see $DATA_ROOT/simulate_${conds[$i]}.log" >&2; failed=1; fi
done
[[ $failed -eq 0 ]] || exit 1

# Same held-out units and the same rows as the real split; only choices differ.
"$VENV_PY" - "$DATA_ROOT" "${RSA_CONDITIONS[@]}" <<'PY'
import json, sys
from pathlib import Path
import pandas as pd
root, conditions = Path(sys.argv[1]), sys.argv[2:]
real = root / "real"
real_units = json.loads((real / "split.json").read_text())["held_out_units"]
for cond in conditions:
    d = root / cond
    if cond == "real" or not (d / "split.json").exists():
        continue
    if json.loads((d / "split.json").read_text())["held_out_units"] != real_units:
        sys.exit(f"{cond}: the simulated split holds out other units than the real split")
    for part in ("train", "test"):
        # The simulated data keep only the rows the loop fits (src.rsa.simulate).
        a = pd.read_csv(real / f"{part}.csv")
        fitted = (a["dv"] == "forced_choice") & a["included"].astype(bool) & (a["objects"] != "[]")
        a = a[fitted].drop(columns=["choice"]).reset_index(drop=True)
        b = pd.read_csv(d / f"{part}.csv").drop(columns=["choice"]).reset_index(drop=True)
        if not a.equals(b):
            sys.exit(f"{cond}: {part}.csv differs from the real {part}.csv outside the choice column")
    print(f"[setup] {cond}: same held-out units and rows as the real split")
PY

(cd "$DATA_ROOT" && sha256sum recovery_*/simulated.csv recovery_*/train.csv recovery_*/test.csv recovery_*/split.json recovery_*/provenance.json > SHA256SUMS.recovery)
cat "$DATA_ROOT/SHA256SUMS.recovery"
echo "[setup] done: code $(cat "$WORK_ROOT/code_commit")"
