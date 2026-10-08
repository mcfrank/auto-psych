#!/bin/bash
#SBATCH --job-name=rsa_design
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# The first live experiment's design and its power (src.rsa.design.run), from
# the promoted seeds, after promote.sbatch: their fits on all existing data are
# reused from its cache, so nothing is refitted. No participants, no agents.
# About 40 s per designed display (60) plus the power table; expect about an hour.
#
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/design.sbatch
#
# Writes data/rsa/live_seeds/design_n<N>/design.json and eig.json in the
# checkout (commit both). N_RESPONSES (default 100): participants per
# experiment the design is chosen for; POWER_NS (default "40 60 100 150"): the
# participant counts its power is reported at.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
SEEDS="${SEEDS:-$REPO/data/rsa/live_seeds}"
WORK="$WORK_ROOT/promote_work"
N_RESPONSES="${N_RESPONSES:-100}"
read -r -a POWER_NS <<< "${POWER_NS:-40 60 100 150}"
for f in "$SEEDS/models/models_manifest.yaml" "$WORK/all_trials.csv"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing: run promote.sbatch first" >&2; exit 1; }
done
cd "$REPO"
"$VENV_PY" -m src.rsa.design.run --models-dir "$SEEDS/models" --data "$WORK/all_trials.csv" \
  --cache "$WORK/.fit_cache" --out "$SEEDS/design_n$N_RESPONSES" --n-responses "$N_RESPONSES" \
  --power-ns "${POWER_NS[@]}"
echo "done: $SEEDS/design_n$N_RESPONSES (commit design.json and eig.json)"
