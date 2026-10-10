#!/bin/bash
#SBATCH --job-name=rsa_design
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=1
#SBATCH --mem=7GB
#SBATCH --time=06:00:00
#SBATCH --output=%x_%j.out
#
# The first live experiment's design and its power (src.rsa.design.run), from
# the promoted seeds, after promote.sbatch: their fits on all existing data are
# reused from its cache, so nothing is refitted. No participants, no agents.
# About 40 s per designed display (10 + 20 + 30 by default) plus the power tables; about an hour.
#
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/design.sbatch
#
# A participant answers TRIALS designed displays (default 10, plus 2 catch
# trials: 12 test trials; PI 2026-10-08), a balanced subset of the design's D.
# For each D in DISPLAYS (default "10 20 30") it picks D displays for
# PARTICIPANTS (default 200) people and reports power at POWER_PARTICIPANTS
# (default "100 200 300"). Writes data/rsa/live_seeds/design_t<TRIALS>/
# design_d<D>.json and eig.json in the checkout (commit them).
# One process on one CPU (HANDOFF_threads.md): memory buys CPUs on mcfrank
# (MaxMemPerCPU 8000 MB), so --mem stays under 8 GB; the design peaked at 1.8 GB.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
SEEDS="${SEEDS:-$REPO/data/rsa/live_seeds}"
WORK="$WORK_ROOT/promote_work"
TRIALS="${TRIALS:-10}"
PARTICIPANTS="${PARTICIPANTS:-200}"
read -r -a DISPLAYS <<< "${DISPLAYS:-10 20 30}"
read -r -a POWER_PARTICIPANTS <<< "${POWER_PARTICIPANTS:-100 200 300}"
for f in "$SEEDS/models/models_manifest.yaml" "$WORK/all_trials.csv"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing: run promote.sbatch first" >&2; exit 1; }
done
cd "$REPO"
"$VENV_PY" -m src.rsa.design.run --models-dir "$SEEDS/models" --data "$WORK/all_trials.csv" \
  --cache "$WORK/.fit_cache" --out "$SEEDS/design_t$TRIALS" --trials-per-participant "$TRIALS" \
  --participants "$PARTICIPANTS" --displays "${DISPLAYS[@]}" --power-participants "${POWER_PARTICIPANTS[@]}"
echo "done: $SEEDS/design_t$TRIALS (commit design_d*.json and eig.json)"
