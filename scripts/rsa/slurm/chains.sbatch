#!/bin/bash
#SBATCH --job-name=rsa_chains
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Split the promoted seeds across the live chains (src.rsa.outer.chains; PI
# 2026-10-09), then each chain's experiment-1 design and its power
# (src.rsa.design.run). After promote.sbatch: every fit is in its cache, so
# nothing is refitted. No participants, no agents. About 40 s per designed
# display (20 + 40 per chain by default) plus the power tables: 2-3 hours.
#
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/chains.sbatch
#
# Writes $OUT/chains.json, $OUT/chain_<k>/ (the chain's seeds and manifest) and
# $OUT/chain_<k>/design_t<TRIALS>/ (design_d<D>.json, eig.json). OUT defaults
# to scratch, so the checkout stays clean: copy it to data/rsa/live_seeds/chains/
# and commit. Overrides: N_CHAINS (3), TRIALS (10), PARTICIPANTS (200),
# DISPLAYS ("20 40"), POWER_PARTICIPANTS ("100 200").
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
PROMOTED="${PROMOTED:-$REPO/data/rsa/live_seeds/models}"
WORK="$WORK_ROOT/promote_work"
OUT="${OUT:-$WORK_ROOT/chains}"
N_CHAINS="${N_CHAINS:-3}"
TRIALS="${TRIALS:-10}"
PARTICIPANTS="${PARTICIPANTS:-200}"
read -r -a DISPLAYS <<< "${DISPLAYS:-20 40}"
read -r -a POWER_PARTICIPANTS <<< "${POWER_PARTICIPANTS:-100 200}"
for f in "$PROMOTED/models_manifest.yaml" "$WORK/all_trials.csv"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing: run promote.sbatch first" >&2; exit 1; }
done
cd "$REPO"
"$VENV_PY" -m src.rsa.outer.chains --promoted "$PROMOTED" --data "$WORK/all_trials.csv" \
  --cache "$WORK/.fit_cache" --out "$OUT" --n-chains "$N_CHAINS"
for ((k = 0; k < N_CHAINS; k++)); do
  echo "[chains] chain $k: design for ${DISPLAYS[*]} displays"
  "$VENV_PY" -m src.rsa.design.run --models-dir "$OUT/chain_$k" --data "$WORK/all_trials.csv" \
    --cache "$WORK/.fit_cache" --out "$OUT/chain_$k/design_t$TRIALS" --trials-per-participant "$TRIALS" \
    --participants "$PARTICIPANTS" --displays "${DISPLAYS[@]}" --power-participants "${POWER_PARTICIPANTS[@]}"
done
echo "done: $OUT (copy to data/rsa/live_seeds/chains/ and commit)"
