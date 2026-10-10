#!/bin/bash
#SBATCH --job-name=rsa_promote
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=20
#SBATCH --mem=64GB
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
#
# Promote run 2's models to seed the live experiments (src.rsa.promote; PI
# decision 2026-10-08): every model the three real-data cells admitted (143),
# refitted on all existing data, grouped into 10 by their predictions, the
# best of each group by grouped CV, plus rsa_l2. No agents: compute only,
# about 860 single-threaded fits (one full fit and 5 CV folds per model),
# 20 at a time; expect 1-3 hours. From the checkout (with run 2's cells
# committed in data/rsa/sherlock_run2/):
#
#   mkdir -p "$WORK_ROOT/logs"
#   sbatch --chdir="$HOME/auto-psych" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/promote.sbatch
#
# Writes data/rsa/live_seeds/ in the checkout (models/ and promotion.json: commit
# both; they hold model files and aggregates only). The combined trial file,
# folds and fits stay in $WORK_ROOT/promote_work. Resubmitting resumes from the
# fit cache. Override with SWEEP, OUT, K.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
source "$REPO/scripts/rsa/slurm/_env.sh"
SWEEP="${SWEEP:-$REPO/data/rsa/sherlock_run2}"
OUT="${OUT:-$REPO/data/rsa/live_seeds}"
K="${K:-10}"
for f in "$DATA_ROOT/real/train.csv" "$DATA_ROOT/real/test.csv" "$SWEEP/real_rep1/history.json"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing (WORK_ROOT=$WORK_ROOT, SWEEP=$SWEEP)" >&2; exit 1; }
done
# The data must be the data run 2 ran on.
(cd "$DATA_ROOT/real" && sha256sum --quiet -c <(grep -E ' (train|test)\.csv$' "$SWEEP/real_rep1/data.sha256")) \
  || { echo "ERROR: $DATA_ROOT/real/{train,test}.csv are not the data run 2 recorded" >&2; exit 1; }
cd "$REPO"
"$VENV_PY" -m src.rsa.promote --sweep "$SWEEP" --train "$DATA_ROOT/real/train.csv" --test "$DATA_ROOT/real/test.csv" \
  --work "$WORK_ROOT/promote_work" --out "$OUT" --k "$K" --workers "${SLURM_CPUS_PER_TASK:-20}"
echo "done: $OUT (commit models/ and promotion.json)"
