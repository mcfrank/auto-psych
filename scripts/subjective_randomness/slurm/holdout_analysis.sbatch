#!/bin/bash
#SBATCH --job-name=holdout_test_retest
#SBATCH --partition=mcfrank
#SBATCH --time=00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Stage 3/3: aggregate the repeats into a test-retest reliability summary.
# Submitted with `--dependency=afterok:<array_job_id>` so it runs once the
# array finishes (use afterany if you want it to run even when a repeat fails).

set -euo pipefail
SLURM_DIR="${HOLDOUT_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

# Cells whose agents wrote the held-out model's name (scan_gt_name.sh
# --after-run). Only a warning: an agent can coin the name unaided.
warnings="$WORK_ROOT/GT_NAME_WARNINGS.txt"
: > "$warnings"
for mentions in "$WORK_ROOT"/run*/*/gt_name_mentions.txt; do
  [[ -f "$mentions" ]] || continue
  echo "$(dirname "$mentions" | sed "s|^$WORK_ROOT/||"): $(wc -l < "$mentions") file(s), see $mentions" >> "$warnings"
done
if [[ -s "$warnings" ]]; then
  echo "[analysis] WARNING: agents wrote a held-out ground-truth name in these cells:"
  cat "$warnings"
else
  rm -f "$warnings"
  echo "[analysis] no agent wrote a held-out ground-truth name"
fi

# Cells whose agents touched paths outside their own directory (listed in each
# cell's agent_activity.md, with the URLs they looked at). Warn-only.
outside="$WORK_ROOT/OUTSIDE_PATH_WARNINGS.txt"
: > "$outside"
for report in "$WORK_ROOT"/run*/*/agent_activity.md; do
  [[ -f "$report" ]] || continue
  if ! grep -A2 "^## Paths outside" "$report" | grep -q "^(none)"; then
    echo "$(dirname "$report" | sed "s|^$WORK_ROOT/||"): see $report" >> "$outside"
  fi
done
if [[ -s "$outside" ]]; then
  echo "[analysis] WARNING: agents touched paths outside their own directory in these cells:"
  cat "$outside"
else
  rm -f "$outside"
fi

# Every expected cell without a result, so a failed cell cannot silently drop
# out of the summary (survivorship). A later retry's summary may fill some in.
missing="$WORK_ROOT/MISSING_CELLS.txt"
"$VENV_PY" "$SLURM_DIR/cell_status.py" missing --work-root "$WORK_ROOT" \
  --n-repeats "$N_REPEATS" --gt-models "$GT_MODELS" > "$missing"
if [[ -s "$missing" ]]; then
  echo "[analysis] WARNING: $(wc -l < "$missing") expected cell(s) have no result (listed in $missing):"
  cat "$missing"
else
  rm -f "$missing"
  echo "[analysis] every expected cell has a result"
fi

"$VENV_PY" scripts/subjective_randomness/holdout_test_retest.py \
  --runs-root "$WORK_ROOT" \
  --n-repeats "$N_REPEATS" --gt-models "$GT_MODELS" \
  --out    "$WORK_ROOT/test_retest.json" \
  --csv    "$WORK_ROOT/test_retest.csv" \
  --figure "$WORK_ROOT/test_retest.png"

echo "[analysis] wrote $WORK_ROOT/test_retest.{json,csv,png}"
