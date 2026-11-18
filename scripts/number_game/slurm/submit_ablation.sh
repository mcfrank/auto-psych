#!/bin/bash
# Submit the number-game ablation (docs/ng_ablation.md): a setup job, then the driver chain.
#
#   cd $HOME/auto-psych && bash scripts/number_game/slurm/submit_ablation.sh            # run 2's final (ng)
#   cd $HOME/auto-psych && FINAL=ew bash scripts/number_game/slurm/submit_ablation.sh   # run 3's (ew)
#
# Resubmitting resumes: every finished fold fit is saved under $WORK_ROOT/$FINAL/cv/. Do not
# resubmit while that final's chain is still queued or running (squeue -u $USER | grep ng_abl),
# or two drivers will submit the same round twice. Each final has its own work root and job names.
set -euo pipefail
export FINAL="${FINAL:-ng}"
[[ "$FINAL" =~ ^(ng|ew)$ ]] || { echo "ERROR: FINAL must be ng or ew, not $FINAL" >&2; exit 1; }
if [[ "$FINAL" == ng ]]; then ABL=ng_abl; DEFAULT_ROOT=ng_ablation_nov26; else ABL="ng_abl_$FINAL"; DEFAULT_ROOT="ng_ablation_nov26_$FINAL"; fi
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/$DEFAULT_ROOT}"
SBATCH="$REPO/scripts/number_game/slurm/ablation.sbatch"
mkdir -p "$WORK_ROOT/logs"
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo "[ablation] WARNING: the checkout is dirty; its state is what gets staged" >&2

setup=$(sbatch --parsable --job-name="${ABL}_setup" -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=setup,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
echo "[ablation] setup $setup"
driver=$(sbatch --parsable --job-name="${ABL}_driver" --dependency="afterok:$setup" \
  -o "$WORK_ROOT/logs/%x_%j.out" --export=ALL,STAGE=driver,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
echo "[ablation] first driver $driver"
echo "[ablation] final $FINAL; work root $WORK_ROOT; logs $WORK_ROOT/logs; watch: squeue -u \$USER | grep $ABL"
