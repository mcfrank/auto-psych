#!/bin/bash
# Submit the number-game ablation (docs/ng_ablation.md): a setup job, then the driver chain.
#
#   cd $HOME/auto-psych && bash scripts/number_game/slurm/submit_ablation.sh
#
# Resubmitting resumes: every finished fold fit is saved under $WORK_ROOT/ng/cv/. Do not
# resubmit while the chain is still queued or running (squeue -u $USER | grep ng_abl),
# or two drivers will submit the same round twice.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/ng_ablation}"
SBATCH="$REPO/scripts/number_game/slurm/ablation.sbatch"
mkdir -p "$WORK_ROOT/logs"
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo "[ablation] WARNING: the checkout is dirty; its state is what gets staged" >&2

setup=$(sbatch --parsable --job-name=ng_abl_setup -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=setup,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
echo "[ablation] setup $setup"
driver=$(sbatch --parsable --job-name=ng_abl_driver --dependency="afterok:$setup" \
  -o "$WORK_ROOT/logs/%x_%j.out" --export=ALL,STAGE=driver,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
echo "[ablation] first driver $driver"
echo "[ablation] work root $WORK_ROOT; logs $WORK_ROOT/logs; watch: squeue -u \$USER | grep ng_abl"
