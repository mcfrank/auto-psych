#!/bin/bash
# Submit the sr-oct26 ablation (docs/sr_ablation.md): a setup job, then one driver chain per final.
#
#   cd $HOME/auto-psych && bash scripts/subjective_randomness/slurm/submit_ablation.sh
#   FINALS="run2" bash scripts/subjective_randomness/slurm/submit_ablation.sh   # one final only
#
# Resubmitting resumes: every finished fold fit is saved under $WORK_ROOT/<final>/cv/. Do not
# resubmit a final whose chain is still queued or running (squeue -u $USER -n sr_abl_<final>_driver),
# or two drivers will submit the same round twice.
set -euo pipefail
export REPO="${REPO:-$HOME/auto-psych}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/sr_ablation}"
FINALS="${FINALS:-run1 run2 run3}"
SBATCH="$REPO/scripts/subjective_randomness/slurm/ablation.sbatch"
mkdir -p "$WORK_ROOT/logs"
cd "$REPO"
[[ -z "$(git status --porcelain --untracked-files=no)" ]] || echo "[ablation] WARNING: the checkout is dirty; its state is what gets staged" >&2

setup=$(sbatch --parsable --job-name=sr_abl_setup -o "$WORK_ROOT/logs/%x_%j.out" \
  --export=ALL,STAGE=setup,REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
echo "[ablation] setup $setup"
# The first plan writes the CV folds every final shares; the other drivers start after it ends
# (two plans writing the same fold files at once could interleave them).
first=""
for final in $FINALS; do
  driver=$(sbatch --parsable --job-name="sr_abl_${final}_driver" --dependency="afterok:$setup${first:+,afterany:$first}" \
    -o "$WORK_ROOT/logs/%x_%j.out" --export=ALL,STAGE=driver,FINAL="$final",REPO="$REPO",WORK_ROOT="$WORK_ROOT" "$SBATCH")
  echo "[ablation] $final: first driver $driver"
  first="${first:-$driver}"
done
echo "[ablation] work root $WORK_ROOT; logs $WORK_ROOT/logs; watch: squeue -u \$USER | grep sr_abl"
