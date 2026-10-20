#!/bin/bash
# Submit the sr-oct26 generalization analysis: setup -> 32 fits -> 24 scoring tasks -> merge.
# Run from this directory on a Sherlock login node (it only calls sbatch):
#   cd $REPO/data/results/human_experiment_sr-oct26/analysis/slurm && ./submit.sh
# Afterwards, copy results/generalization/ back and run report_generalization.py.
set -euo pipefail
cd "$(dirname "$0")"

setup=$(sbatch --parsable setup.sbatch)
fits=$(sbatch --parsable --dependency=afterok:"$setup" fit_array.sbatch)
score=$(sbatch --parsable --dependency=afterok:"$fits" analysis_array.sbatch)
merge=$(sbatch --parsable --dependency=afterok:"$score" merge.sbatch)
echo "setup $setup -> fits $fits (32 tasks) -> scoring $score (24 tasks) -> merge $merge"
echo "watch: squeue -u \$USER; logs: $(pwd)/srgen_*.out"
