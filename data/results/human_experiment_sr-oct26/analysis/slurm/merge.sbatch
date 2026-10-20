#!/bin/bash
#SBATCH --job-name=srgen_merge
#SBATCH --partition=normal
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4GB
#SBATCH --output=%x_%j.out
#
# Stage 4/4: assemble each analysis' parts into its tables (fails if a model
# is missing or scored twice). Then copy results/generalization/ back and run
# report_generalization.py.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$(pwd)}/_env.sh"

while IFS=$'\t' read -r analysis source target; do
  uv run --no-sync python "$ANALYSIS/generalization_analysis.py" \
    --analysis "$analysis" --source "$source" --target "$target" --merge
done < <(cut -f1-3 "$ANALYSIS/results/generalization/scores.tsv" | sort -u)
