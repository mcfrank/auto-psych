#!/bin/bash
#SBATCH --job-name=rsa_finals_pool
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# Pooled fit of each chain's latest exported best model; predictions on novelty.plain_pool().
set -euo pipefail
export REPO=$HOME/auto-psych
export WORK_ROOT=$SCRATCH/auto-psych/rsa_campaign
source "$REPO/scripts/rsa/slurm/_env.sh"
cd "$WORK_ROOT/harness_repo"
PYTHONPATH="$WORK_ROOT/harness_repo" "$VENV_PY" "$WORK_ROOT/finals_pool_fit.py" "$WORK_ROOT" "$WORK_ROOT/finals_pool_preds.json"
