#!/bin/bash
#SBATCH --job-name=rsa_prepare
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=01:30:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
# prepare_data.sh as a job: Sherlock's rules keep venv builds and data
# processing off the login nodes (run 1 ran it this way; compute nodes reach
# PyPI and every data host). Submit from the checkout, with the sweep's
# environment exported (WORK_ROOT etc.; --export=ALL is the default):
#
#   sbatch --chdir="$HOME/auto-psych" scripts/rsa/slurm/prepare.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?submit from the checkout}"
exec bash scripts/rsa/slurm/prepare_data.sh
