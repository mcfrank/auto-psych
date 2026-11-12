#!/bin/bash
#SBATCH --job-name=paper_information_criteria
#SBATCH --partition=normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=%x-%A_%a.out
# ============================================================================
#  AIC, BIC and the log evidence (Bayes factors) of each domain's best models
#  and seeds on all runs pooled (scripts/paper/information_criteria.py), one
#  task per model: 28 tasks (`--mode tasks` lists them). Each writes a record
#  under $WORK_ROOT/information_criteria/records; then, anywhere with the records:
#
#    cd $HOME/auto-psych && sbatch --array=3-27 scripts/paper/information_criteria.sbatch
#    sbatch --array=0-2 --time=2-00:00:00 scripts/paper/information_criteria.sbatch  # the SR winners
#    uv run python -m scripts.paper.information_criteria --mode summarize --work-dir <records' dir>
#
#  Reads the subjective-randomness fits sr_figure_data.sbatch cached and the
#  RSA campaign's pooled data (checked against the bar's sha256).
# ============================================================================
set -euo pipefail
REPO="${REPO:-$HOME/auto-psych}"
WORK_ROOT="${WORK_ROOT:-$SCRATCH/auto-psych/paper}"
VENV_PY="${VENV_PY:-$GROUP_HOME/venvs/auto-psych/bin/python}"
RSA_POOLED="${RSA_POOLED:-$SCRATCH/auto-psych/rsa_chain_models/pool/pooled.csv}"

ml purge
ml load devel
ml load gcc/14.2.0  # pytensor compiles every PyMC model with a C compiler
export PYTENSOR_FLAGS="base_compiledir=${L_SCRATCH:-/tmp}/pytensor/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
# arviz stamps a daily warning under the cache dir; a shared /tmp races between tasks on one node
export XDG_CACHE_HOME="${L_SCRATCH:-/tmp}/cache/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"

cd "$REPO"
echo "[ic] code $(git rev-parse HEAD) $( [[ -z "$(git status --porcelain)" ]] && echo clean || echo DIRTY)"
"$VENV_PY" -m scripts.paper.information_criteria --mode run --task "$SLURM_ARRAY_TASK_ID" \
  --work-dir "$WORK_ROOT/information_criteria" --sr-fit-dir "$WORK_ROOT/sr_figure_data" \
  --rsa-pooled "$RSA_POOLED" --n-samples 16000
