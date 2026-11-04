#!/bin/bash
#SBATCH --job-name=open_ref_fit
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --cpus-per-task=1
#SBATCH --mem=18G
#SBATCH --time=03:30:00
#SBATCH --requeue
# One reference fit per task (no agents, no Prolific): queue wait, preemption and speed against the lab node.
set -uo pipefail
B="$SCRATCH/auto-psych/sched_test"
export REPO="$B/clone"
export WORK_ROOT="$B/work"
cd "$REPO"
source scripts/rsa/slurm/_env.sh
set +e
export PYTHONPATH="$REPO"
"$VENV_PY" "$B/ref_fit.py" "$B/ref_fits"
