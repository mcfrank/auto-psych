#!/bin/bash
#SBATCH --job-name=open_opt
#SBATCH --partition=owners
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --requeue
# One §10 test per array task: the task's arguments are line $SLURM_ARRAY_TASK_ID of $OPT_TABLE.
set -uo pipefail
B="$SCRATCH/auto-psych/sched_test"
export REPO="$B/clone"
export WORK_ROOT="$B/work"
export OPT_BASE="$B/opt"
cd "$REPO"
source scripts/rsa/slurm/_env.sh
set +e
export PYTHONPATH="$REPO"
ARGS=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OPT_TABLE")
echo "task $SLURM_ARRAY_TASK_ID: $ARGS on $(hostname)"
"$VENV_PY" "$B/opt_tests.py" $ARGS
