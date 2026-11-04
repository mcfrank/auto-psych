#!/bin/bash
#SBATCH --job-name=open_opt_thin
#SBATCH --partition=owners
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --no-requeue
# §10 test 1: thinning saved folds, on a copy of chain 1's experiment 2 loop (removed at the end). No fits, no agents.
set -uo pipefail
B="$SCRATCH/auto-psych/sched_test"
export REPO="$B/clone"
export WORK_ROOT="$B/work"
export OPT_BASE="$B/opt"
C="$SCRATCH/auto-psych/agent_trees/587bd35ad48e9e44/repo/_runs/chain_0/experiment2"
P="$B/opt/thin_copy"
mkdir -p "$P"
rsync -a --exclude 'round_*' "$C/loop/" "$P/loop/"
cp "$C/corpus.json" "$P/corpus.json"
cd "$REPO"
source scripts/rsa/slurm/_env.sh
set +e
export PYTHONPATH="$REPO"
"$VENV_PY" "$B/opt_tests.py" thin "$P"
rc=$?
rm -rf "$P"
echo "exit $rc; copy removed"
exit $rc
