#!/bin/bash
#SBATCH --job-name=open_profile_resume
#SBATCH --partition=owners
#SBATCH --constraint=CPU_GEN:SIE
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --no-requeue
# Where a resume's time goes, on a copy of chain 1's experiment 2 loop (the chain's own files are only read).
set -uo pipefail
B="$SCRATCH/auto-psych/sched_test"
export REPO="$B/clone"
export WORK_ROOT="$B/work"
C="$SCRATCH/auto-psych/agent_trees/587bd35ad48e9e44/repo/_runs/chain_0/experiment2"
P="$B/resume_profile"
mkdir -p "$P"
rsync -a --exclude 'round_*' "$C/loop/" "$P/loop/"
cp "$C/corpus.json" "$P/corpus.json"
cd "$REPO"
source scripts/rsa/slurm/_env.sh
set +e
export PYTHONPATH="$REPO"
"$VENV_PY" "$B/profile_resume.py" "$P"
rc=$?
rm -rf "$P/loop" "$P/corpus.json"
echo "exit $rc; copy removed"
exit $rc
