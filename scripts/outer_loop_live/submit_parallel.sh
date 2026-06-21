#!/bin/bash
# Launch K parallel LIVE outer-loop runs on Sherlock. Each run gets:
#   * its own rsync copy of the repo -> isolated public/, firebase.generated.json,
#                                     opencode.json, functions/ (captures your
#                                     CURRENT working tree; no commit needed)
#   * its own --run-label          -> /e{N}-{label}/ hosting path + Firestore
#                                     collection_session_id + its own Prolific study
#   * its own AUTO_PSYCH_OUTPUT_DIR -> isolated experiment{N}/ data tree
#   * its own private XDG dirs (set in run_live.sbatch) -> no opencode DB collision
# The brief `firebase deploy` step is serialized across runs by the shared
# AUTO_PSYCH_DEPLOY_LOCK (set in _env.sh); everything else runs concurrently.
#
# Usage:   bash scripts/outer_loop_live/submit_parallel.sh
# Knobs:   K=3 EXPERIMENT=1 N_PARTICIPANTS=20 PROJECT=subjective_randomness \
#          DESIGN_MODE=agent WORK_ROOT=$SCRATCH/auto-psych/outer_loop_live \
#            bash scripts/outer_loop_live/submit_parallel.sh
#
# Prereqs (see README.md): run setup.sbatch once; put PROLIFIC_API_TOKEN +
# FIREBASE_TOKEN in .secrets; set a real prolific_config.yaml for the project.

set -euo pipefail

# Absolute dir of this script — passed to every job so they find _env.sh.
OUTER_LIVE_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTER_LIVE_SLURM_DIR
source "$OUTER_LIVE_SLURM_DIR/_env.sh"   # sets REPO, WORK_ROOT, caches, lock

K="${K:-3}"
export PROJECT="${PROJECT:-subjective_randomness}"
export EXPERIMENT="${EXPERIMENT:-1}"
export N_PARTICIPANTS="${N_PARTICIPANTS:-1}"
export FIREBASE_PROJECT="${FIREBASE_PROJECT:-auto-psych-2c5da}"
export DESIGN_MODE="${DESIGN_MODE:-agent}"
export CODING_AGENT="${CODING_AGENT:-opencode}"

# Optional Slurm overrides (else use run_live.sbatch's directives). For a long
# multi-experiment live run: WALLTIME=2-00:00:00 QOS=long.
WALLTIME="${WALLTIME:-}"
QOS="${QOS:-}"
EXTRA_SBATCH=()
[[ -n "$WALLTIME" ]] && EXTRA_SBATCH+=(--time="$WALLTIME")
[[ -n "$QOS" ]] && EXTRA_SBATCH+=(--qos="$QOS")

LOGDIR="$WORK_ROOT/slurm_logs"; mkdir -p "$LOGDIR"
RUNS_ROOT="$WORK_ROOT/runs"; mkdir -p "$RUNS_ROOT"
SRC_SHA="$( (cd "$REPO" && git rev-parse --short HEAD) 2>/dev/null || echo working-tree)"

# RUNS: which run indices to launch (default 1..K). Use e.g. RUNS="2 3" to
# (re)launch only those runs without disturbing the others (e.g. a salvaged run1).
if [[ -n "${RUNS:-}" ]]; then read -r -a _runs <<< "${RUNS//,/ }"; else _runs=($(seq 1 "$K")); fi

for i in "${_runs[@]}"; do
  LABEL="run${i}"
  WT="$RUNS_ROOT/$LABEL/repo"
  OUT="$WORK_ROOT/$LABEL/data"; mkdir -p "$OUT" "$WT"
  # Per-run rsync copy of the CURRENT working tree (el7 git lacks `git worktree`;
  # rsync needs no commit). Isolates public/, firebase.generated.json, opencode.json.
  rsync -a --delete \
    --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data' \
    --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees' \
    --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules' \
    --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
    "$REPO"/ "$WT"/
  touch "$WT/.here"   # pyprojroot sentinel (.git excluded)
  # Each run deploys to its OWN Firebase Hosting site so concurrent deploys don't
  # clobber the shared live site (the bug that 404'd all but the last run). Site
  # IDs must be lowercase and <=30 chars; the deploy creates it if missing.
  HOST_SITE="$(echo "${FIREBASE_PROJECT}-${LABEL}" | tr '[:upper:]' '[:lower:]')"
  jid=$(sbatch --parsable \
    --job-name="outer_live_$LABEL" \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    ${EXTRA_SBATCH[@]+"${EXTRA_SBATCH[@]}"} \
    --export=ALL,RUN_LABEL="$LABEL",RUN_WORKTREE="$WT",CODING_AGENT="$CODING_AGENT",AUTO_PSYCH_OUTPUT_DIR="$OUT",AUTO_PSYCH_HOSTING_SITE="$HOST_SITE" \
    "$OUTER_LIVE_SLURM_DIR/run_live.sbatch")
  echo "submitted $LABEL: job $jid  (copy=$WT  out=$OUT)"
done

echo
echo "code @ $SRC_SHA   |   watch: squeue --me   |   logs: $LOGDIR"
echo "Tear down run copies when done:  rm -rf $RUNS_ROOT/<label>"
