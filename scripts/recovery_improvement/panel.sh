#!/bin/bash
# Start a review panel: several coding agents (Claude, Codex, ...) review both
# model-discovery loops — auto-psych's holdout-recovery sweeps and the
# llm-verbal-protocol loop's recovery studies — then discuss in written
# rounds, and a moderator writes a ranked consensus plan. Nothing is launched
# or modified; the plan (+ an optional proposed sweep) is the output.
#
# Usage:
#   bash scripts/recovery_improvement/panel.sh <panel_name> [baseline_root ...]
#
# Codex on Sherlock: its sandbox cannot start (the nodes cap net/uts namespaces
# at 0 and the ChatGPT workspace policy forbids bypassing it), so Codex members
# are PROMPT-ONLY — they get the digests, the thread and an inlined evidence
# pack (EVIDENCE_FILES) and return their note as their final message. Give the
# file-reading lenses (search, crossloop) to Claude members.
#
# Knobs (env vars; pinned into panel.env):
#   MEMBERS="methods:codex:gpt-5.6-sol:methods search:claude:claude-fable-5-1:search crossloop:claude:claude-fable-5-1:crossloop"
#                               name:backend:model:lens; lenses: methods search crossloop systems
#   MODERATOR=<name>            member whose backend/model writes the synthesis
#                               (default: the first Claude member — it must read files)
#   PROMPT_ONLY_BACKENDS=codex  backends that cannot run commands here
#   EVIDENCE_FILES="..."        files inlined into prompt-only members' briefs
#                               (default: the two loops' orchestrators, prompts,
#                               comparison code, spec and prior analyses, ~200 KB)
#   EVIDENCE_MAX_BYTES=300000
#   N_DISCUSSION_ROUNDS=2       rounds after the independent one
#   CAMPAIGN=<campaign_name>    improvement campaign whose prescriptions/sweeps the
#                               panel sees (default: recovery_2026_09_07 if it exists)
#   VERBAL_REPO=~/llm-verbal-protocol
#   MEMBER_MAX_TURNS=300  MEMBER_TIMEOUT_SEC=5400  MEMBER_MAX_BUDGET_USD=100
#   PANEL_PARTITION=normal PANEL_TIME=10:00:00 PANEL_CPUS=4 PANEL_MEM=16GB
#   PANEL_VENV=<path>           reuse an existing venv (e.g. a campaign's)
#   CODEX_MODULE=codex/0.151.0
#   DRY_RUN=1                   write panel.env and print the sbatch chain, submit nothing
set -euo pipefail

usage() { sed -n '2,36p' "$0" | sed 's/^# \{0,1\}//'; exit 1; }
[[ $# -ge 1 ]] || usage
NAME="$1"; shift
[[ "$NAME" =~ ^[A-Za-z0-9_-]+$ ]] || { echo "ERROR: bad panel name '$NAME'" >&2; exit 1; }

DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_REPO="$(cd "$DRIVER_DIR/../.." && pwd)"
cd "$SOURCE_REPO"
SCRATCH_BASE="${SCRATCH:-$GROUP_SCRATCH}/auto-psych"
PANEL_ROOT="${PANEL_ROOT:-$SCRATCH_BASE/review_panel/$NAME}"
[[ -e "$PANEL_ROOT" ]] && { echo "ERROR: $PANEL_ROOT exists; pick another name" >&2; exit 1; }

MEMBERS="${MEMBERS:-methods:codex:gpt-5.6-sol:methods search:claude:claude-fable-5-1:search crossloop:claude:claude-fable-5-1:crossloop}"
PROMPT_ONLY_BACKENDS="${PROMPT_ONLY_BACKENDS:-codex}"
if [[ -z "${MODERATOR:-}" ]]; then
  # The moderator checks contested claims in the files, so prefer a member
  # whose backend can read them.
  for m in $MEMBERS; do
    backend="$(echo "$m" | cut -d: -f2)"
    if [[ " $PROMPT_ONLY_BACKENDS " != *" $backend "* ]]; then MODERATOR="${m%%:*}"; break; fi
  done
  MODERATOR="${MODERATOR:-${MEMBERS%%:*}}"
fi
N_DISCUSSION_ROUNDS="${N_DISCUSSION_ROUNDS:-2}"
VERBAL_REPO="${VERBAL_REPO:-$HOME/llm-verbal-protocol}"
[[ -d "$VERBAL_REPO" ]] || { echo "ERROR: VERBAL_REPO $VERBAL_REPO does not exist" >&2; exit 1; }
EVIDENCE_FILES="${EVIDENCE_FILES:-src/pipelines/inner_loop/pymc_orchestrator.py src/pipelines/inner_loop/prompts/pymc_theory.md src/pipelines/inner_loop/prompts/critique.md src/model_comparison/posterior.py src/pipelines/outer_loop/eig.py scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml $VERBAL_REPO/src/cog_models/disco_loop/FUNCTIONAL_SPEC.md $VERBAL_REPO/src/cog_models/disco_loop/prompts/theorist.md $VERBAL_REPO/src/cog_models/disco_loop/prompts/critic.md $VERBAL_REPO/.claude/docs/alien-critique-analysis.md $VERBAL_REPO/.claude/docs/loop-improvements-2026-08-26.md}"
for f in $EVIDENCE_FILES; do
  [[ "$f" = /* ]] || f="$SOURCE_REPO/$f"
  [[ -f "$f" ]] || echo "WARNING: evidence file missing: $f" >&2
done

CAMPAIGN_ROOT=""
if [[ -n "${CAMPAIGN:-}" ]]; then
  CAMPAIGN_ROOT="$SCRATCH_BASE/recovery_improvement/$CAMPAIGN"
  [[ -f "$CAMPAIGN_ROOT/campaign.env" ]] || { echo "ERROR: no campaign at $CAMPAIGN_ROOT" >&2; exit 1; }
elif [[ -f "$SCRATCH_BASE/recovery_improvement/recovery_2026_09_07/campaign.env" ]]; then
  CAMPAIGN_ROOT="$SCRATCH_BASE/recovery_improvement/recovery_2026_09_07"
fi

if [[ $# -gt 0 ]]; then BASELINE_ROOTS=("$@"); else
  BASELINE_ROOTS=("$SCRATCH_BASE/holdout_faithful_32eig_32random" "$SCRATCH_BASE/holdout_faithful_64eig"
                  "$SCRATCH_BASE/holdout_faithful_64random" "$SCRATCH_BASE/holdout_faithful_test_retest_v8")
fi
for root in "${BASELINE_ROOTS[@]}"; do [[ -d "$root" ]] || { echo "ERROR: sweep root missing: $root" >&2; exit 1; }; done

# --- preflight: logins for every backend named ------------------------------------
if [[ "$MEMBERS" == *":claude:"* ]]; then
  [[ -f "$HOME/.claude/.credentials.json" ]] || echo "WARNING: claude is not logged in (~/.claude/.credentials.json missing)" >&2
fi
if [[ "$MEMBERS" == *":codex:"* ]]; then
  [[ -f "${CODEX_HOME:-$HOME/.codex}/auth.json" ]] \
    || echo "WARNING: codex is not logged in — run: ml load devel ${CODEX_MODULE:-codex/0.151.0} && codex login --device-auth" >&2
fi
CLAUDE_BIN_DIR=""; command -v claude >/dev/null 2>&1 && CLAUDE_BIN_DIR="$(dirname "$(command -v claude)")"

mkdir -p "$PANEL_ROOT/slurm_logs"
cat > "$PANEL_ROOT/panel.env" <<ENV
# Written by panel.sh on $(date '+%Y-%m-%d %H:%M:%S'); read by every stage job.
PANEL_NAME=$NAME
PANEL_ROOT=$PANEL_ROOT
SOURCE_REPO=$SOURCE_REPO
BASE_COMMIT=$(git rev-parse HEAD)
VERBAL_REPO=$VERBAL_REPO
CAMPAIGN_ROOT=$CAMPAIGN_ROOT
BASELINE_ROOTS="${BASELINE_ROOTS[*]}"
MEMBERS="$MEMBERS"
MODERATOR=$MODERATOR
PROMPT_ONLY_BACKENDS="$PROMPT_ONLY_BACKENDS"
EVIDENCE_FILES="$EVIDENCE_FILES"
EVIDENCE_MAX_BYTES=${EVIDENCE_MAX_BYTES:-300000}
N_DISCUSSION_ROUNDS=$N_DISCUSSION_ROUNDS
MEMBER_MAX_TURNS=${MEMBER_MAX_TURNS:-300}
MEMBER_TIMEOUT_SEC=${MEMBER_TIMEOUT_SEC:-5400}
MEMBER_MAX_BUDGET_USD=${MEMBER_MAX_BUDGET_USD:-100}
PANEL_PARTITION=${PANEL_PARTITION:-normal}
PANEL_TIME=${PANEL_TIME:-10:00:00}
PANEL_CPUS=${PANEL_CPUS:-4}
PANEL_MEM=${PANEL_MEM:-16GB}
PANEL_VENV=${PANEL_VENV:-}
DRIVER_SBATCH=$DRIVER_DIR/panel_round.sbatch
CLAUDE_BIN_DIR=$CLAUDE_BIN_DIR
CODEX_MODULE=${CODEX_MODULE:-codex/0.151.0}
ENV
set -a; source "$PANEL_ROOT/panel.env"; set +a

# --- the chain: round1 -> ... -> roundN -> synthesis ---------------------------------
stages=(); for ((k = 1; k <= 1 + N_DISCUSSION_ROUNDS; k++)); do stages+=("$k"); done; stages+=("synthesis")
prev=""
for stage in "${stages[@]}"; do
  args=(--parsable ${prev:+--dependency=afterany:$prev}
        --job-name="panel_$NAME" --partition="$PANEL_PARTITION" --time="$PANEL_TIME"
        --cpus-per-task="$PANEL_CPUS" --mem="$PANEL_MEM"
        --output="$PANEL_ROOT/slurm_logs/stage${stage}_%j.out" --error="$PANEL_ROOT/slurm_logs/stage${stage}_%j.out"
        --export=ALL,PANEL_ROOT="$PANEL_ROOT",STAGE="$stage" "$DRIVER_SBATCH")
  if [[ -n "${DRY_RUN:-}" ]]; then echo "would submit stage $stage: sbatch ${args[*]}"; prev="DRY"; continue; fi
  id=$(sbatch "${args[@]}")
  echo "submitted stage $stage: job $id${prev:+ (after $prev)}"
  echo "$stage $id" >> "$PANEL_ROOT/jobs.txt"
  prev="$id"
done

cat <<MSG

panel '$NAME' at $PANEL_ROOT
  members:   $MEMBERS
  moderator: $MODERATOR   discussion rounds: $N_DISCUSSION_ROUNDS   prompt-only backends: $PROMPT_ONLY_BACKENDS
  campaign:  ${CAMPAIGN_ROOT:-(none)}   verbal repo: $VERBAL_REPO
read:    $PANEL_ROOT/thread.md   then   $PANEL_ROOT/synthesis/plan.md
follow:  tail -f $PANEL_ROOT/slurm_logs/stage1_<jobid>.out
stop:    touch $PANEL_ROOT/STOP  (+ scancel the queued panel_$NAME jobs)
hand-off: PLAN=$PANEL_ROOT/synthesis/plan.md bash $DRIVER_DIR/review.sh <campaign>   # implement item 1
MSG
