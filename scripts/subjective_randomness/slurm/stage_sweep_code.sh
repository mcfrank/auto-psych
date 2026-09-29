#!/bin/bash
# Stage a sweep's code once: the harness repo and the agent source, then record
# the staged code's identity in $WORK_ROOT/code_commit (written last, so a
# setup that died mid-way stages again).
#
#   REPO=<checkout> WORK_ROOT=<sweep root> bash stage_sweep_code.sh
#
# Every job after setup (the array, its retries, their analysis) runs on these
# copies, never on the live checkout: retries used to re-run setup days later,
# re-staging whatever $REPO held by then under cells that had started on other
# code. With code_commit present this stages nothing and fails if the
# checkout's code (code_commit.sh) differs from what was staged. It refuses a
# WORK_ROOT that has cells but no code_commit: those cells started before code
# was staged once per sweep.
#
#   $WORK_ROOT/harness_repo  full copy; the harness process and the Slurm
#                            scripts of every later job run from here
#   $WORK_ROOT/agent_src     the repo scrubbed with agent_tree.exclude; each
#                            cell builds its agent tree from it
set -euo pipefail
: "${REPO:?REPO is not set}"
: "${WORK_ROOT:?WORK_ROOT is not set}"
HARNESS_REPO="${HARNESS_REPO:-$WORK_ROOT/harness_repo}"
AGENT_SRC="${AGENT_SRC:-$WORK_ROOT/agent_src}"
CODE_COMMIT_FILE="$WORK_ROOT/code_commit"
repo_code=$(bash "$REPO/scripts/subjective_randomness/slurm/code_commit.sh" "$REPO")

if [[ -f "$CODE_COMMIT_FILE" ]]; then
  staged_code=$(cat "$CODE_COMMIT_FILE")
  if [[ "$staged_code" != "$repo_code" ]]; then
    echo "ERROR: $WORK_ROOT was staged from code $staged_code, but $REPO is now at $repo_code." >&2
    echo "       Its cells must not resume on other code. Check out $staged_code, or use a new WORK_ROOT." >&2
    exit 1
  fi
  echo "[stage] code already staged for this sweep ($staged_code); not re-staging"
  exit 0
fi
if compgen -G "$WORK_ROOT/run*/*/" >/dev/null; then
  echo "ERROR: $WORK_ROOT has cells but no record of the code they ran on ($CODE_COMMIT_FILE);" >&2
  echo "       they started before code was staged once per sweep. Resume them with the scripts they started under, or use a new WORK_ROOT." >&2
  exit 1
fi

echo "[stage] staging harness repo -> $HARNESS_REPO"
mkdir -p "$HARNESS_REPO"
rsync -a --delete --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.nc' --exclude 'node_modules' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  "$REPO"/ "$HARNESS_REPO"/
touch "$HARNESS_REPO/.here"  # pyprojroot sentinel (.git is not copied)

echo "[stage] staging agent source -> $AGENT_SRC"
mkdir -p "$AGENT_SRC"
rsync -a --delete \
  --exclude-from="$HARNESS_REPO/scripts/subjective_randomness/slurm/agent_tree.exclude" \
  "$REPO"/ "$AGENT_SRC"/

printf '%s\n' "$repo_code" > "$CODE_COMMIT_FILE.tmp"
mv "$CODE_COMMIT_FILE.tmp" "$CODE_COMMIT_FILE"
echo "[stage] staged code $repo_code"
