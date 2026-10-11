#!/bin/bash
# Stage the RSA sweep's code once: the harness repo and the agent source, then
# record the staged code's identity in $WORK_ROOT/code_commit (written last,
# so a setup that died mid-way stages again).
#
#   REPO=<checkout> WORK_ROOT=<sweep root> bash stage_code.sh
#
# Mirrors scripts/subjective_randomness/slurm/stage_sweep_code.sh (and reuses
# its code_commit.sh), which cannot be called as is: it builds agent_src with
# the subjective-randomness exclude list and guards that sweep's cell layout
# (run<r>/<gt>/). Every job after setup runs on these copies, never on the
# live checkout; with code_commit present this stages nothing and fails if the
# checkout's code differs from what was staged, unless RESTAGE_REASON names the
# logged bug fix being applied (re-staged from a clean checkout, recorded in
# $WORK_ROOT/code_history; a frozen campaign changes code only this way).
#
#   $WORK_ROOT/harness_repo  full copy (no .git, no data/, no .secrets); the
#                            loop, simulation and scoring run from here
#   $WORK_ROOT/agent_src     harness_repo scrubbed with agent_tree.exclude;
#                            each cell builds its agent tree from it
set -euo pipefail
: "${REPO:?REPO is not set}"
: "${WORK_ROOT:?WORK_ROOT is not set}"
HARNESS_REPO="${HARNESS_REPO:-$WORK_ROOT/harness_repo}"
AGENT_SRC="${AGENT_SRC:-$WORK_ROOT/agent_src}"
CODE_COMMIT_FILE="$WORK_ROOT/code_commit"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_code=$(bash "$here/../../subjective_randomness/slurm/code_commit.sh" "$REPO")

RESTAGING=""
if [[ -f "$CODE_COMMIT_FILE" ]]; then
  staged_code=$(cat "$CODE_COMMIT_FILE")
  if [[ "$staged_code" == "$repo_code" ]]; then
    echo "[stage] code already staged for this sweep ($staged_code); not re-staging"
    exit 0
  fi
  if [[ -z "${RESTAGE_REASON:-}" ]]; then
    echo "ERROR: $WORK_ROOT was staged from code $staged_code, but $REPO is now at $repo_code." >&2
    echo "       Its cells must not run on other code. Check out $staged_code, or use a new WORK_ROOT," >&2
    echo "       or apply logged bug fixes deliberately: RESTAGE_REASON='<the campaign-log entry>'." >&2
    exit 1
  fi
  # A frozen campaign takes logged bug fixes only through here (PI 2026-10-11):
  # from a clean checkout, recorded with the reason in code_history.
  [[ "$repo_code" != *-dirty-* ]] || { echo "ERROR: re-staging needs a clean checkout, not $repo_code" >&2; exit 1; }
  printf '%s\t%s -> %s\t%s\n' "$(date -Is)" "$staged_code" "$repo_code" "$RESTAGE_REASON" >> "$WORK_ROOT/code_history"
  echo "[stage] RE-STAGING $staged_code -> $repo_code: $RESTAGE_REASON"
  RESTAGING=1
  rm -f "$CODE_COMMIT_FILE"
fi
if [[ -z "$RESTAGING" ]] && compgen -G "$WORK_ROOT/cells/*/" >/dev/null; then
  echo "ERROR: $WORK_ROOT has cells but no record of the code they ran on ($CODE_COMMIT_FILE). Use a new WORK_ROOT." >&2
  exit 1
fi

# A re-stage compares contents, not size and time: a fixed file of the same size
# written within the staged one's second would otherwise keep the old code.
COMPARE=()
[[ -n "$RESTAGING" ]] && COMPARE=(--checksum)
echo "[stage] staging harness repo -> $HARNESS_REPO"
mkdir -p "$HARNESS_REPO"
rsync -a ${COMPARE[@]+"${COMPARE[@]}"} --delete --exclude '.git' --exclude '.venv' --exclude '__pycache__' \
  --exclude '*.nc' --exclude 'node_modules' --exclude '/data/' \
  --exclude '.secrets*' --exclude '*.secrets' --exclude '.xdg_data' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  "$REPO"/ "$HARNESS_REPO"/
touch "$HARNESS_REPO/.here"  # pyprojroot sentinel (.git is not copied)

echo "[stage] staging agent source -> $AGENT_SRC"
mkdir -p "$AGENT_SRC"
rsync -a ${COMPARE[@]+"${COMPARE[@]}"} --delete --delete-excluded \
  --exclude-from="$HARNESS_REPO/scripts/rsa/slurm/agent_tree.exclude" \
  "$HARNESS_REPO"/ "$AGENT_SRC"/

printf '%s\n' "$repo_code" > "$CODE_COMMIT_FILE.tmp"
mv "$CODE_COMMIT_FILE.tmp" "$CODE_COMMIT_FILE"
echo "[stage] staged code $repo_code"
