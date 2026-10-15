#!/bin/bash
# Launch the pre-registered final experiment (docs/auto_rsa/PREREG_FINAL.md) on Sherlock.
#
#   PROLIFIC_MODE=test bash scripts/rsa/live/prereg_launch.sh   # deploy + a Prolific draft to preview
#   PROLIFIC_MODE=live bash scripts/rsa/live/prereg_launch.sh   # publish: real people, real money
#
# Checks the config and shows what the launch recruits and costs; a live launch also
# needs confirm_live_recruitment: true in the config and a typed "yes" here.
set -euo pipefail
REPO="${REPO:-$HOME/auto-psych}"
CONFIG="${CONFIG:-$REPO/scripts/rsa/live/rsa_prereg.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live}"
WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
cd "$REPO"
"$VENV_PY" -m src.rsa.live.config --config "$CONFIG" --check
EXTRA_ENV=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  echo "This PUBLISHES a Prolific study: real people, real money. Scancel does not stop a published study." >&2
  read -r -p 'Type "yes" to launch: ' answer
  [[ "$answer" == "yes" ]] || { echo "not launched" >&2; exit 1; }
  EXTRA_ENV+=(CONFIRM_LIVE_RECRUITMENT=1)
fi
mkdir -p "$WORK_ROOT/logs"
env ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} PROLIFIC_MODE="$PROLIFIC_MODE" CONFIG="$CONFIG" WORK_ROOT="$WORK_ROOT" \
  sbatch --chdir="$REPO" -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/prereg_live.sbatch
