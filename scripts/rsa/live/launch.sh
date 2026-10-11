#!/bin/bash
# Launch the RSA live campaign (or a test deploy, or a pilot) on Sherlock.
#
#   PROLIFIC_MODE=test bash scripts/rsa/live/launch.sh               # deploy + Prolific drafts to preview
#   PROLIFIC_MODE=live PILOT=1 PARTICIPANTS=20 CHAINS_TO_RUN=0 bash scripts/rsa/live/launch.sh
#   PROLIFIC_MODE=live bash scripts/rsa/live/launch.sh               # the campaign: every chain
#   PROLIFIC_MODE=live N_EXPERIMENTS=1 bash scripts/rsa/live/launch.sh   # experiment 1 of every chain first
#
# Checks the config and shows what the launch recruits and costs. A live launch
# also needs confirm_live_recruitment: true in the config, and a typed "yes" here;
# only then is CONFIRM_LIVE_RECRUITMENT=1 passed to the job. CHAINS_TO_RUN picks
# array tasks (indices into the config's chains; default all).
set -euo pipefail
REPO="${REPO:-$HOME/auto-psych}"
CONFIG="${CONFIG:-$REPO/scripts/rsa/live/rsa_live.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live}"
WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_live}"
source "$REPO/scripts/rsa/slurm/_env.sh"
cd "$REPO"
"$VENV_PY" -m src.rsa.live.config --config "$CONFIG" --check
n_chains="$("$VENV_PY" -c 'import sys, yaml; print(len(yaml.safe_load(open(sys.argv[1]))["chains"]))' "$CONFIG")"
ARRAY="${CHAINS_TO_RUN:-0-$((n_chains - 1))}"
echo "mode: $PROLIFIC_MODE${PILOT:+ (pilot: stops once experiment 1 data are in)}${PARTICIPANTS:+, $PARTICIPANTS participants per study}; array $ARRAY" >&2
if [[ -n "${N_EXPERIMENTS:-}" ]]; then
  # A campaign run experiment by experiment (PI 2026-10-11): resubmitting with a
  # larger N_EXPERIMENTS continues each chain in place.
  "$VENV_PY" -c 'import sys
from src.rsa.live.config import cost, load
c = cost(load(sys.argv[1])); chains = len(sys.argv[3].replace("-", ",").split(",")) if "-" not in sys.argv[3] else int(sys.argv[3].split("-")[1]) - int(sys.argv[3].split("-")[0]) + 1
print(f"this launch    : experiments 1-{sys.argv[2]} of each chain, {chains} chain(s): up to {chains * int(sys.argv[2])} studies, ~${chains * int(sys.argv[2]) * c[\"per_study_cents\"] / 100:,.2f} (finished experiments are not rerun)", file=sys.stderr)' \
    "$CONFIG" "$N_EXPERIMENTS" "$ARRAY"
fi
EXTRA_ENV=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  echo "This PUBLISHES Prolific studies: real people, real money. Scancel does not stop a published study." >&2
  read -r -p 'Type "yes" to launch: ' answer
  [[ "$answer" == "yes" ]] || { echo "not launched" >&2; exit 1; }
  EXTRA_ENV+=(CONFIRM_LIVE_RECRUITMENT=1)
fi
mkdir -p "$WORK_ROOT/logs"
env ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} PROLIFIC_MODE="$PROLIFIC_MODE" CONFIG="$CONFIG" WORK_ROOT="$WORK_ROOT" \
  sbatch --chdir="$REPO" --array="$ARRAY" -o "$WORK_ROOT/logs/%x_%A_%a.out" scripts/rsa/slurm/outer_live.sbatch
