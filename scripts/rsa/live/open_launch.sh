#!/bin/bash
# Launch chains of the open campaign, either arm, on Sherlock (docs/auto_open/PLAN.md Phase 3b,
# items 6-7; scripts/rsa/slurm/open_live.sbatch runs each chain).
#
#   PROLIFIC_MODE=test ARM=open CHAINS="0" bash scripts/rsa/live/open_launch.sh      # page + Prolific draft, then stop
#   PROLIFIC_MODE=live ARM=open CHAINS="0" PARTITION=mcfrank TIME=7-00:00:00 LINKS=1 bash scripts/rsa/live/open_launch.sh
#                                                                                       # the feasibility chain
#   PROLIFIC_MODE=live ARM=fixed CHAINS="0 1 2 3" bash scripts/rsa/live/open_launch.sh   # on a pass (normal partition)
#   PROLIFIC_MODE=live ARM=open CHAINS="0" PARTICIPANTS=15 SMOKE_FROM=<run>/experiment1 \
#     bash scripts/rsa/live/open_launch.sh                                               # a smoke test: never modelled
#
# Checks the config (scripts/rsa/live/open_live.yaml) and shows what the launch recruits and costs.
# A live launch also needs confirm_live_recruitment: true in the config, a typed "yes" here, and
# (except a smoke test) the checkout at the config's frozen code_tag, clean; only then is
# CONFIRM_LIVE_RECRUITMENT=1 passed to the jobs.
#
# Each chain gets a ladder of LINKS jobs (default 4 x 48 h on the first of mcfrank, owners, normal with room), each starting
# after the previous one ends (afterany): a job that runs out of time leaves the chain for the
# next, which resumes it; one that stops (finished, awaiting review, a test deploy, a failure)
# writes the cell's STOP, and the ladder's later jobs exit at once. A stopped chain is relaunched
# only with CLEAR_STOP=1 (its STOP is kept as STOP.<time>).
set -euo pipefail
REPO="${REPO:-$HOME/auto-psych}"
CONFIG="${CONFIG:-$REPO/scripts/rsa/live/open_live.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live}"
ARM="${ARM:?set ARM=open or fixed}"
read -r -a CHAIN_LIST <<< "${CHAINS:?set CHAINS, e.g. \"0\" or \"0 1 2 3\"}"
WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_live}"
PARTITION="${PARTITION:-mcfrank,owners,normal}"  # the first with room (run D1: a whole node waited hours on normal)
TIME="${TIME:-48:00:00}"
LINKS="${LINKS:-4}"
N_EXPERIMENTS="${N_EXPERIMENTS:-4}"
source "$REPO/scripts/rsa/slurm/_env.sh"
cd "$REPO"
SMOKE=""
SIZE=()
if [[ -n "${PARTICIPANTS:-}" ]]; then
  SMOKE=_smoke
  SIZE=(--cpus-per-task=4 --mem=16G)  # a smoke test deploys and collects: no fits (run C′'s report)
fi
"$VENV_PY" -m src.rsa_open.live --config "$CONFIG" --check --arm "$ARM" --chains "${CHAIN_LIST[@]}" \
  --n-experiments "$([[ -n "$SMOKE" ]] && echo 1 || echo "$N_EXPERIMENTS")" ${PARTICIPANTS:+--participants "$PARTICIPANTS"}
echo "mode: $PROLIFIC_MODE${SMOKE:+ (smoke test: $PARTICIPANTS people, never modelled)}; $LINKS job(s) of $TIME on $PARTITION per chain" >&2

EXTRA_ENV=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  tag="$("$VENV_PY" -c 'import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get("code_tag") or "")' "$CONFIG")"
  code="$(bash scripts/subjective_randomness/slurm/code_commit.sh "$REPO")"
  if [[ -z "$SMOKE" ]]; then
    [[ -n "$tag" ]] || { echo "not launched: a live chain runs on a frozen tag (code_tag in $CONFIG)" >&2; exit 1; }
    [[ "$code" == "$(git rev-parse "$tag^{commit}")" ]] || {
      echo "not launched: the checkout ($code) is not the frozen tag $tag, clean" >&2; exit 1; }
    echo "code: $tag ($code)" >&2
  else
    echo "code: $code (a smoke test needs no frozen tag)" >&2
  fi
  echo "This PUBLISHES Prolific studies: real people, real money. Scancel does not stop a published study." >&2
  read -r -p 'Type "yes" to launch: ' answer
  [[ "$answer" == "yes" ]] || { echo "not launched" >&2; exit 1; }
  EXTRA_ENV+=(CONFIRM_LIVE_RECRUITMENT=1)
fi
mkdir -p "$WORK_ROOT/logs"
for chain in "${CHAIN_LIST[@]}"; do
  cell="$WORK_ROOT/cells/${ARM}_chain_$chain$SMOKE"
  if [[ -f "$cell/STOP" ]]; then
    if [[ "${CLEAR_STOP:-}" == "1" ]]; then
      mv "$cell/STOP" "$cell/STOP.$(date +%Y%m%dT%H%M%S)"
      echo "[launch] ${ARM}_chain_$chain$SMOKE: cleared its STOP" >&2
    else
      echo "[launch] ${ARM}_chain_$chain$SMOKE is stopped ($(cat "$cell/STOP")); relaunch with CLEAR_STOP=1" >&2
      continue
    fi
  fi
  dep=()
  for ((link = 1; link <= LINKS; link++)); do
    id="$(env ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"} PROLIFIC_MODE="$PROLIFIC_MODE" CONFIG="$CONFIG" WORK_ROOT="$WORK_ROOT" \
          ARM="$ARM" CHAIN="$chain" N_EXPERIMENTS="$N_EXPERIMENTS" \
          ${PARTICIPANTS:+PARTICIPANTS="$PARTICIPANTS"} ${SMOKE_FROM:+SMOKE_FROM="$SMOKE_FROM"} \
          sbatch --parsable --chdir="$REPO" --partition="$PARTITION" --time="$TIME" ${SIZE[@]+"${SIZE[@]}"} \
            --job-name="rsa_open_${ARM}_c$chain$SMOKE" ${dep[@]+"${dep[@]}"} \
            -o "$WORK_ROOT/logs/%x_%j.out" scripts/rsa/slurm/open_live.sbatch)"
    echo "[launch] ${ARM}_chain_$chain$SMOKE: job $link of $LINKS: $id" >&2
    dep=(--dependency="afterany:$id")
  done
done
