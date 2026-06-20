#!/bin/bash
# ============================================================================
#  Launch a PILOT human experiment on Sherlock, configured from ONE file.
#
#  Everything is set in scripts/outer_loop_live/pilot.yaml (or CONFIG=<path>):
#  project, run label, #experiments, design mode, Slurm walltime/qos, the
#  Prolific study (participants/reward/length/name), and the modeling settings.
#  This launcher renders the project's prolific_config.yaml from that file,
#  shows a cost summary, asks you to confirm, then submits the live run.
#
#  THIS DEPLOYS TO FIREBASE AND PUBLISHES A PROLIFIC STUDY — it recruits REAL
#  humans and spends REAL money.
#
#  Usage:
#    bash scripts/outer_loop_live/run_pilot.sh
#    CONFIG=my_pilot.yaml bash scripts/outer_loop_live/run_pilot.sh
#    CONFIRM=yes bash scripts/outer_loop_live/run_pilot.sh     # skip the prompt
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTER_LIVE_SLURM_DIR="$DIR"
source "$DIR/_env.sh"

CONFIG="${CONFIG:-$DIR/pilot.yaml}"

# --- preflight: fail BEFORE anything is deployed or published --------------
fail() { echo "PREFLIGHT FAILED: $*" >&2; exit 1; }
[[ -f "$CONFIG" ]]                       || fail "config not found: $CONFIG"
[[ -x "$VENV_PY" ]]                      || fail "venv not built — run once: sbatch $DIR/setup.sbatch"
[[ -n "${FIREBASE_TOKEN:-}" ]]           || fail "FIREBASE_TOKEN missing in .secrets (run: firebase login:ci)"
[[ -n "${PROLIFIC_API_TOKEN:-}" ]]       || fail "PROLIFIC_API_TOKEN missing in .secrets"
[[ -f "$REPO/templates/consent.txt" ]]   || fail "no IRB consent text at templates/consent.txt"

cd "$REPO"

# Parse pilot.yaml: prints the cost summary + Prolific token check (stderr) and
# emits run/modeling env (stdout). --check = validate only, do NOT render
# prolific_config yet (we render into the per-run worktree below, leaving your
# main checkout untouched). Aborts here on any validation/token error.
if ! PILOT_ENV="$("$VENV_PY" "$DIR/_pilot_config.py" "$CONFIG" --check)"; then
  exit 1
fi
eval "$PILOT_ENV"   # PROJECT RUN_LABEL N_EXPERIMENTS DESIGN_MODE FIREBASE_PROJECT WALLTIME QOS N_PARTICIPANTS DRAWS TUNE CHAINS INNER_LOOP_*

EXP_BASE="https://${FIREBASE_PROJECT}.web.app"
echo
if (( N_EXPERIMENTS > 1 )); then
  echo "  experiment URLs   : ${EXP_BASE}/e{1..$N_EXPERIMENTS}-${RUN_LABEL}/"
  echo "  NOTE: $N_EXPERIMENTS studies run in SEQUENCE — make sure walltime ($WALLTIME${QOS:+, qos=$QOS})"
  echo "        covers ~${N_EXPERIMENTS}x (deploy + up to 2h recruiting + modeling)."
else
  echo "  experiment URL    : ${EXP_BASE}/e1-${RUN_LABEL}/"
fi
echo "  output dir        : $WORK_ROOT/$RUN_LABEL/data"
echo

# --- explicit confirmation (mode-aware) ------------------------------------
case "${PROLIFIC_MODE:-test}" in
  live) PROMPT='LIVE: publish a REAL study and recruit + PAY participants. Type "yes" to proceed: ' ;;
  test) PROMPT='TEST: create a DRAFT study (NOT published) + deploy, to preview yourself. Type "yes": ' ;;
  *)    PROMPT='Deploy the experiment (no Prolific study). Type "yes" to proceed: ' ;;
esac
if [[ "${CONFIRM:-}" != "yes" ]]; then
  read -r -p "$PROMPT" reply
  [[ "$reply" == "yes" ]] || { echo "Aborted — nothing was deployed or published."; exit 1; }
fi

# --- per-run isolated checkout (rsync copy) --------------------------------
# Coding agents — opencode especially — keep per-project working state; parallel
# runs must NOT share a working directory (its public/, firebase.generated.json,
# opencode.json). Give this run its OWN copy of the repo on $SCRATCH. We rsync
# (not `git worktree`: el7's git is too old for it) so the copy captures your
# CURRENT working tree — no commit required. Excludes .git/.secrets/data/caches;
# code + assets come along, and we render the study config into the copy.
WT="$WORK_ROOT/runs/$RUN_LABEL/repo"; mkdir -p "$WT"
rsync -a --delete \
  --exclude '.git' --exclude '.secrets' --exclude '.venv' --exclude 'data' \
  --exclude '__pycache__' --exclude '*.nc' --exclude 'scratch' --exclude '.worktrees' \
  --exclude 'public' --exclude 'firebase.generated.json' --exclude 'functions/node_modules' \
  --exclude '.uv_cache' --exclude '.pip_cache' --exclude '.cache' --exclude '.hf' \
  "$REPO"/ "$WT"/
touch "$WT/.here"   # pyprojroot sentinel (.git is excluded from the copy)
# Render the study config into the COPY from your current pilot.yaml (leaves your
# main checkout untouched).
"$VENV_PY" "$WT/scripts/outer_loop_live/_pilot_config.py" "$CONFIG" --render-only \
  || fail "failed to render prolific_config into the run copy"

# --- assemble + submit the live-run job ------------------------------------
LOGDIR="$WORK_ROOT/slurm_logs"; mkdir -p "$LOGDIR"
OUT="$WORK_ROOT/$RUN_LABEL/data"; mkdir -p "$OUT"

EXPORTS="ALL,RUN_LABEL=$RUN_LABEL,RUN_WORKTREE=$WT,PROJECT=$PROJECT,DESIGN_MODE=$DESIGN_MODE,CODING_AGENT=$CODING_AGENT,PROLIFIC_MODE=$PROLIFIC_MODE,FIREBASE_PROJECT=$FIREBASE_PROJECT,N_PARTICIPANTS=$N_PARTICIPANTS,AUTO_PSYCH_OUTPUT_DIR=$OUT"
if (( N_EXPERIMENTS > 1 )); then EXPORTS="$EXPORTS,EXPERIMENTS=1-$N_EXPERIMENTS"; else EXPORTS="$EXPORTS,EXPERIMENT=1"; fi
add() { [[ -n "${2:-}" ]] && EXPORTS="$EXPORTS,$1=$2" || true; }
add DRAWS "${DRAWS:-}"
add TUNE "${TUNE:-}"
add CHAINS "${CHAINS:-}"
add INNER_LOOP_ITERATIONS "${INNER_LOOP_ITERATIONS:-}"
add INNER_LOOP_CANDIDATES "${INNER_LOOP_CANDIDATES:-}"

QOS_OPT=""; [[ -n "${QOS:-}" ]] && QOS_OPT="--qos=$QOS"

jid=$(sbatch --parsable \
  --job-name="pilot_${RUN_LABEL}" --time="$WALLTIME" $QOS_OPT \
  --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
  --export="$EXPORTS" \
  "$DIR/run_live.sbatch")

echo
echo "Pilot submitted: job $jid  (N=$N_PARTICIPANTS participants x $N_EXPERIMENTS experiment(s))"
echo "  monitor : squeue --me ; tail -f $LOGDIR/pilot_${RUN_LABEL}_${jid}.out"
echo "  data    : $OUT/$PROJECT/experiment<k>/data/responses.csv"
echo "  run copy: $WT   (isolated cwd; remove when done: rm -rf $WORK_ROOT/runs/$RUN_LABEL)"
echo "  studies : one Prolific study per experiment appears in your dashboard (Draft -> Published)"
echo
echo "TO STOP THE PILOT:"
echo "  1. Stop/pause the study (EACH study, if >1) in the Prolific dashboard <-- this is"
echo "     what stops recruiting and charging. scancel alone does NOT stop a published study."
echo "  2. scancel $jid   (stops the pipeline job: polling + modeling)"
