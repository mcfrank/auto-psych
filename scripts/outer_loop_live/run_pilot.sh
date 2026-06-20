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

# --- explicit confirmation (real humans + money) ---------------------------
if [[ "${CONFIRM:-}" != "yes" ]]; then
  read -r -p 'Publish this study and recruit participants? Type "yes" to proceed: ' reply
  [[ "$reply" == "yes" ]] || { echo "Aborted — nothing was deployed or published."; exit 1; }
fi

# --- per-run worktree (isolated cwd) ---------------------------------------
# Coding agents — opencode especially — keep per-project working state; parallel
# runs must NOT share a working directory (its public/, firebase.generated.json,
# opencode.json). Give this run its own git worktree. Worktrees check out HEAD,
# so the run uses your COMMITTED code + config; render the study config into the
# worktree from your (committed) pilot.yaml. Commit before launching.
if [[ -n "$(git -C "$REPO" status --porcelain)" ]]; then
  fail "uncommitted changes in $REPO — the run uses a worktree checked out at HEAD. Commit pilot.yaml + code first (then re-run)."
fi
WT="$WORK_ROOT/worktrees/$RUN_LABEL"; mkdir -p "$WORK_ROOT/worktrees"
[[ -d "$WT" ]] || git -C "$REPO" worktree add --detach "$WT" HEAD
"$VENV_PY" "$WT/scripts/outer_loop_live/_pilot_config.py" "$CONFIG" --render-only \
  || fail "failed to render prolific_config into the worktree"

# --- assemble + submit the live-run job ------------------------------------
LOGDIR="$WORK_ROOT/slurm_logs"; mkdir -p "$LOGDIR"
OUT="$WORK_ROOT/$RUN_LABEL/data"; mkdir -p "$OUT"

EXPORTS="ALL,RUN_LABEL=$RUN_LABEL,RUN_WORKTREE=$WT,PROJECT=$PROJECT,DESIGN_MODE=$DESIGN_MODE,CODING_AGENT=$CODING_AGENT,FIREBASE_PROJECT=$FIREBASE_PROJECT,N_PARTICIPANTS=$N_PARTICIPANTS,AUTO_PSYCH_OUTPUT_DIR=$OUT"
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
echo "  worktree: $WT   (isolated cwd; remove when done: git -C $REPO worktree remove $WT)"
echo "  studies : one Prolific study per experiment appears in your dashboard (Draft -> Published)"
echo
echo "TO STOP THE PILOT:"
echo "  1. Stop/pause the study (EACH study, if >1) in the Prolific dashboard <-- this is"
echo "     what stops recruiting and charging. scancel alone does NOT stop a published study."
echo "  2. scancel $jid   (stops the pipeline job: polling + modeling)"
