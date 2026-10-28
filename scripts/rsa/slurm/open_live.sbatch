#!/bin/bash
#SBATCH --job-name=rsa_open_live
#SBATCH --partition=normal
#SBATCH --cpus-per-task=24
#SBATCH --mem=96GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%j.out
#
# One chain of the open campaign with real participants (docs/auto_open/PLAN.md Phase 3b, items 6
# and 7; PREREG_DRAFT.md): ARM (open | fixed) and CHAIN (0-3) pick the chain; each experiment is
# design -> deploy the page to the chain's Hosting site -> Prolific -> collect -> inner loop
# (src.rsa_open.run_chain --collection live; src.rsa_open.live). Submit it through
# scripts/rsa/live/open_launch.sh, which checks the config and the frozen code tag, shows the cost,
# asks for a typed "yes" before a live launch, and submits a ladder of jobs per chain.
#
#   PROLIFIC_MODE=test: deploy experiment 1's page and a Prolific DRAFT to preview, then stop.
#   PROLIFIC_MODE=live: publish (real money). Needs confirm_live_recruitment: true in the config
#     AND CONFIRM_LIVE_RECRUITMENT=1. An open chain's experiment 1 is published only once the PI
#     has approved its paradigm (python -m src.rsa_open.review).
#   PARTICIPANTS=<n> SMOKE_FROM=<another run's experiment dir>: a smoke test (cell <arm>_chain_<k>_smoke):
#     that experiment's frozen design, recruited at n people, collected, never modelled.
#
# How a job ends (the ladder reads it): the chain finished (DONE, and STOP so the ladder's later
# jobs exit at once); it stopped for the PI's review or after a test deploy (STOP, with the
# reason); too little of the job was left for its next stage (no STOP: the next job resumes; a
# 48-hour job never starts a stage it cannot finish, src.rsa_open.chain.STAGE_MIN_HOURS); or it
# failed (STOP with the error: a person looks before it resumes, PREREG_DRAFT.md §4a). A job killed
# by its time limit or a lost node leaves no STOP, and the next job resumes it: finished stages
# are skipped, a recorded study is never deployed or published again (main's guard), and a design
# or loop that stopped part way starts again (its attempt kept as <stage>.abandoned_<i>).
#
# One study recruits at a time across the campaign (a lock in $WORK_ROOT/private/campaign), so the
# chains' studies interleave in the order they are ready. Participant ids are the campaign's; the
# raw responses (Prolific ids) stay in $CELL_DIR/private and the campaign directory, on Sherlock.
#
# Overrides: CONFIG (scripts/rsa/live/open_live.yaml), N_EXPERIMENTS (4), MAX_ITERATIONS (5),
# CANDIDATES (6), CV_FOLDS (5), PILOT_PARTICIPANTS (12), THEORIST_TIMEOUT_SEC (3600), SEED (0),
# RUN2_ROOT (E's source), CODE_TAG (the frozen tag the staged code must be).
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_open_live}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }
command -v node >/dev/null && command -v npm >/dev/null || { echo "FATAL: node/npm needed to deploy (ml load nodejs)" >&2; exit 1; }
for v in PROLIFIC_API_TOKEN AUTO_PSYCH_RESULTS_TOKEN FIREBASE_TOKEN; do
  [[ -n "${!v:-}" ]] || { echo "FATAL: $v is not set (.secrets)" >&2; exit 1; }
done
(cd "$REPO" && "$VENV_PY" -m src.rsa_open.env_check) || {
  echo "FATAL: the venv cannot run the open campaign; on a login node: source scripts/rsa/slurm/_env.sh &&" \
       "source scripts/_venv.sh && build_shared_venv" >&2; exit 1; }

ARM="${ARM:?set ARM=open or fixed}"
case "$ARM" in open|fixed) ;; *) echo "FATAL: ARM must be open or fixed, not $ARM" >&2; exit 1 ;; esac
CHAIN="${CHAIN:?set CHAIN (0-3)}"
CONFIG="${CONFIG:-$REPO/scripts/rsa/live/open_live.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live (use scripts/rsa/live/open_launch.sh)}"
SMOKE=""
if [[ -n "${PARTICIPANTS:-}" ]]; then
  [[ -n "${SMOKE_FROM:-}" ]] || { echo "FATAL: a smoke test (PARTICIPANTS) needs SMOKE_FROM, a frozen design to field" >&2; exit 1; }
  SMOKE=_smoke
fi
CELL="${ARM}_chain_$CHAIN$SMOKE"
CELL_DIR="$CELLS_ROOT/$CELL"
mkdir -p "$CELL_DIR/private"
if [[ -f "$CELL_DIR/STOP" ]]; then
  echo "[open live] $CELL is stopped; not resuming: $(cat "$CELL_DIR/STOP")"
  exit 0
fi
stop() { printf '%s\t%s\n' "$(date -Is)" "$1" > "$CELL_DIR/STOP"; echo "[open live] STOP: $1"; }

cfg() { "$VENV_PY" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); v = c.get(sys.argv[2]); print("" if v is None else v)' "$CONFIG" "$1"; }
CONFIRM_FLAG=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  [[ "$(cfg confirm_live_recruitment)" == "True" ]] || { echo "FATAL: live needs confirm_live_recruitment: true in $CONFIG" >&2; exit 1; }
  [[ "${CONFIRM_LIVE_RECRUITMENT:-}" == "1" ]] || { echo "FATAL: live needs CONFIRM_LIVE_RECRUITMENT=1" >&2; exit 1; }
  CONFIRM_FLAG=(--confirm-live-recruitment)
elif [[ "$PROLIFIC_MODE" != "test" ]]; then
  echo "FATAL: PROLIFIC_MODE must be test or live, not $PROLIFIC_MODE" >&2; exit 1
fi
N_EXPERIMENTS="${N_EXPERIMENTS:-4}"
MAX_ITERATIONS="${MAX_ITERATIONS:-5}"
CANDIDATES="${CANDIDATES:-6}"
CV_FOLDS="${CV_FOLDS:-5}"
PILOT_PARTICIPANTS="${PILOT_PARTICIPANTS:-12}"
THEORIST_TIMEOUT_SEC="${THEORIST_TIMEOUT_SEC:-3600}"
SEED="${SEED:-0}"
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"
EXISTING="$RUN2_ROOT/promote_work/all_trials.csv"   # E: every literature trial
SEEDS_SRC="$REPO/data/rsa_open/seeds"
for f in "$EXISTING" "$SEEDS_SRC/seeds.json" "$SEEDS_SRC/bar/models_manifest.yaml"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing (run promote.sbatch; pull auto-psych-open)" >&2; exit 1; }
done
CAMPAIGN_DIR="$WORK_ROOT/private/campaign"
mkdir -p "$CAMPAIGN_DIR"

# --- once per campaign: staged code (the frozen tag), the seeds, the Prolific settings -------
(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  [[ -d "$WORK_ROOT/open_seeds" ]] || { cp -r "$SEEDS_SRC" "$WORK_ROOT/open_seeds.tmp" && mv "$WORK_ROOT/open_seeds.tmp" "$WORK_ROOT/open_seeds"; }
  (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa_open.live --config "$CONFIG" --render)
) 9>"$WORK_ROOT/.setup.lock"
CODE_TAG="${CODE_TAG:-$(cfg code_tag)}"
if [[ -n "$CODE_TAG" ]]; then
  tag_commit="$(cd "$REPO" && git rev-parse "$CODE_TAG^{commit}")"
  [[ "$(cat "$WORK_ROOT/code_commit")" == "$tag_commit" ]] || {
    echo "FATAL: the staged code ($(cat "$WORK_ROOT/code_commit")) is not the frozen tag $CODE_TAG ($tag_commit)" >&2; exit 1; }
elif [[ "$PROLIFIC_MODE" == "live" && -z "$SMOKE" ]]; then
  echo "FATAL: a live chain runs on a frozen tag: set code_tag in $CONFIG (or CODE_TAG)" >&2; exit 1
fi
HARNESS_REPO="$WORK_ROOT/harness_repo"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"
[[ -s "$CELL_DIR/code_commit" ]] || cp "$WORK_ROOT/code_commit" "$CELL_DIR/code_commit"

# --- the cell: its deploy copy and its agent tree -------------------------------------------
DEPLOY_REPO="$CELL_DIR/deploy_repo"
rsync -a --delete --exclude .git --exclude /public --exclude /data --exclude 'node_modules/' \
  "$HARNESS_REPO"/ "$DEPLOY_REPO"/
"$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout "$REPO" --copy "$DEPLOY_REPO"
[[ -s "$CELL_DIR/agent_tree_id" ]] || head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CELL_DIR/agent_tree_id"
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
RUN_DIR="$RUN_REPO/_runs/chain_$CHAIN"
mkdir -p "$RUN_REPO"
export XDG_DATA_HOME="$AGENT_DIR/.xdg/data" XDG_STATE_HOME="$AGENT_DIR/.xdg/state" XDG_CACHE_HOME="$AGENT_DIR/.xdg/cache"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME"
PYTENSOR_BASE="${L_SCRATCH:-$AGENT_DIR/.cache}/pytensor/job_${SLURM_JOB_ID:-local}"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"
VENV_DIR="$(dirname "$(dirname "$VENV_PY")")"
ln -sfn "$VENV_DIR" "$AGENT_DIR/venv"
AGENT_PY="$AGENT_DIR/venv/bin/python"
bash "$SLURM_DIR/build_agent_tree.sh" "$WORK_ROOT/agent_src" "$RUN_REPO" "$AGENT_PY"
bash "$SLURM_DIR/check_agent_tree.sh" "$RUN_REPO"

# --- a smoke test fields another run's frozen design (its design only, never its data) ------
SMOKE_FLAGS=()
if [[ -n "$SMOKE" ]]; then
  [[ -f "$SMOKE_FROM/design/frozen/record.json" ]] || { echo "ERROR: $SMOKE_FROM has no frozen design" >&2; exit 1; }
  if [[ ! -d "$RUN_DIR/experiment1/design" ]]; then
    mkdir -p "$RUN_DIR/experiment1"
    cp -r "$SMOKE_FROM/design" "$RUN_DIR/experiment1/design.tmp" && mv "$RUN_DIR/experiment1/design.tmp" "$RUN_DIR/experiment1/design"
  fi
  SMOKE_FLAGS=(--participants "$PARTICIPANTS" --stop-after-collect)
  N_EXPERIMENTS=1
  unset SMOKE_FROM
fi

# --- the other chains of the campaign: their finished experiments' timing reaches every design ---
OTHER=()
for other in "$CELLS_ROOT"/*_chain_*; do
  [[ -d "$other" && "$other" != "$CELL_DIR" && "$other" != *_smoke && -s "$other/agent_tree_id" ]] || continue
  k="${other##*_chain_}"
  OTHER+=("$AGENT_TREES_ROOT/$(cat "$other/agent_tree_id")/repo/_runs/chain_$k")
done
OTHER_FLAGS=()
[[ ${#OTHER[@]} -gt 0 ]] && OTHER_FLAGS=(--other-chains "${OTHER[@]}")

# --- deploying from Sherlock (as outer_live.sbatch) -------------------------------------------
LIVE_BIN="$(dirname "$WORK_ROOT")/rsa_live_bin"
mkdir -p "$LIVE_BIN"
NODE_REAL="$(command -v node)"
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec %q "$@"\n' \
  "${LD_LIBRARY_PATH:-}" "$NODE_REAL" > "$LIVE_BIN/node.tmp.$$" && chmod 755 "$LIVE_BIN/node.tmp.$$" && mv -f "$LIVE_BIN/node.tmp.$$" "$LIVE_BIN/node"
export PATH="$LIVE_BIN:$PATH"
export npm_config_cache="$(dirname "$WORK_ROOT")/.npm_rsa_live"
SMOKE_SITE=()
[[ -n "$SMOKE" ]] && SMOKE_SITE=(--smoke)
export AUTO_PSYCH_HOSTING_SITE="$(cd "$HARNESS_REPO" && "$VENV_PY" -m src.rsa_open.live --config "$CONFIG" --arm "$ARM" \
  --site "$CHAIN" ${SMOKE_SITE[@]+"${SMOKE_SITE[@]}"})"
export AUTO_PSYCH_DEPLOY_LOCK="$(dirname "$WORK_ROOT")/rsa_live_firebase_deploy.lock"

# --- when this job ends: no stage starts without the time it needs ----------------------------
DEADLINE_FLAG=()
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  end="$(squeue -h -j "$SLURM_JOB_ID" -o %e 2>/dev/null || true)"
  if [[ -n "$end" && "$end" != "N/A" ]]; then DEADLINE_FLAG=(--job-deadline "$(date -d "$end" +%s)"); fi
fi

# --- the chain ----------------------------------------------------------------------------
WITHHELD=()
for name in $(compgen -e); do
  value="${!name}"
  if [[ "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* || "$value" == *"$RUN2_ROOT"* ]]; then
    WITHHELD+=("$name")
  fi
done
UNSET_ARGS=()
for name in ${WITHHELD[@]+"${WITHHELD[@]}"}; do UNSET_ARGS+=(-u "$name"); done
echo "[open live] $CELL: $PROLIFIC_MODE, $N_EXPERIMENTS experiments, site $AUTO_PSYCH_HOSTING_SITE, code $(cat "$WORK_ROOT/code_commit")${CODE_TAG:+ ($CODE_TAG)} -> $RUN_DIR"
cd "$HARNESS_REPO"
set +e
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa_open.run_chain --chain "$CHAIN" --arm "$ARM" --run-dir "$RUN_DIR" \
    --private-dir "$CELL_DIR/private" --campaign-dir "$CAMPAIGN_DIR" --agent-root "$RUN_REPO" \
    --literature "$EXISTING" --seeds "$WORK_ROOT/open_seeds" --collection live --prolific-mode "$PROLIFIC_MODE" \
    ${CONFIRM_FLAG[@]+"${CONFIRM_FLAG[@]}"} --live-config "$CONFIG" --run-label "$ARM-c$CHAIN${SMOKE:+-smoke}" \
    --repo-root "$DEPLOY_REPO" --n-experiments "$N_EXPERIMENTS" --max-iterations "$MAX_ITERATIONS" \
    --candidate-count "$CANDIDATES" --pilot-participants "$PILOT_PARTICIPANTS" --cv-folds "$CV_FOLDS" \
    --theorist-timeout-sec "$THEORIST_TIMEOUT_SEC" --seed "$SEED" --code-commit "$(cat "$WORK_ROOT/code_commit")" \
    ${OTHER_FLAGS[@]+"${OTHER_FLAGS[@]}"} ${SMOKE_FLAGS[@]+"${SMOKE_FLAGS[@]}"} \
    ${DEADLINE_FLAG[@]+"${DEADLINE_FLAG[@]}"}
rc=$?
set -e

# --- outputs: the run without fit caches; private/ stays on Sherlock (Prolific ids) -------------
if [[ -d "$RUN_DIR" ]]; then
  rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/ ||
    echo "[open live] WARNING: copying the outputs failed (the run itself is in $RUN_DIR)"
fi
case "$rc" in
  0) date -Is > "$CELL_DIR/DONE"; stop "finished" ;;
  3) stop "awaiting the PI's review of experiment 1's paradigm: python -m src.rsa_open.review --frozen $RUN_DIR/experiment1/design/frozen --by <name>; then relaunch with CLEAR_STOP=1" ;;
  4) stop "test deploy done: the page and a Prolific draft are up (see the log); relaunch live with CLEAR_STOP=1" ;;
  5) echo "[open live] $CELL: paused for the next job of its ladder (too little of this job left for the next stage)" ;;
  *) stop "failed (exit $rc, job ${SLURM_JOB_ID:-local}): a person looks before it resumes (PREREG_DRAFT.md §4a)"; exit "$rc" ;;
esac
