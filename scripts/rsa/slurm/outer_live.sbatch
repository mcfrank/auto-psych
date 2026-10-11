#!/bin/bash
#SBATCH --job-name=rsa_outer_live
#SBATCH --partition=mcfrank
#SBATCH --array=0-2
#SBATCH --cpus-per-task=8
#SBATCH --mem=48GB
#SBATCH --time=48:00:00
#SBATCH --output=%x_%A_%a.out
#
# The RSA live campaign: array task i runs chain CHAINS[i] of the config
# (scripts/rsa/live/rsa_live.yaml) as one outer-loop run with real participants:
# design -> deploy the page to the chain's Hosting site -> Prolific -> collect ->
# prospective score -> inner loop, for each experiment. Submit it through
# scripts/rsa/live/launch.sh, which checks the config, shows the cost and asks
# for a typed "yes" before a live launch.
#
#   PROLIFIC_MODE=test: deploy experiment 1's page and create a Prolific DRAFT to
#     preview; the run stops there. Nothing is published or paid.
#   PROLIFIC_MODE=live: publish the study (real money). Needs
#     confirm_live_recruitment: true in the config AND CONFIRM_LIVE_RECRUITMENT=1.
#   PILOT=1 (with PARTICIPANTS, e.g. 20): stop after experiment 1's data are in.
#
# Resubmitting a task resumes its run where it stopped: a recorded live study is
# never deployed or published again (main's guard). Each chain deploys from its
# own copy of the staged code (CELL/deploy_repo, provenance recorded); deploys
# are serialized by a lock shared across nodes.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_live}"
source "$REPO/scripts/rsa/slurm/_env.sh"
rsa_require_opencode
rsa_require_gemini_key
command -v bwrap >/dev/null || { echo "FATAL: no bwrap (sandboxed agents need it)" >&2; exit 1; }
command -v flock >/dev/null || { echo "FATAL: no flock" >&2; exit 1; }
command -v node >/dev/null && command -v npm >/dev/null || { echo "FATAL: node/npm needed to deploy (ml load nodejs)" >&2; exit 1; }
for v in PROLIFIC_API_TOKEN AUTO_PSYCH_RESULTS_TOKEN FIREBASE_TOKEN; do
  [[ -n "${!v:-}" ]] || { echo "FATAL: $v is not set (.secrets)" >&2; exit 1; }
done

CONFIG="${CONFIG:-$REPO/scripts/rsa/live/rsa_live.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live (use scripts/rsa/live/launch.sh)}"
TASK="${SLURM_ARRAY_TASK_ID:?run as an array task}"
cfg() { "$VENV_PY" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); v = c[sys.argv[2]]; print(" ".join(map(str, v)) if isinstance(v, list) else v)' "$CONFIG" "$1"; }
read -r -a CHAINS <<< "$(cfg chains)"
CHAIN="${CHAINS[$TASK]:?array task $TASK has no chain in $CONFIG}"
FIREBASE_PROJECT="$(cfg firebase_project)"
SERIES="$(cfg series_label)"
PARTICIPANTS="${PARTICIPANTS:-$(cfg participants)}"
DISPLAYS="${DISPLAYS:-$(cfg displays)}"
TRIALS="${TRIALS:-$(cfg trials)}"
N_EXPERIMENTS="${N_EXPERIMENTS:-$(cfg experiments)}"
CONFIRM_FLAG=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  [[ "$(cfg confirm_live_recruitment)" == "True" ]] || { echo "FATAL: live needs confirm_live_recruitment: true in $CONFIG" >&2; exit 1; }
  [[ "${CONFIRM_LIVE_RECRUITMENT:-}" == "1" ]] || { echo "FATAL: live needs CONFIRM_LIVE_RECRUITMENT=1" >&2; exit 1; }
  CONFIRM_FLAG=(--confirm-live-recruitment)
elif [[ "$PROLIFIC_MODE" != "test" ]]; then
  echo "FATAL: PROLIFIC_MODE must be test or live, not $PROLIFIC_MODE" >&2; exit 1
fi
PILOT_FLAG=()
[[ "${PILOT:-}" == "1" ]] && PILOT_FLAG=(--stop-after-collect)
RUN2_ROOT="${RUN2_ROOT:-$(rsa_default_work_root)}"
EXISTING="$RUN2_ROOT/promote_work/all_trials.csv"
PROMOTED="$REPO/data/rsa/live_seeds/models"
CHAIN_SEEDS="$REPO/data/rsa/live_seeds/chains/chain_$CHAIN"
AGENT_MODEL="${AGENT_MODEL:-google/gemini-3.8-flash}"
AGENT_TIMEOUT_SEC="${AGENT_TIMEOUT_SEC:-2400}"
SEED="${SEED:-0}"
for f in "$EXISTING" "$PROMOTED/models_manifest.yaml" "$CHAIN_SEEDS/models_manifest.yaml"; do
  [[ -f "$f" ]] || { echo "ERROR: $f is missing" >&2; exit 1; }
done
mkdir -p "$WORK_ROOT" "$CELLS_ROOT"

# --- once per campaign: staged code, the rendered Prolific settings ----------------
(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa.live.config --config "$CONFIG" --render)
) 9>"$WORK_ROOT/.setup.lock"
HARNESS_REPO="$WORK_ROOT/harness_repo"
SLURM_DIR="$HARNESS_REPO/scripts/rsa/slurm"

# --- the cell: its seeds, its deploy copy, its agent tree ---------------------------
CELL="live_${SERIES}_c$CHAIN${PILOT:+_pilot}"
CELL_DIR="$CELLS_ROOT/$CELL"
mkdir -p "$CELL_DIR/private"
[[ -d "$CELL_DIR/seeds" ]] || { cp -r "$CHAIN_SEEDS" "$CELL_DIR/seeds.tmp" && mv "$CELL_DIR/seeds.tmp" "$CELL_DIR/seeds"; }
DEPLOY_REPO="$CELL_DIR/deploy_repo"
rsync -a --delete --exclude .git --exclude /public --exclude /data --exclude 'node_modules/' \
  "$HARNESS_REPO"/ "$DEPLOY_REPO"/
"$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout "$REPO" --copy "$DEPLOY_REPO"
[[ -s "$CELL_DIR/agent_tree_id" ]] || head -c 8 /dev/urandom | od -An -tx1 | tr -d ' \n' > "$CELL_DIR/agent_tree_id"
AGENT_DIR="$AGENT_TREES_ROOT/$(cat "$CELL_DIR/agent_tree_id")"
RUN_REPO="$AGENT_DIR/repo"
RUN_DIR="$RUN_REPO/_runs/outer"
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

# --- deploying from Sherlock (as scripts/outer_loop_live/_env.sh) --------------------
# firebase-tools' functions deploy runs a child node that needs the gcc module's
# libstdc++: a node wrapper carries LD_LIBRARY_PATH to it (2026-09-29). Kept
# outside WORK_ROOT so the harness keeps it (paths under WORK_ROOT are withheld).
LIVE_BIN="$(dirname "$WORK_ROOT")/rsa_live_bin"
mkdir -p "$LIVE_BIN"
NODE_REAL="$(command -v node)"
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec %q "$@"\n' \
  "${LD_LIBRARY_PATH:-}" "$NODE_REAL" > "$LIVE_BIN/node.tmp.$$" && chmod 755 "$LIVE_BIN/node.tmp.$$" && mv -f "$LIVE_BIN/node.tmp.$$" "$LIVE_BIN/node"
export PATH="$LIVE_BIN:$PATH"
export npm_config_cache="$(dirname "$WORK_ROOT")/.npm_rsa_live"
export AUTO_PSYCH_HOSTING_SITE="$FIREBASE_PROJECT-$SERIES-c$CHAIN"
export AUTO_PSYCH_DEPLOY_LOCK="$(dirname "$WORK_ROOT")/rsa_live_firebase_deploy.lock"

# --- the run ------------------------------------------------------------------------
WITHHELD=()
for name in $(compgen -e); do
  value="${!name}"
  if [[ "$value" == *"$WORK_ROOT"* || "$value" == *"$REPO"* || "$value" == *"$RUN2_ROOT"* ]]; then
    WITHHELD+=("$name")
  fi
done
UNSET_ARGS=()
for name in ${WITHHELD[@]+"${WITHHELD[@]}"}; do UNSET_ARGS+=(-u "$name"); done
echo "[live c$CHAIN] $PROLIFIC_MODE${PILOT:+ pilot}: $N_EXPERIMENTS experiments x $PARTICIPANTS people, site $AUTO_PSYCH_HOSTING_SITE -> $RUN_DIR"
cd "$HARNESS_REPO"
env ${UNSET_ARGS[@]+"${UNSET_ARGS[@]}"} PATH="$RUN_REPO/.agent_bin:$PATH" \
  "$AGENT_PY" -m src.rsa.outer.run --run-dir "$RUN_DIR" --private-dir "$CELL_DIR/private" \
    --seeds "$CELL_DIR/seeds" --promoted "$PROMOTED" --existing-data "$EXISTING" \
    --collection live --prolific-mode "$PROLIFIC_MODE" ${CONFIRM_FLAG[@]+"${CONFIRM_FLAG[@]}"} \
    ${PILOT_FLAG[@]+"${PILOT_FLAG[@]}"} --firebase-project "$FIREBASE_PROJECT" --run-label "c$CHAIN${PILOT:+pilot}" \
    --repo-root "$DEPLOY_REPO" --n-experiments "$N_EXPERIMENTS" --participants "$PARTICIPANTS" \
    --trials "$TRIALS" --displays "$DISPLAYS" --seed "$SEED" \
    --agent-root "$RUN_REPO" --agent-model "$AGENT_MODEL" --agent-timeout-sec "$AGENT_TIMEOUT_SEC"

# --- outputs: the run without fit caches; private/ stays on Sherlock (Prolific ids) ---
rsync -a --delete --exclude '.fit_cache/' --exclude '.home/' --exclude '.agent_shell/' "$RUN_DIR"/ "$CELL_DIR/outputs"/
date -Is > "$CELL_DIR/LAST_EXIT_OK"
echo "[live c$CHAIN] done: $CELL_DIR/outputs (private/: raw responses with Prolific ids, never copied off Sherlock)"
