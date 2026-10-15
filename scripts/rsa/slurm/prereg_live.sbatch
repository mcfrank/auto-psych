#!/bin/bash
#SBATCH --job-name=rsa_prereg_live
#SBATCH --partition=mcfrank
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --time=72:00:00
#SBATCH --output=%x_%j.out
#
# The pre-registered final experiment, live (docs/auto_rsa/PREREG_FINAL.md): the frozen
# design's page on its own Hosting site -> one Prolific study -> collection, catch
# exclusions -> the frozen analysis (experiment1/prereg_analysis.json). Submit it through
# scripts/rsa/live/prereg_launch.sh (config check, cost, typed "yes").
#
#   PROLIFIC_MODE=test  deploy the page and make a Prolific DRAFT to preview; stops there.
#   PROLIFIC_MODE=live  publish (real money): confirm_live_recruitment: true in the config
#                       AND CONFIRM_LIVE_RECRUITMENT=1.
#
# Resubmitting resumes: a recorded study is never published twice. No agents run.
set -euo pipefail
export REPO="${REPO:-${SLURM_SUBMIT_DIR:-$HOME/auto-psych}}"
export WORK_ROOT="${WORK_ROOT:-${SCRATCH:?}/auto-psych/rsa_prereg}"
source "$REPO/scripts/rsa/slurm/_env.sh"
command -v node >/dev/null && command -v npm >/dev/null || { echo "FATAL: node/npm needed to deploy (ml load nodejs)" >&2; exit 1; }
for v in PROLIFIC_API_TOKEN AUTO_PSYCH_RESULTS_TOKEN FIREBASE_TOKEN; do
  [[ -n "${!v:-}" ]] || { echo "FATAL: $v is not set (.secrets)" >&2; exit 1; }
done
CONFIG="${CONFIG:-$REPO/scripts/rsa/live/rsa_prereg.yaml}"
PROLIFIC_MODE="${PROLIFIC_MODE:?set PROLIFIC_MODE=test or live (use scripts/rsa/live/prereg_launch.sh)}"
cfg() { "$VENV_PY" -c 'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); v = c[sys.argv[2]]; print(" ".join(map(str, v)) if isinstance(v, list) else v)' "$CONFIG" "$1"; }
preds() { "$VENV_PY" -c 'import sys, yaml; print(" ".join(p[sys.argv[2]] for p in yaml.safe_load(open(sys.argv[1]))["predictions"]))' "$CONFIG" "$1"; }
FIREBASE_PROJECT="$(cfg firebase_project)"
SERIES="$(cfg series_label)"
PARTICIPANTS="$(cfg participants)"
DESIGN_SHA="$(cfg design_sha256)"
[[ -n "$DESIGN_SHA" ]] || { echo "FATAL: $CONFIG has no design_sha256: freeze the design first" >&2; exit 1; }
read -r -a PRED_PATHS <<< "$(preds path)"
read -r -a PRED_SHAS <<< "$(preds sha256)"
[[ ${#PRED_SHAS[@]} -eq ${#PRED_PATHS[@]} ]] || { echo "FATAL: every predictions file needs its sha256 in $CONFIG" >&2; exit 1; }
for i in "${!PRED_PATHS[@]}"; do PRED_PATHS[$i]="$REPO/${PRED_PATHS[$i]}"; done
CONFIRM_FLAG=()
if [[ "$PROLIFIC_MODE" == "live" ]]; then
  [[ "$(cfg confirm_live_recruitment)" == "True" ]] || { echo "FATAL: live needs confirm_live_recruitment: true in $CONFIG" >&2; exit 1; }
  [[ "${CONFIRM_LIVE_RECRUITMENT:-}" == "1" ]] || { echo "FATAL: live needs CONFIRM_LIVE_RECRUITMENT=1" >&2; exit 1; }
  CONFIRM_FLAG=(--confirm-live-recruitment)
elif [[ "$PROLIFIC_MODE" != "test" ]]; then
  echo "FATAL: PROLIFIC_MODE must be test or live, not $PROLIFIC_MODE" >&2; exit 1
fi

(
  flock 9
  bash "$REPO/scripts/rsa/slurm/stage_code.sh"
  (cd "$WORK_ROOT/harness_repo" && "$VENV_PY" -m src.rsa.live.config --config "$CONFIG" --render)
) 9>"$WORK_ROOT/.setup.lock"
HARNESS_REPO="$WORK_ROOT/harness_repo"
CELL_DIR="$WORK_ROOT/cells/prereg"
mkdir -p "$CELL_DIR/private"
DEPLOY_REPO="$CELL_DIR/deploy_repo"
rsync -a --delete --exclude .git --exclude /public --exclude /data --exclude 'node_modules/' \
  "$HARNESS_REPO"/ "$DEPLOY_REPO"/
"$VENV_PY" -m src.pipelines.outer_loop.deployment.record_provenance --checkout "$REPO" --copy "$DEPLOY_REPO"

LIVE_BIN="$(dirname "$WORK_ROOT")/rsa_live_bin"   # the campaign's node wrapper (outer_live.sbatch)
mkdir -p "$LIVE_BIN"
NODE_REAL="$(command -v node)"
printf '#!/bin/bash\nexport LD_LIBRARY_PATH=%q"${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\nexec %q "$@"\n' \
  "${LD_LIBRARY_PATH:-}" "$NODE_REAL" > "$LIVE_BIN/node.tmp.$$" && chmod 755 "$LIVE_BIN/node.tmp.$$" && mv -f "$LIVE_BIN/node.tmp.$$" "$LIVE_BIN/node"
export PATH="$LIVE_BIN:$PATH"
export npm_config_cache="$(dirname "$WORK_ROOT")/.npm_rsa_live"
export AUTO_PSYCH_HOSTING_SITE="$FIREBASE_PROJECT-$SERIES-c0"
export AUTO_PSYCH_DEPLOY_LOCK="$(dirname "$WORK_ROOT")/rsa_live_firebase_deploy.lock"

echo "[prereg] $PROLIFIC_MODE: $PARTICIPANTS people, design ${DESIGN_SHA:0:12}, site $AUTO_PSYCH_HOSTING_SITE"
cd "$HARNESS_REPO"
"$VENV_PY" -m src.rsa.prereg.run --run-dir "$CELL_DIR/run" --private-dir "$CELL_DIR/private" \
  --design-dir "$WORK_ROOT/design" --design-sha256 "$DESIGN_SHA" \
  --predictions "${PRED_PATHS[@]}" --predictions-sha256 "${PRED_SHAS[@]}" \
  --participants "$PARTICIPANTS" --collection live --prolific-mode "$PROLIFIC_MODE" ${CONFIRM_FLAG[@]+"${CONFIRM_FLAG[@]}"} \
  --firebase-project "$FIREBASE_PROJECT" --run-label prereg --repo-root "$DEPLOY_REPO"
date -Is > "$CELL_DIR/LAST_EXIT_OK"
echo "[prereg] done: $CELL_DIR/run (private/: raw responses with Prolific ids, never copied off Sherlock)"
