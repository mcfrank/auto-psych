#!/bin/bash
#SBATCH --job-name=outer_live
#SBATCH --partition=normal
#SBATCH --time=1-00:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.out
#
# Run ONE live experiment end-to-end: theory/design/implement (coding agents) ->
# deploy to Firebase -> create + publish a Prolific study -> poll for human
# submissions (<= 2 h/experiment) -> fetch results -> inner model loop.
#
# No GPU: the coding agents are remote, participants are real humans, and only
# the inner-loop MCMC runs locally (CPU). For sequences longer than ~2 days add
# `--qos=long` (up to 7 days) when submitting.
#
# MUST run from a PER-RUN copy of the repo (RUN_WORKTREE): the coding agents —
# opencode especially — keep per-project state and must not share a working dir
# or session DB across parallel runs. run_pilot.sh and submit_parallel.sh make a
# per-run rsync copy (el7's git is too old for `git worktree`).
#
# Per-run knobs (env; set by run_pilot.sh / submit_parallel.sh, or via --export):
#   RUN_LABEL              unique label -> /e{N}-{label}/ hosting path + session id
#   RUN_WORKTREE           per-run repo copy (its own public/, functions/, opencode.json)
#   AUTO_PSYCH_OUTPUT_DIR  this run's data tree (avoids experiment{N}/ collisions)
#   CODING_AGENT (opencode)  PROJECT (subjective_randomness)  EXPERIMENT (1)
#   N_PARTICIPANTS (1)  FIREBASE_PROJECT  DESIGN_MODE (agent|exhaustive)

set -euo pipefail
SLURM_DIR="${OUTER_LIVE_SLURM_DIR:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "$SLURM_DIR/_env.sh"

PROJECT="${PROJECT:-subjective_randomness}"
EXPERIMENT="${EXPERIMENT:-1}"
N_PARTICIPANTS="${N_PARTICIPANTS:-1}"
FIREBASE_PROJECT="${FIREBASE_PROJECT:-auto-psych-2c5da}"
DESIGN_MODE="${DESIGN_MODE:-agent}"
RUN_LABEL="${RUN_LABEL:-run-${SLURM_JOB_ID:-manual}}"
RUN_WORKTREE="${RUN_WORKTREE:-$REPO}"
export AUTO_PSYCH_OUTPUT_DIR="${AUTO_PSYCH_OUTPUT_DIR:-$WORK_ROOT/${RUN_LABEL}/data}"
mkdir -p "$AUTO_PSYCH_OUTPUT_DIR"

# Per-run PyTensor compile dir off NFS $HOME: stage-5 PyMC JIT-compiles models,
# and concurrent runs sharing ~/.pytensor contend on its compile filelock
# ("Stale file handle"). Node-local SSD is fast and gives a reliable flock.
PYTENSOR_BASE="${L_SCRATCH:-$WORK_ROOT/$RUN_LABEL/.cache}/pytensor"
mkdir -p "$PYTENSOR_BASE"
export PYTENSOR_FLAGS="base_compiledir=$PYTENSOR_BASE"

# Run from this run's worktree so its public/, firebase.generated.json, and
# functions/ are isolated from other parallel runs (only the remote Firebase
# site is shared; the deploy step serializes on AUTO_PSYCH_DEPLOY_LOCK).
cd "$RUN_WORKTREE"

# Expose the shared venv to the Claude Code agents — their prompts run
# `python3 -m src...`, so prepend the venv bin to make python3 resolve to the
# interpreter that has the full dep set. Then drop UV_* so a stray agent
# `uv run` can't re-sync (and clobber) the shared venv; it would target a
# throwaway .venv in the worktree instead. VENV_PY is already an absolute path.
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
unset UV_PROJECT_ENVIRONMENT UV_CACHE_DIR UV_PYTHON UV_PYTHON_INSTALL_DIR

# --- opencode parallel-safety (per-run isolation) ----------------------------
# opencode keeps a per-project session SQLite DB under XDG_DATA_HOME (default
# ~/.local/share/opencode — NFS $HOME, SHARED across runs). Concurrent runs
# sharing it corrupt the DB ("SQLITE_CORRUPT: database disk image is malformed").
# Give each run PRIVATE XDG data/state/cache + tmp, keyed by RUN_LABEL + job id
# so two runs on one node still never collide. Prefer node-local SSD (reliable
# SQLite locking, off NFS). XDG_CONFIG_HOME is left alone so opencode still finds
# its global config/auth under ~/.config/opencode.
OC_STATE="${L_SCRATCH:-$WORK_ROOT/$RUN_LABEL/.cache}/oc_${RUN_LABEL}_${SLURM_JOB_ID:-$$}"
export XDG_DATA_HOME="$OC_STATE/data"
export XDG_STATE_HOME="$OC_STATE/state"
export XDG_CACHE_HOME="$OC_STATE/cache"
export TMPDIR="$OC_STATE/tmp"
mkdir -p "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_CACHE_HOME" "$TMPDIR"

CODING_AGENT="${CODING_AGENT:-opencode}"
# opencode operates on its cwd and reaches paths outside it only via
# external_directory grants (no Claude-style --add-dir). The experiment outputs
# live on $SCRATCH outside the run copy, so grant the agents access. We patch
# the run copy's opencode.json (a per-run rsync copy on $SCRATCH), never your
# tracked repo file.
if [[ "$CODING_AGENT" == "opencode" ]]; then
  "$VENV_PY" - "$RUN_WORKTREE/opencode.json" "$AUTO_PSYCH_OUTPUT_DIR" <<'PY'
import json, sys, pathlib
cfg_path, out = pathlib.Path(sys.argv[1]), sys.argv[2]
cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {"$schema": "https://opencode.ai/config.json"}
perm = cfg.setdefault("permission", {})
for k in ("edit", "bash", "webfetch"):
    perm.setdefault(k, "allow")
ext = perm.setdefault("external_directory", {})
ext[f"{out}/**"] = "allow"; ext[out] = "allow"
cfg_path.write_text(json.dumps(cfg, indent=2))
print(f"[opencode] private XDG under {pathlib.Path(out).anchor}...; external_directory granted for outputs")
PY
fi

# Single experiment by default. Set EXPERIMENTS (e.g. "3" => 1..3, or "2-4") to
# run a SEQUENCE: each experiment is its own deploy + Prolific study, run one
# after another, and experiment k's design is informed by experiment k-1's
# posterior.
if [[ -n "${EXPERIMENTS:-}" ]]; then
  EXP_FLAG=(--experiments "$EXPERIMENTS")
  EXP_DESC="experiments=$EXPERIMENTS"
else
  EXP_FLAG=(--experiment "$EXPERIMENT")
  EXP_DESC="experiment=$EXPERIMENT"
fi

# Optional modeling overrides (from pilot.yaml via env); unset -> run.py defaults.
EXTRA=()
[[ -n "${DRAWS:-}" ]]                  && EXTRA+=(--draws "$DRAWS")
[[ -n "${TUNE:-}" ]]                   && EXTRA+=(--tune "$TUNE")
[[ -n "${CHAINS:-}" ]]                 && EXTRA+=(--chains "$CHAINS")
[[ -n "${INNER_LOOP_ITERATIONS:-}" ]]  && EXTRA+=(--inner-loop-iterations "$INNER_LOOP_ITERATIONS")
[[ -n "${INNER_LOOP_CANDIDATES:-}" ]]  && EXTRA+=(--inner-loop-candidates "$INNER_LOOP_CANDIDATES")

echo "[run] label=$RUN_LABEL worktree=$RUN_WORKTREE out=$AUTO_PSYCH_OUTPUT_DIR $EXP_DESC N=$N_PARTICIPANTS design=$DESIGN_MODE"

"$VENV_PY" -m src.pipelines.outer_loop.run \
  --project "$PROJECT" \
  "${EXP_FLAG[@]}" \
  --mode live \
  --deploy-target firebase \
  --prolific-mode live \
  --firebase-project "$FIREBASE_PROJECT" \
  --design-mode "$DESIGN_MODE" \
  --run-label "$RUN_LABEL" \
  --n-participants "$N_PARTICIPANTS" \
  --coding-agent "$CODING_AGENT" \
  --validate \
  --resume \
  ${EXTRA[@]+"${EXTRA[@]}"}

echo "[run] done -> $AUTO_PSYCH_OUTPUT_DIR/$PROJECT/experiment$EXPERIMENT"
