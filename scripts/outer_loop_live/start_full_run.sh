#!/bin/bash
# ============================================================================
#  START THE FULL LIVE RUN
#
#  Launches K parallel, independent runs of the config in full_run.yaml
#  (default: 3 runs x 3 experiments x 40 participants  =>  9 Prolific studies,
#  360 paid participants; the cost summary it prints gives the estimate).
#  RECRUITS REAL HUMANS AND SPENDS REAL MONEY.
#
#  All run settings (experiments, participants, reward, walltime, modeling)
#  come from the config file — NOT hardcoded here. Only K (the number of
#  parallel runs) is a launcher knob.
#
#  Run:  bash scripts/outer_loop_live/start_full_run.sh
#        K=5 CONFIG=path/to/other.yaml CONFIRM=yes bash scripts/outer_loop_live/start_full_run.sh
#
#  Relaunching a run index DELETES that run's earlier output and run copy
#  ($WORK_ROOT/run<i>, $WORK_ROOT/runs/run<i>) once you confirm; the prompt
#  lists them. CONFIRM=yes skips the prompt, the deletion included.
# ============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OUTER_LIVE_SLURM_DIR="$DIR"
source "$DIR/_env.sh"   # REPO, WORK_ROOT, VENV_PY, caches, deploy lock, secrets

CONFIG="${CONFIG:-$DIR/full_run.yaml}"
K="${K:-3}"
[[ -f "$CONFIG" ]] || { echo "config not found: $CONFIG" >&2; exit 1; }

# RUNS: which run indices to (re)launch (default 1..K). Use RUNS="2 3" to redo
# only those runs without touching the others (e.g. a working run1).
if [[ -n "${RUNS:-}" ]]; then read -r -a _runs <<< "${RUNS//,/ }"; else _runs=($(seq 1 "$K")); fi

# --- 1. Validate config + Prolific token, print per-run cost (stderr), and
#        import the config's settings as env vars (stdout: export lines). -----
if ! CFG_ENV="$("$VENV_PY" "$DIR/_pilot_config.py" "$CONFIG" --check)"; then
  echo "Config / Prolific-token validation failed — aborting (nothing launched)." >&2
  exit 1
fi
eval "$CFG_ENV"  # PROJECT N_EXPERIMENTS N_PARTICIPANTS PROLIFIC_MODE WALLTIME QOS
                 # CODING_AGENT FIREBASE_PROJECT DRAWS TUNE CHAINS INNER_LOOP_*

# --- 2. Confirm (the --check above already printed per-run cost + token).
#        Nothing is rendered or deleted before the typed "yes". ---------------
# Relaunching a run index starts that run afresh, so its earlier output and
# run-copy directories are cleared: list exactly the ones that exist.
to_delete=()
for i in "${_runs[@]}"; do
  for d in "$WORK_ROOT/run$i" "$WORK_ROOT/runs/run$i"; do
    [[ -e "$d" ]] && to_delete+=("$d")
  done
done
cat <<EOF

  => launching run(s): ${_runs[*]}  (${#_runs[@]} parallel; Prolific cost ≈ ${#_runs[@]} x the per-run estimate shown; AI agent costs not included)
     config=$CONFIG  mode=$PROLIFIC_MODE  experiments=$N_EXPERIMENTS  N=$N_PARTICIPANTS/exp
     walltime=$WALLTIME  qos=${QOS:-<default normal>}

EOF
if (( ${#to_delete[@]} )); then
  echo "  => on \"yes\", first DELETES these directories (the earlier results of those runs):"
  printf '       %s\n' "${to_delete[@]}"
else
  echo "  => deletes nothing (no earlier directories for run(s) ${_runs[*]})"
fi
echo
if [[ "${CONFIRM:-}" != "yes" ]]; then
  read -r -p "Type \"yes\" to delete the directories listed above and launch ${#_runs[@]} live run(s) (publishes studies, recruits, pays): " reply
  [[ "$reply" == "yes" ]] || { echo "Aborted — nothing deleted, nothing launched."; exit 1; }
fi

# --- 3. Render the project's prolific_config.yaml from this config -----------
"$VENV_PY" "$DIR/_pilot_config.py" "$CONFIG" --render-only

# --- 4. Clear the selected runs' prior artifacts so the live --resume is fresh -
for d in ${to_delete[@]+"${to_delete[@]}"}; do
  rm -rf "$d"
done
echo "[clean] removed ${#to_delete[@]} dir(s) for run(s) ${_runs[*]} under $WORK_ROOT"

# --- 5. Launch the selected runs. submit_parallel inherits the exported config
#        settings; we pass K + RUNS and map N_EXPERIMENTS -> EXPERIMENTS (the
#        sequence form run_live.sbatch reads). WALLTIME/QOS come from the config.
K="$K" \
RUNS="${RUNS:-}" \
EXPERIMENTS="$N_EXPERIMENTS" \
WORK_ROOT="$WORK_ROOT" \
  bash "$DIR/submit_parallel.sh"

echo
echo "Launched. Monitor:  squeue --me"
echo "Logs:               $WORK_ROOT/slurm_logs/outer_live_run*_*.out"
echo "TO STOP: pause/stop EACH study in the Prolific dashboard (stops recruiting/charging),"
echo "         then  scancel <jobids>."
