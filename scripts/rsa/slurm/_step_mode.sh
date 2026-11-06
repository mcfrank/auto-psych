# Step mode for a chain's job (DISTRIBUTED_DESIGN.md v3 §0b), sourced by open_live.sbatch and
# open_rehearsal.sbatch. `rsa_step_mode <script> <var>...` writes $CELL_DIR/continuation.sh, which
# submits this chain's next step (the same <script>) after a Slurm dependency and prints its job
# id, and $CELL_DIR/step_settings.sh, the named variables (and the step's own) that the next step
# sources first (STEP_SETTINGS: a file, since sbatch --export cannot carry a comma, as in
# mcfrank,owners). It sets STEP_ENV: the settings the chain's process needs
# (src.rsa_open.loop.slurm.StepConfig.from_env). Needs CELL, CELL_DIR, WORK_ROOT, REPO, SLURM_DIR,
# VENV_PY, HARNESS_REPO, JAX_COMPILATION_CACHE_DIR and a Slurm job (its name makes steps
# singletons). A step a person submitted (RSA_STEP_NEW_LAUNCH=1, set by the job script when it found no
# STEP_SETTINGS) starts a launch: one experiment, at most 40 steps (src.rsa_open.loop.slurm), and it
# refuses to start while another step of the chain is queued or running. Steps run on STEP_PARTITION (mcfrank) with STEP_CPUS / STEP_MEM / STEP_TIME; fits on
# STEP_FIT_PARTITION (mcfrank,owners) with STEP_FIT_CONSTRAINT (CPU_GEN:SIE; "none" for none).

rsa_step_mode() {
  local script="$1"; shift
  [[ -n "${SLURM_JOB_NAME:-}" ]] || { echo "FATAL: step mode runs as a Slurm job (its name makes steps singletons)" >&2; exit 1; }
  if [[ "${RSA_STEP_NEW_LAUNCH:-0}" == 1 ]]; then
    local others
    # itself, as squeue names it: its job id, or <array job>_<task> for a one-task array's step
    local self_ids=("${SLURM_JOB_ID:-}")
    [[ -n "${SLURM_ARRAY_JOB_ID:-}" ]] && self_ids+=("${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}")
    others="$(squeue --me -h -n "$SLURM_JOB_NAME" -o %i | grep -vxF "${self_ids[@]/#/-e}" || true)"
    if [[ -n "$others" ]]; then
      echo "FATAL: $CELL already has a step queued or running (jobs: $(echo $others)); not launching a second" >&2
      exit 1
    fi
  fi
  STEP_PARTITION="${STEP_PARTITION:-mcfrank}" STEP_TIME="${STEP_TIME:-2-00:00:00}"
  STEP_CPUS="${STEP_CPUS:-8}" STEP_MEM="${STEP_MEM:-32G}"
  STEP_FIT_PARTITION="${STEP_FIT_PARTITION:-mcfrank,owners}" STEP_FIT_CONSTRAINT="${STEP_FIT_CONSTRAINT:-CPU_GEN:SIE}"
  CONT="$CELL_DIR/continuation.sh"
  local keep=("$@" WORK_ROOT REPO STEP_MODE STEP_PARTITION STEP_TIME STEP_CPUS STEP_MEM STEP_FIT_PARTITION
              STEP_FIT_CONSTRAINT MAIL_TYPE MAIL_USER)
  local settings="$CELL_DIR/step_settings.sh" k
  [[ "$settings" != *,* ]] || { echo "FATAL: $settings contains a comma; sbatch --export cannot carry it" >&2; exit 1; }
  {
    echo "# Written by $(basename "$script") (step mode, job ${SLURM_JOB_ID:-local}); sourced by $CELL's next step."
    for k in "${keep[@]}"; do
      if [[ -n "${!k:-}" ]]; then printf 'export %s=%q\n' "$k" "${!k}"; fi
    done
  } > "$settings.tmp" && mv "$settings.tmp" "$settings"
  local exports="ALL,STEP_SETTINGS=$settings"
  local mail=(--mail-type="${MAIL_TYPE:-FAIL,TIME_LIMIT}")
  [[ -n "${MAIL_USER:-}" ]] && mail+=(--mail-user="$MAIL_USER")
  local submit=(sbatch --parsable --no-requeue --chdir="$REPO" --partition="$STEP_PARTITION" --time="$STEP_TIME"
                --cpus-per-task="$STEP_CPUS" --mem="$STEP_MEM" "${mail[@]}" --job-name="$SLURM_JOB_NAME"
                -o "$WORK_ROOT/logs/%x_%j.out" --export="$exports")
  mkdir -p "$WORK_ROOT/logs"
  # Rewritten by each step from its own settings. The next step is submitted as from a login shell
  # (env -i, bash -l): the chain's process has the withheld variables removed, and its PATH puts the
  # agents' wrappers first.
  {
    echo '#!/bin/bash'
    echo "# Written by $(basename "$script") (step mode, job ${SLURM_JOB_ID:-local}): submit $CELL's next step"
    echo '# after $1 (a Slurm dependency, e.g. afterany:<ids>,singleton) and print its job id.'
    echo 'set -euo pipefail'
    printf 'jid="$(env -i HOME=%q USER=%q LOGNAME=%q bash -lc '"'"'exec "$@"'"'"' bash ' \
      "$HOME" "${USER:-$(id -un)}" "${LOGNAME:-${USER:-$(id -un)}}"
    printf '%q ' "${submit[@]}"
    printf -- '--dependency="$1" %q)"\n' "$script"
    echo 'echo "${jid%%;*}"'
  } > "$CONT.tmp" && chmod 700 "$CONT.tmp" && mv "$CONT.tmp" "$CONT"
  # For the chain's process only; never a sandbox's (agents get an allowlisted environment).
  STEP_ENV=(RSA_OPEN_STEP_MODE=1 RSA_OPEN_FIT_BACKEND=slurm RSA_OPEN_NEW_LAUNCH="${RSA_STEP_NEW_LAUNCH:-0}"
            RSA_OPEN_CELL_DIR="$CELL_DIR"
            RSA_OPEN_CHAIN_JOB="$SLURM_JOB_NAME" RSA_OPEN_CONTINUATION="$CONT"
            RSA_OPEN_FIT_SCRIPT="$SLURM_DIR/fit_array.sbatch" RSA_OPEN_LOGS="$WORK_ROOT/logs"
            RSA_OPEN_FIT_PARTITION="$STEP_FIT_PARTITION" RSA_OPEN_FIT_CONSTRAINT="$STEP_FIT_CONSTRAINT"
            VENV_PY="$VENV_PY" HARNESS_REPO="$HARNESS_REPO" JAX_COMPILATION_CACHE_DIR="$JAX_COMPILATION_CACHE_DIR")
}

# After the chain's process: what its exit code means for a step (prints why; returns 0 to go on).
#   6 the fits and the next step are submitted; 7 the step refused to submit (its STOP is written);
#   5 paused for time: the step submits its own continuation (singleton); 8 this launch's experiment is
#   done: STOP with how to launch the next, and 15 (Slurm mails).
rsa_step_exit() {
  case "$1" in
    6) echo "[step] $CELL: done; its fits and the next step are submitted (steps.jsonl)" ;;
    8) printf '%s\t%s\n' "$(date -Is)" "this launch's experiment is done; launch the next: CLEAR_STOP=1 STEP_MODE=1 (scripts/rsa/live/open_launch.sh)" > "$CELL_DIR/STOP"
       echo "[step] $CELL: $(cat "$CELL_DIR/STOP")"; return 15 ;;
    7) echo "[step] $CELL: refused to submit: $(cat "$CELL_DIR/STOP" 2>/dev/null)"; return 7 ;;
    5) local next
       next="$(bash "$CONT" singleton)" || {
         printf '%s\t%s\n' "$(date -Is)" "a step paused for time could not submit its continuation" > "$CELL_DIR/STOP"; return 1; }
       echo "[step] $CELL: paused for time; continues in job $next" ;;
    *) return "$1" ;;
  esac
}
