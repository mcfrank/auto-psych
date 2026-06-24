#!/bin/bash
# wd.sh -- OOM watchdog toolkit (deterministic Slurm mechanics).
#
# This script does the fiddly, error-prone Slurm bookkeeping so the Claude Code
# supervisor (oom_watchdog.sbatch) only has to make judgement calls. Every
# failure here is LOUD: a missing recipe, an infeasible memory request, or a
# failed sbatch aborts with a clear message rather than silently doing the wrong
# thing.
#
# Subcommands:
#   init                       discover all of my current jobs -> managed set
#   capture <arrayjobid>       snapshot a running job's submit env -> recipe
#   scan                       print unhandled OOM candidates (one JSON per line)
#   resubmit <recipe> <task> <mem_gb> <logical_key> <prev_jobid>
#                              resubmit one task with more memory; record it
#   handle <prev_jobid> <task> <logical_key> <reason>
#                              mark a failure handled WITHOUT resubmitting
#   status                     human-readable summary of everything we've done
#   active                     print count of managed jobs still R/PD (loop uses this)
#
# Core idea -- exact reproduction: a job submitted with `--export=ALL` carries
# its submitter's full environment. We read that environment back from the
# running batch process (/proc/<pid>/environ via `srun --overlap`) and replay it
# on resubmit, so the new task maps to the IDENTICAL (seed, ground-truth, flags)
# cell. Combined with the pipeline's `--resume`, the bumped task continues from
# where the OOM-killed one left off instead of redoing finished work.

set -euo pipefail

# --- configuration (all overridable from the environment) ------------------
WD_STATE_DIR="${WD_STATE_DIR:-${SCRATCH:-$GROUP_SCRATCH}/auto-psych/watchdog}"
WD_MEM_CAP_GB="${WD_MEM_CAP_GB:-256}"        # never request more than this
WD_MAX_RETRIES="${WD_MAX_RETRIES:-3}"        # resubmits per logical task, then give up
WD_BIGMEM_THRESHOLD_GB="${WD_BIGMEM_THRESHOLD_GB:-64}"  # above this -> bigmem partition
WD_NORMAL_PERCORE_GB="${WD_NORMAL_PERCORE_GB:-8}"       # `normal` partition mem/core ceiling
WD_BIGMEM_PERCORE_GB="${WD_BIGMEM_PERCORE_GB:-64}"      # `bigmem` partition mem/core ceiling
WD_DRY_RUN="${WD_DRY_RUN:-0}"                # 1 => print the sbatch but do not submit/record

RECIPE_DIR="$WD_STATE_DIR/recipes"
MANAGED_FILE="$WD_STATE_DIR/managed_arrays.txt"   # array/job ids we are responsible for
JOBMAP_FILE="$WD_STATE_DIR/jobid_map.tsv"         # resubmitted jobid -> logical_key,recipe,task
LEDGER_FILE="$WD_STATE_DIR/ledger.jsonl"          # append-only record of every action
LOG_FILE="$WD_STATE_DIR/watchdog.log"

mkdir -p "$RECIPE_DIR"
: >> "$MANAGED_FILE"; : >> "$JOBMAP_FILE"; : >> "$LEDGER_FILE"; : >> "$LOG_FILE"

# --- small helpers ---------------------------------------------------------
log()  { printf '%s %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "$LOG_FILE" >&2; }
die()  { log "ERROR: $*"; exit 1; }

# Parse a Slurm memory string already normalised to gigabytes (e.g. "32G",
# "0.86G", "32") into an integer number of GB, rounding up. Empty -> 0.
mem_to_gb() {
  local raw="${1:-}"
  raw="${raw%[A-Za-z]}"                       # strip a trailing unit letter (G)
  [[ -z "$raw" || "$raw" == "0" ]] && { echo 0; return; }
  awk -v x="$raw" 'BEGIN { printf "%d", (x == int(x) ? x : int(x) + 1) }'
}

# Decide the next memory + partition given the failed job's current memory and
# how many times this logical task has already been retried. Echoes either
# "<mem_gb> <partition>" or the single word "EXHAUSTED".
next_step() {
  local cur_gb="$1" retries="$2"
  if (( retries >= WD_MAX_RETRIES )); then echo "EXHAUSTED"; return; fi
  local next=$(( cur_gb * 2 ))
  (( next < 1 )) && next=$WD_NORMAL_PERCORE_GB        # guard against a 0/unknown reading
  if (( next > WD_MEM_CAP_GB )); then
    if (( cur_gb < WD_MEM_CAP_GB )); then next=$WD_MEM_CAP_GB; else echo "EXHAUSTED"; return; fi
  fi
  echo "$next $(partition_for "$next")"
}

# The user's policy: stay on `normal` up to the bigmem threshold, escalate above.
partition_for() {
  local gb="$1"
  if (( gb <= WD_BIGMEM_THRESHOLD_GB )); then echo normal; else echo bigmem; fi
}

# Append one JSON object to the ledger. Args are key=value pairs; values are
# emitted verbatim as JSON strings (our values are ids/states/paths -- no quotes
# or newlines -- so this stays valid without a JSON library).
ledger_append() {
  local ts line="{" first=1 kv k v
  ts="$(date '+%Y-%m-%dT%H:%M:%S')"
  line+="\"ts\":\"$ts\""
  for kv in "$@"; do
    k="${kv%%=*}"; v="${kv#*=}"
    line+=",\"$k\":\"$v\""
  done
  line+="}"
  printf '%s\n' "$line" >> "$LEDGER_FILE"
}

# A failure is "handled" once any ledger row references its slurm jobid. The scan
# uses this set so it never acts on the same failure twice.
handled_jobids() { sed -n 's/.*"prev_jobid":"\([0-9]\+\)".*/\1/p' "$LEDGER_FILE" | sort -u; }

# How many times has this logical task already been resubmitted?
retries_for() {
  local key="$1"
  grep -c "\"action\":\"resubmitted\".*\"logical_key\":\"$key\"" "$LEDGER_FILE" 2>/dev/null || true
}

# ===========================================================================
# init -- snapshot every job I currently own (the user chose "all my jobs").
# ===========================================================================
cmd_init() {
  local me_job="${SLURM_JOB_ID:-}" id
  : > "$MANAGED_FILE"
  # Never watch the watchdog's own tick jobs (named oom_watchdog) -- whether this
  # runs from a scheduled tick (exclude its own SLURM_JOB_ID) or by hand.
  local exclude; exclude="$(squeue --me -h -n oom_watchdog -O ArrayJobID 2>/dev/null | tr -d ' ' | sort -u)"
  squeue --me -h -O ArrayJobID 2>/dev/null | tr -d ' ' | sort -u | while read -r id; do
    [[ -z "$id" || "$id" == "$me_job" ]] && continue
    grep -qx "$id" <<<"$exclude" && continue
    echo "$id" >> "$MANAGED_FILE"
  done
  local n; n="$(grep -c . "$MANAGED_FILE" || echo 0)"
  log "init: managing $n job/array id(s): $(tr '\n' ' ' < "$MANAGED_FILE")"
}

# ===========================================================================
# capture -- read a running job's submit environment and store a resubmit recipe.
# ===========================================================================
cmd_capture() {
  local arrayid="$1"
  local env_file="$RECIPE_DIR/$arrayid.env" meta_file="$RECIPE_DIR/$arrayid.meta"
  [[ -s "$env_file" && -s "$meta_file" ]] && return 0      # already captured

  # Need a task of this array that is actually RUNNING (only then is /proc live).
  local realid node
  realid="$(squeue -j "$arrayid" -h -t R -O JobID 2>/dev/null | tr -d ' ' | head -1)"
  [[ -n "$realid" ]] || { log "capture: $arrayid has no running task yet; deferring"; return 2; }
  node="$(squeue -j "$realid" -h -O NodeList 2>/dev/null | tr -d ' ' | head -1)"

  log "capture: reading submit env of $arrayid (jobid $realid on $node)"

  # Read the batch script's own environment from inside the allocation. The
  # remote side only locates the job's batch process and base64-dumps its raw
  # /proc/<pid>/environ (base64 so the NUL separators survive srun's stdout). We
  # match the specific job's slurm_script path so we never grab another job on a
  # shared node. All filtering/quoting happens locally (below), where we control
  # the shell options -- doing it remotely is fragile because `set -u` leaks into
  # the srun shell via the auto-exported SHELLOPTS.
  # NB: `--input=none` + `</dev/null` keep srun from forwarding (and thus
  # consuming) our stdin -- without it, calling capture inside a `while read`
  # loop would swallow the rest of the loop's input.
  local captured_b64
  captured_b64="$(srun --jobid="$realid" --overlap --input=none --mem=0 --time=00:02:00 \
      bash -c '
        pid=$(pgrep -u "$USER" -f "job'"$realid"'/slurm_script" | head -1)
        if [ -n "$pid" ] && [ -r /proc/$pid/environ ]; then
          base64 -w0 < /proc/$pid/environ
        else
          echo -n __NO_PID__
        fi
      ' </dev/null 2>/dev/null)" || true

  [[ -n "$captured_b64" && "$captured_b64" != *"__NO_PID__"* ]] \
    || { log "capture: could not read /proc environ for $arrayid (jobid $realid); deferring"; return 2; }

  # Keep ONLY genuine run/project overrides. Everything in the standard Sherlock
  # login/session/module/node-local environment is dropped: the watchdog
  # re-supplies it fresh via `sbatch --export=ALL`, and replaying a stale copy
  # onto a different node is what causes subtle breakage (a dead KRB5CCNAME
  # ticket path, a frozen Lmod module table, /tmp paths from the old node).
  local kv k v
  : > "$env_file"
  while IFS= read -r -d '' kv; do
    k="${kv%%=*}"; v="${kv#*=}"
    case "$k" in
      SLURM*|SLURMD*|SRUN*|SBATCH*|OMPI_MCA_*|PRTE_MCA_*|HYDRA*|I_MPI*) continue ;;
      LMOD*|__LMOD*|_Module*|_LMFILES_|LOADEDMODULES|MODULE*|MODULESHOME|MODULEPATH*|BASH_ENV|__Init_Default_Modules) continue ;;
      SSH_CLIENT|SSH_TTY|SSH_CONNECTION|SSH_AUTH_SOCK|SSH_AGENT_PID) continue ;;
      PATH|LD_LIBRARY_PATH|MANPATH|PKG_CONFIG_PATH|CPATH|LIBRARY_PATH|C_INCLUDE_PATH|CPLUS_INCLUDE_PATH|NODE_PATH|NODE_OPTIONS) continue ;;
      PWD|OLDPWD|SHLVL|SHELL|_|TERM|ENVIRONMENT|LOGNAME|LS_COLORS|LESSOPEN|HIST*|TMOUT|PROMPT_COMMAND|BASH_FUNC*) continue ;;
      TMPDIR|TMP|TEMP|HOSTNAME|HOST|HOSTTYPE|KRB5*|APPTAINER*|PROOT*|PYTHONPYCACHEPREFIX) continue ;;
      L_SCRATCH*|LOCAL_SCRATCH|CUDA_VISIBLE_DEVICES|GPU_DEVICE_ORDINAL|XDG_*) continue ;;
      SCRATCH|GROUP_SCRATCH|GROUP_HOME|HOME|OAK|COMMON_DATASETS|GROUP|USER|MAIL|SHERLOCK|SRCC_PATH|SH_*|LANG|LC_*) continue ;;
      CLAUDE_CODE*|CLAUDECODE|AI_AGENT) continue ;;
      "" ) continue ;;
    esac
    printf 'export %s=%q\n' "$k" "$v" >> "$env_file"
  done < <(printf '%s' "$captured_b64" | base64 -d)

  [[ -s "$env_file" ]] || { log "capture: filtered env for $arrayid is empty; deferring"; rm -f "$env_file"; return 2; }

  # Static job facts come straight from the controller (authoritative).
  local info cmd jobname cpus partition timelimit is_array
  info="$(scontrol show job "$realid" -o 2>/dev/null)" || die "scontrol failed for $realid"
  cmd="$(grep -o 'Command=[^ ]*'   <<<"$info" | head -1 | cut -d= -f2-)"
  jobname="$(grep -o 'JobName=[^ ]*' <<<"$info" | head -1 | cut -d= -f2-)"
  cpus="$(grep -o 'NumCPUs=[0-9]*'  <<<"$info" | head -1 | cut -d= -f2)"
  partition="$(grep -o 'Partition=[^ ]*' <<<"$info" | head -1 | cut -d= -f2)"
  timelimit="$(grep -o 'TimeLimit=[^ ]*'  <<<"$info" | head -1 | cut -d= -f2)"
  is_array=0; grep -q 'ArrayJobId=' <<<"$info" && is_array=1

  [[ -f "$cmd" ]] || die "captured Command for $arrayid is not a readable script: '$cmd'"

  {
    echo "script_path=$cmd"
    echo "job_name=$jobname"
    echo "cpus=${cpus:-8}"
    echo "base_partition=${partition:-normal}"
    echo "time_limit=${timelimit:-1-00:00:00}"
    echo "is_array=$is_array"
  } > "$meta_file"

  log "capture: stored recipe for $arrayid -> $cmd (cpus=${cpus:-8}, array=$is_array)"
}

# ===========================================================================
# scan -- emit unhandled OOM candidates as JSON lines for Claude to act on.
# ===========================================================================
cmd_scan() {
  local handled; handled="$(handled_jobids)"
  local arrayid
  while read -r arrayid; do
    [[ -z "$arrayid" ]] && continue
    scan_one_array "$arrayid" "$handled"
  done < "$MANAGED_FILE"
}

scan_one_array() {
  local arrayid="$1" handled="$2"
  # MaxRSS lives on the ".batch" step; collect it keyed by parent task id first.
  declare -A maxrss
  local rec parent rss state line
  while IFS='|' read -r jid st ec req mr name part cpus; do
    [[ "$jid" == *.batch ]] || continue
    parent="${jid%.batch}"
    maxrss["$parent"]="$mr"
  done < <(sacct -j "$arrayid" --units=G --parsable2 --noheader \
             --format=JobID,State,ExitCode,ReqMem,MaxRSS,JobName,Partition,AllocCPUS 2>/dev/null)

  while IFS='|' read -r jid st ec req mr name part cpus; do
    [[ "$jid" == *.* ]] && continue                  # skip .batch/.extern/.N steps
    # Only failure-ish terminal states are interesting.
    local conf=""
    case "$st" in
      OUT_OF_MEMORY*)            conf="high" ;;
      FAILED*|CANCELLED*|NODE_FAIL*)
        # A SIGKILL (137) or Slurm's OOM exit (125) often *is* an OOM that wasn't
        # tagged; flag as "maybe" so Claude confirms from the log before acting.
        case "$ec" in 0:125|0:9|137:*|9:*|*:125|*:9) conf="maybe" ;; *) conf="" ;; esac ;;
      *) conf="" ;;
    esac
    [[ -z "$conf" ]] && continue

    # Resolve the underlying numeric jobid (for the handled-set check) and the
    # array task index.
    local realid task
    realid="$(sacct -j "$jid" --noheader --parsable2 --format=JobIDRaw 2>/dev/null | head -1)"
    realid="${realid%%.*}"
    if [[ "$jid" == *_* ]]; then task="${jid##*_}"; else task=""; fi

    grep -qx "$realid" <<<"$handled" && continue       # already acted on this failure

    # Map this failure to its logical task + recipe. Resubmitted singletons are
    # recorded in the jobid map; original-array tasks key off "<array>_<task>".
    local logical recipe maprow
    maprow="$(grep -P "^$arrayid\t" "$JOBMAP_FILE" | head -1 || true)"
    if [[ -n "$maprow" ]]; then
      logical="$(cut -f2 <<<"$maprow")"
      recipe="$(cut -f3 <<<"$maprow")"
      task="$(cut -f4 <<<"$maprow")"
    else
      recipe="$arrayid"
      logical="${arrayid}${task:+_$task}"
    fi

    local cur_gb rss_gb retries suggestion sug_mem sug_part exhausted=0
    cur_gb="$(mem_to_gb "$req")"
    rss_gb="$(mem_to_gb "${maxrss[$jid]:-}")"
    retries="$(retries_for "$logical")"; retries="${retries:-0}"
    suggestion="$(next_step "$cur_gb" "$retries")"
    if [[ "$suggestion" == "EXHAUSTED" ]]; then
      exhausted=1; sug_mem=0; sug_part=""
    else
      sug_mem="${suggestion%% *}"; sug_part="${suggestion##* }"
    fi

    local recipe_present=0; [[ -s "$RECIPE_DIR/$recipe.env" ]] && recipe_present=1
    local logf; logf="$(logfile_for "$realid" "$arrayid" "$task")"

    printf '{"arrayid":"%s","task":"%s","jobid":"%s","state":"%s","exitcode":"%s","oom_confidence":"%s","cur_mem_gb":%s,"maxrss_gb":%s,"retries":%s,"recipe":"%s","recipe_present":%s,"logical_key":"%s","exhausted":%s,"suggested_mem_gb":%s,"suggested_partition":"%s","logfile":"%s"}\n' \
      "$arrayid" "$task" "$realid" "$st" "$ec" "$conf" "$cur_gb" "$rss_gb" "$retries" \
      "$recipe" "$recipe_present" "$logical" "$exhausted" "$sug_mem" "$sug_part" "$logf"
  done < <(sacct -j "$arrayid" --units=G --parsable2 --noheader \
             --format=JobID,State,ExitCode,ReqMem,MaxRSS,JobName,Partition,AllocCPUS 2>/dev/null)
}

# Best-effort path to a failed task's log so Claude can confirm the OOM. The
# holdout pipeline writes to $WORK_ROOT/slurm_logs/<name>_<arrayjob>_<task>.out;
# fall back to scontrol's recorded StdOut.
logfile_for() {
  local realid="$1" arrayid="$2" task="$3" out=""
  out="$(scontrol show job "$realid" -o 2>/dev/null | grep -o 'StdOut=[^ ]*' | head -1 | cut -d= -f2- || true)"
  echo "$out"
}

# ===========================================================================
# resubmit -- the only command that submits work. Validates feasibility loudly.
# ===========================================================================
cmd_resubmit() {
  local recipe="$1" task="$2" mem_gb="$3" logical="$4" prev_jobid="$5"
  local env_file="$RECIPE_DIR/$recipe.env" meta_file="$RECIPE_DIR/$recipe.meta"
  [[ -s "$env_file"  ]] || die "no env recipe for '$recipe' ($env_file) -- refusing to guess the submit environment"
  [[ -s "$meta_file" ]] || die "no meta recipe for '$recipe' ($meta_file)"

  [[ "$mem_gb" =~ ^[0-9]+$ ]] || die "memory must be an integer GB, got '$mem_gb'"
  (( mem_gb <= WD_MEM_CAP_GB )) || die "requested ${mem_gb}GB exceeds cap ${WD_MEM_CAP_GB}GB"

  # Strip the watchdog job's own Slurm context so the child gets fresh ids, then
  # replay the captured submit environment.
  unset $(compgen -v | grep -E '^(SLURM|SLURMD|SRUN|SBATCH)' || true) 2>/dev/null || true
  # shellcheck disable=SC1090
  source "$env_file"
  # shellcheck disable=SC1090
  source "$meta_file"

  local cpus="${cpus:-8}" partition; partition="$(partition_for "$mem_gb")"

  # Feasibility: mem must fit the partition's per-core ceiling at this core count.
  local percore=$WD_NORMAL_PERCORE_GB
  [[ "$partition" == bigmem ]] && percore=$WD_BIGMEM_PERCORE_GB
  (( mem_gb <= percore * cpus )) \
    || die "${mem_gb}GB needs more than ${percore}GB/core on $partition with $cpus cores; raise cpus or cap"

  # bigmem tops out at 1 day of walltime.
  local walltime="${time_limit:-1-00:00:00}"
  if [[ "$partition" == bigmem ]] && exceeds_one_day "$walltime"; then
    log "resubmit: clamping walltime $walltime -> 1-00:00:00 for bigmem"
    walltime="1-00:00:00"
  fi

  local logdir="${WORK_ROOT:-$WD_STATE_DIR/resubmit}/slurm_logs"
  mkdir -p "$logdir"

  local -a sbatch_args=(
    --parsable --export=ALL
    --job-name="${job_name:-resubmit}"
    --partition="$partition"
    --cpus-per-task="$cpus"
    --mem="${mem_gb}GB"
    --time="$walltime"
    --output="$logdir/%x_%A_%a.out" --error="$logdir/%x_%A_%a.out"
  )
  [[ "$task" != "" && "${is_array:-0}" == "1" ]] && sbatch_args+=(--array="$task")

  log "resubmit: logical=$logical task=${task:-<none>} ${WD_DRY_RUN:+[DRY] }-> $partition ${mem_gb}GB ${cpus}c, script=$script_path"

  if [[ "$WD_DRY_RUN" == "1" ]]; then
    printf 'DRY RUN sbatch %s %s\n' "${sbatch_args[*]}" "$script_path"
    return 0
  fi

  local new_jobid
  new_jobid="$(sbatch "${sbatch_args[@]}" "$script_path")" \
    || die "sbatch failed for logical=$logical (recipe=$recipe task=$task)"
  new_jobid="${new_jobid%%;*}"          # strip ";cluster" suffix if present
  [[ "$new_jobid" =~ ^[0-9]+$ ]] || die "sbatch returned unexpected id: '$new_jobid'"

  # Persist: ledger row, jobid->logical map, and add the new id to the managed
  # set so the next scan watches it too.
  local old_mem_gb
  old_mem_gb="$(mem_to_gb "$(sacct -j "$prev_jobid" --units=G --noheader --parsable2 --format=ReqMem 2>/dev/null | head -1)")"
  ledger_append action=resubmitted logical_key="$logical" recipe="$recipe" \
    task="${task:-}" old_mem_gb="$old_mem_gb" new_mem_gb="$mem_gb" \
    partition="$partition" prev_jobid="$prev_jobid" new_jobid="$new_jobid"
  printf '%s\t%s\t%s\t1\n' "$new_jobid" "$logical" "$recipe" >> "$JOBMAP_FILE"
  grep -qx "$new_jobid" "$MANAGED_FILE" || echo "$new_jobid" >> "$MANAGED_FILE"

  log "resubmit: submitted new job $new_jobid (logical=$logical) at ${mem_gb}GB on $partition"
  echo "$new_jobid"
}

exceeds_one_day() {
  case "$1" in
    *-*) return 0 ;;                 # has a day field => >= 1 day
    *)   return 1 ;;
  esac
}

# ===========================================================================
# handle -- record a failure we are NOT resubmitting (not an OOM / exhausted /
# no recipe), so the scan stops surfacing it.
# ===========================================================================
cmd_handle() {
  local prev_jobid="$1" task="$2" logical="$3" reason="$4"
  ledger_append action=skipped logical_key="$logical" task="${task:-}" \
    prev_jobid="$prev_jobid" reason="$reason"
  log "handle: skipping $prev_jobid (logical=$logical): $reason"
}

# ===========================================================================
# active -- number of managed jobs still queued or running (loop exit signal).
# ===========================================================================
cmd_active() {
  local me_job="${SLURM_JOB_ID:-}" arrayid total=0
  # One query for all my active jobs, then count managed ids among them. NB:
  # `squeue -j <purged-id>` prints "Invalid job id specified" to STDOUT and exits
  # 0, so a per-id loop would miscount finished jobs as active -- avoid it. The
  # `^[0-9]+$` filter drops any such error text.
  local active_ids
  active_ids="$(squeue --me -h -t R,PD,CF,CG -O ArrayJobID 2>/dev/null | tr -d ' ' | grep -E '^[0-9]+$' | sort -u)"
  while read -r arrayid; do
    [[ -z "$arrayid" || "$arrayid" == "$me_job" ]] && continue
    grep -qx "$arrayid" <<<"$active_ids" && total=$(( total + 1 ))
  done < "$MANAGED_FILE"
  echo "$total"
}

# ===========================================================================
# status -- human summary.
# ===========================================================================
cmd_status() {
  echo "== OOM watchdog status =="
  echo "state dir : $WD_STATE_DIR"
  echo "managing  : $(grep -c . "$MANAGED_FILE" || echo 0) job/array id(s)"
  echo "still live : $(cmd_active) task(s) queued/running"
  echo
  local n_resub n_skip
  n_resub="$(grep -c '"action":"resubmitted"' "$LEDGER_FILE" 2>/dev/null || echo 0)"
  n_skip="$(grep -c '"action":"skipped"' "$LEDGER_FILE" 2>/dev/null || echo 0)"
  echo "resubmitted: $n_resub    skipped: $n_skip"
  echo
  echo "-- ledger --"
  cat "$LEDGER_FILE"
}

# --- dispatch --------------------------------------------------------------
sub="${1:-}"; shift || true
case "$sub" in
  init)     cmd_init "$@" ;;
  capture)  cmd_capture "$@" ;;
  scan)     cmd_scan "$@" ;;
  resubmit) cmd_resubmit "$@" ;;
  handle)   cmd_handle "$@" ;;
  active)   cmd_active "$@" ;;
  status)   cmd_status "$@" ;;
  *) echo "usage: wd.sh {init|capture <id>|scan|resubmit <recipe> <task> <mem_gb> <logical_key> <prev_jobid>|handle <prev_jobid> <task> <logical_key> <reason>|active|status}" >&2; exit 2 ;;
esac
