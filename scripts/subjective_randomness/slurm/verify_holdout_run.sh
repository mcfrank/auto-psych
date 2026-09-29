#!/bin/bash
# Verify a finished raw-features run: the checks that separate "this arm is
# invalid" from "this arm gave a disappointing number". Writes VERDICT.md into
# the run root and exits non-zero if any check fails, so a Slurm --mail-type
# tells you which happened without reading anything.
#
# Run it directly, or gate it on the run's last job so it needs no polling:
#   sbatch --dependency=afterany:<analysis_id> --mail-type=END,FAIL \
#          --export=ALL,WORK_ROOT=<root> ... verify_holdout_run.sh
set -uo pipefail
W="${WORK_ROOT:?WORK_ROOT must be set}"
RAW_HEADER="sequence_a,sequence_b,participant_id,trial_index,chose_left"
V="$W/VERDICT.md"
fails=0
say() { echo "$@" | tee -a "$V"; }

: > "$V"
say "# Raw-features run verdict"
say ""
say "- run root: \`$W\`"
say "- checked: $(date '+%Y-%m-%d %H:%M:%S %Z')"
say ""

say "## Slurm"
say '```'
sacct -X -n --name=holdout_setup,holdout_recovery,holdout_test_retest \
      --starttime=$(date -d '3 days ago' +%Y-%m-%d) \
      --format=JobID%14,JobName%20,State%12,Elapsed 2>/dev/null \
  | grep -E "$(basename "$W")" >/dev/null 2>&1 || true
sacct -X -n -j "${ALL_JOB_IDS:-0}" --format=JobID%14,JobName%20,State%12,Elapsed 2>/dev/null | tee -a "$V"
say '```'
say ""

say "## Checks"
n_cells=$(ls -d "$W"/run*/*/ 2>/dev/null | wc -l)
say "- cells present: $n_cells"

# 1. Every cell must have its result. Cells are judged by their results
#    (holdout.json, and the summary job's MISSING_CELLS.txt), not by logs: a
#    cell that failed and was resumed by a retry leaves the failed attempt's
#    log behind, which used to FAIL every sweep that needed a retry.
n_result=0; unfinished=()
for C in "$W"/run*/*/; do
  [[ -d "$C" ]] || continue
  C="${C%/}"
  if [[ -f "$C/holdout.json" ]]; then n_result=$((n_result + 1))
  else unfinished+=("$(basename "$(dirname "$C")")/$(basename "$C")"); fi
done
MISSING_FILE="$W/MISSING_CELLS.txt"
if [[ -s "$MISSING_FILE" ]]; then
  say "- [FAIL] $(wc -l < "$MISSING_FILE" | tr -d ' ') expected cell(s) have no result (MISSING_CELLS.txt):"
  head -5 "$MISSING_FILE" | sed 's/^/      /' | tee -a "$V"
  fails=$((fails + 1))
elif [[ "${#unfinished[@]}" -gt 0 ]]; then
  say "- [FAIL] ${#unfinished[@]} cell(s) have no holdout.json: ${unfinished[*]}"
  fails=$((fails + 1))
elif [[ "$n_result" == "0" ]]; then
  say "- [FAIL] no cell has a result"
  fails=$((fails + 1))
else say "- [ok]   all $n_result cell(s) have a result"; fi

# The task logs checks 2 and 3 read: every attempt that finished, and every
# attempt whose cell has no result. A failed attempt of a cell that a later
# attempt finished is superseded (its log is listed, not judged). The cell is
# the directory in the task's "[task T] repeat=R gt=G seed=S -> <dir>" line.
judged_logs=(); superseded=()
for L in "$W"/slurm_logs/holdout_recovery_*.out; do
  [[ -f "$L" ]] || continue
  if ! grep -q "\[task .*\] done ->" "$L"; then
    cell=$(sed -n 's/^\[task [0-9]*\] repeat=.* -> \(.*\)$/\1/p' "$L" | head -1)
    if [[ -n "$cell" && -f "$cell/holdout.json" ]]; then superseded+=("$L"); continue; fi
  fi
  judged_logs+=("$L")
done
if [[ "${#superseded[@]}" -gt 0 ]]; then
  say "- [info] ${#superseded[@]} failed attempt(s) of cells a later attempt finished (their logs are not judged):"
  printf '%s\n' "${superseded[@]}" | head -5 | sed 's/^/      /' | tee -a "$V"
fi
_judged_grep() {  # grep -h over the judged logs; nothing when there are none
  [[ "${#judged_logs[@]}" -gt 0 ]] || return 0
  grep -h "$@" "${judged_logs[@]}" 2>/dev/null
}

# 2. THE check for this arm: a raw seed that cannot bind to raw rows is
#    [drop]ped, not fatal, so the run would quietly proceed with fewer models.
n_drop=$(_judged_grep "\[drop\]" | wc -l)
if [[ "$n_drop" == "0" ]]; then say "- [ok]   no model dropped (every raw seed bound to raw rows)"
else
  say "- [FAIL] $n_drop dropped model(s) — a seed could not bind:"
  _judged_grep "\[drop\]" | sort -u | head -5 | sed 's/^/      /' | tee -a "$V"
  fails=$((fails + 1))
fi

# 3. No traceback, and specifically no feature-column collision.
n_err=$(_judged_grep -E "Traceback|collides with" | wc -l | tr -d ' ')
if [[ "${n_err:-0}" -eq 0 ]]; then say "- [ok]   no traceback or column collision"
else
  say "- [FAIL] ${n_err} error line(s):"
  _judged_grep -E "Traceback|collides with" | head -3 | sed 's/^/      /' | tee -a "$V"
  fails=$((fails + 1))
fi

# 4. The agents' CSVs must carry the raw columns and nothing else.
#    Check BOTH data/responses.csv AND model_loop/responses.csv — arm C's bug
#    was that data/ was raw but model_loop/ was featurized.
bad_csv=0; n_csv=0
_check_csv_header() {
  local CSV="$1" LABEL="$2"
  n_csv=$((n_csv + 1))
  [[ "$(head -1 "$CSV" | tr -d '\r')" == "$RAW_HEADER" ]] \
    || { bad_csv=$((bad_csv + 1)); say "      not raw: $LABEL"; }
}

# Live repo copies (before archival).
for CSV in "$W"/run*/*/repo/_runs/*/experiment*/data/responses.csv \
           "$W"/run*/*/repo/_runs/*/experiment*/model_loop/responses.csv; do
  [[ -f "$CSV" ]] || continue
  _check_csv_header "$CSV" "$CSV"
done
# Archived runs (the common case: successful tasks tar and remove the repo).
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  while IFS= read -r MEMBER; do
    n_csv=$((n_csv + 1))
    [[ "$(tar xzOf "$TAR" "$MEMBER" 2>/dev/null | head -1 | tr -d '\r')" == "$RAW_HEADER" ]] \
      || { bad_csv=$((bad_csv + 1)); say "      not raw: $TAR :: $MEMBER"; }
  done < <(tar tzf "$TAR" 2>/dev/null | grep -E "experiment[0-9]+/(data|model_loop)/responses\.csv$")
done
if [[ "$n_csv" == "0" ]]; then
  say "- [FAIL] no agent-facing responses CSV found — cannot verify the arm"
  fails=$((fails + 1))
elif [[ "$bad_csv" == "0" ]]; then say "- [ok]   all $n_csv agent CSV(s) carry only the raw five columns"
else say "- [FAIL] $bad_csv of $n_csv agent CSV(s) carry extra columns"; fails=$((fails + 1)); fi

# 5. Did any candidate import the featurizer this arm removed from the data?
#    Isolation here is by data, not by import.
IMPORT_RE='^[[:space:]]*(from|import)[[:space:]].*(subjective_randomness\.features|featurize_stimulus)'
n_imp=$(grep -rlE "$IMPORT_RE" \
        "$W"/run*/*/repo/_runs/*/experiment*/model_loop/models/*.py 2>/dev/null | wc -l)
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  n_imp=$((n_imp + $(tar xzOf "$TAR" --wildcards "*/model_loop/models/*.py" 2>/dev/null \
            | grep -cE "$IMPORT_RE" || true)))
done
if [[ "$n_imp" == "0" ]]; then say "- [ok]   no candidate imported the project featurizer"
else say "- [FAIL] $n_imp candidate(s) imported the featurizer — the arm's numbers are not a raw-features result"; fails=$((fails + 1)); fi

# 6. Check candidate CONTEXT.md files do not list engineered column names.
bad_ctx=0; n_ctx=0
# Common engineered column prefixes that must NOT appear in a raw run's context.
CTX_BAD_RE='(rep_motifs|occ_n20|multiscale_imbalance|p_alts|periodicity)'
for CTX in "$W"/run*/*/repo/_runs/*/experiment*/model_loop/*/candidate*/CONTEXT.md; do
  [[ -f "$CTX" ]] || continue
  n_ctx=$((n_ctx + 1))
  grep -qE "$CTX_BAD_RE" "$CTX" && { bad_ctx=$((bad_ctx + 1)); say "      engineered cols in: $CTX"; }
done
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  while IFS= read -r MEMBER; do
    n_ctx=$((n_ctx + 1))
    tar xzOf "$TAR" "$MEMBER" 2>/dev/null | grep -qE "$CTX_BAD_RE" \
      && { bad_ctx=$((bad_ctx + 1)); say "      engineered cols in: $TAR :: $MEMBER"; }
  done < <(tar tzf "$TAR" 2>/dev/null | grep -E "CONTEXT\.md$")
done
if [[ "$n_ctx" == "0" ]]; then say "- [info] no candidate CONTEXT.md found (may be normal for short runs)"
elif [[ "$bad_ctx" == "0" ]]; then say "- [ok]   all $n_ctx candidate CONTEXT.md(s) list only raw columns"
else say "- [FAIL] $bad_ctx of $n_ctx candidate CONTEXT.md(s) list engineered columns"; fails=$((fails + 1)); fi

# 7. Every design/screened_out.json must be empty ([]), except for models
#    screened out because their p_left is undefined (NaN or outside [0, 1]) on
#    some design-pool pairs (entries with "invalid_pairs"): those are recorded
#    and the design went on without them, so they are a warning, not a
#    failure. Any other entry (a model that cannot bind a stimulus row) fails.
bad_screen=0; n_screen=0; n_undefined=0
_classify_screen() {  # stdin: one screened_out.json; $1: label
  local counts total undefined
  counts=$(python3 -c 'import json,sys; d=json.load(sys.stdin); print(len(d), sum("invalid_pairs" in e for e in d))' 2>/dev/null) \
    || { bad_screen=$((bad_screen + 1)); say "      unreadable: $1"; return; }
  read -r total undefined <<< "$counts"
  n_undefined=$((n_undefined + undefined))
  if [[ "$total" -gt "$undefined" ]]; then bad_screen=$((bad_screen + 1)); say "      non-empty: $1"; fi
  if [[ "$undefined" -gt 0 ]]; then say "      undefined p_left, screened out of a design: $1"; fi
}
for SO in "$W"/run*/*/repo/_runs/*/experiment*/design/screened_out.json; do
  [[ -f "$SO" ]] || continue
  n_screen=$((n_screen + 1))
  _classify_screen "$SO" < "$SO"
done
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  while IFS= read -r MEMBER; do
    n_screen=$((n_screen + 1))
    _classify_screen "$TAR :: $MEMBER" < <(tar xzOf "$TAR" "$MEMBER" 2>/dev/null)
  done < <(tar tzf "$TAR" 2>/dev/null | grep -E "design/screened_out\.json$")
done
if [[ "$bad_screen" -gt 0 ]]; then say "- [FAIL] $bad_screen of $n_screen screened_out.json(s) are non-empty"; fails=$((fails + 1))
elif [[ "$n_undefined" -gt 0 ]]; then say "- [WARN] $n_undefined model(s) screened out of a design because their p_left is undefined on some pairs (a warning, not a failure)"
elif [[ "$n_screen" -gt 0 ]]; then say "- [ok]   all $n_screen screened_out.json(s) are empty"; fi

# 8. Agent-tree isolation: forbidden paths from agent_tree.exclude must not
#    appear in the agent's repo copy (whether live or archived).
_TREE_FORBIDDEN_ANCHORED=(
  "src/subjective_randomness/features.py"
  "src/subjective_randomness/stimulus_design.py"
)
# CLAUDE.md / AGENTS.md: agent CLIs load them into every session, and the
# project's CLAUDE.md names the held-out model.
_TREE_FORBIDDEN_UNANCHORED=(ground_truth_models.py evaluate_recovery.py preprocess.py CLAUDE.md AGENTS.md)
_TREE_FORBIDDEN_DIRS=("src/subjective_randomness/model_families")
bad_tree=0; n_tree=0
for TREPO in "$W"/run*/*/repo; do
  [[ -d "$TREPO" ]] || continue
  n_tree=$((n_tree + 1))
  for fp in "${_TREE_FORBIDDEN_ANCHORED[@]}"; do
    [[ -f "$TREPO/$fp" ]] && { bad_tree=$((bad_tree + 1)); say "      leaked: $TREPO/$fp"; }
  done
  for fd in "${_TREE_FORBIDDEN_DIRS[@]}"; do
    [[ -d "$TREPO/$fd" ]] && { bad_tree=$((bad_tree + 1)); say "      leaked dir: $TREPO/$fd"; }
  done
  for fn in "${_TREE_FORBIDDEN_UNANCHORED[@]}"; do
    while IFS= read -r found; do
      bad_tree=$((bad_tree + 1)); say "      leaked: $found"
    # -H: run<r>/<gt>/repo is a symlink to the real tree under agent_trees/,
    # and find does not descend into a symlinked starting point without it.
    done < <(find -H "$TREPO" -name "$fn" -not -path "*/_runs/*" 2>/dev/null)
  done
done
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  n_tree=$((n_tree + 1))
  listing=$(tar tzf "$TAR" 2>/dev/null || true)
  for fp in "${_TREE_FORBIDDEN_ANCHORED[@]}"; do
    echo "$listing" | grep -qF "$fp" \
      && { bad_tree=$((bad_tree + 1)); say "      leaked in archive: $TAR :: $fp"; }
  done
  for fn in "${_TREE_FORBIDDEN_UNANCHORED[@]}"; do
    echo "$listing" | grep -q "/${fn}$" \
      && { bad_tree=$((bad_tree + 1)); say "      leaked in archive: $TAR :: $fn"; }
  done
done
if [[ "$n_tree" == "0" ]]; then say "- [info] no agent tree or archive found (check manually)"
elif [[ "$bad_tree" == "0" ]]; then say "- [ok]   agent-tree isolation: no forbidden paths in $n_tree tree(s)/archive(s)"
else say "- [FAIL] $bad_tree forbidden path(s) in agent trees"; fails=$((fails + 1)); fi

# 9. Import allowlist: every .py in model_loop/models/ (including pruned/) must
#    import only from the candidate allowlist. Uses system python3 (stdlib only).
_PY3=$(command -v python3 2>/dev/null || echo "")
if [[ -z "$_PY3" ]]; then
  say "- [skip] import allowlist check (python3 not found on this node)"
else
  _IMPORT_CHECKER=$(mktemp /tmp/check_imports_XXXXXX.py)
  cat > "$_IMPORT_CHECKER" <<'PYEOF'
import ast, sys
ALLOWLIST = frozenset({
    "numpy", "pymc", "pytensor", "arviz", "scipy", "math",
    "itertools", "functools", "collections", "re", "typing",
    "dataclasses", "statistics", "operator",
})
path = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] != "-" else "/dev/stdin"
try:
    source = open(path).read()
    tree = ast.parse(source)
except Exception:
    sys.exit(0)
forbidden = []
for node in ast.walk(tree):
    if isinstance(node, ast.Import):
        for alias in node.names:
            top = alias.name.split(".")[0]
            if top not in ALLOWLIST:
                forbidden.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        if node.level:
            forbidden.append(f"relative(level={node.level})")
        elif node.module:
            top = node.module.split(".")[0]
            if top not in ALLOWLIST:
                forbidden.append(node.module)
if forbidden:
    print(" ".join(sorted(set(forbidden))))
    sys.exit(1)
PYEOF
  bad_imp=0; n_imp_files=0
  for MODELS_DIR in "$W"/run*/*/repo/_runs/*/experiment*/model_loop/models; do
    [[ -d "$MODELS_DIR" ]] || continue
    while IFS= read -r PY; do
      n_imp_files=$((n_imp_files + 1))
      result=$("$_PY3" "$_IMPORT_CHECKER" "$PY" 2>/dev/null) \
        || { bad_imp=$((bad_imp + 1)); say "      forbidden import: $PY — $result"; }
    done < <(find "$MODELS_DIR" -name "*.py" -not -name "__init__.py" 2>/dev/null)
  done
  for TAR in "$W"/run*/*/agent_runs.tar.gz; do
    [[ -f "$TAR" ]] || continue
    while IFS= read -r MEMBER; do
      n_imp_files=$((n_imp_files + 1))
      result=$(tar xzOf "$TAR" "$MEMBER" 2>/dev/null \
        | "$_PY3" "$_IMPORT_CHECKER" - 2>/dev/null) \
        || { bad_imp=$((bad_imp + 1)); say "      forbidden import: $TAR :: $MEMBER — $result"; }
    done < <(tar tzf "$TAR" 2>/dev/null | grep -E "model_loop/models/.*\.py$" | grep -v "__init__\.py")
  done
  rm -f "$_IMPORT_CHECKER"
  if [[ "$n_imp_files" == "0" ]]; then say "- [info] no model .py files found for import check"
  elif [[ "$bad_imp" == "0" ]]; then say "- [ok]   all $n_imp_files model .py file(s) pass the import allowlist"
  else say "- [FAIL] $bad_imp of $n_imp_files model .py file(s) import outside the allowlist"; fails=$((fails + 1)); fi
fi

# 10. Candidate admission: at least one admitted per experiment. Abandoned
#     rounds (retried then skipped) are a warning; only excessive abandonment
#     is a failure. Threshold: more than half the rounds in any experiment.
_LEDGER_CHECKER=$(mktemp /tmp/check_ledger_XXXXXX.py)
cat > "$_LEDGER_CHECKER" <<'PYEOF'
"""Check candidate admission health from ledger files.

Reads attempted_hypotheses.jsonl, checks that each experiment admitted at
least one candidate, counts round_abandoned entries, and reports warnings
for retried rounds. Fails on: zero admissions per experiment, or more than
half the rounds in an experiment abandoned.
"""
import json, sys, re
from collections import defaultdict

fails = 0
warns = 0
ledger_path = sys.argv[1]
lines = open(ledger_path).read().strip().splitlines()
if not lines:
    sys.exit(0)

entries = [json.loads(line) for line in lines if line.strip()]

exp_admitted = defaultdict(int)
exp_abandoned = defaultdict(int)
exp_rounds = defaultdict(set)
for e in entries:
    ctx = e.get("context", "")
    m = re.search(r"experiment(\d+)", ctx)
    exp = int(m.group(1)) if m else 0
    if e["outcome"] == "admitted":
        exp_admitted[exp] += 1
    if e["outcome"] == "round_abandoned":
        exp_abandoned[exp] += 1
    rm = re.search(r"round (\d+)", ctx)
    if rm:
        exp_rounds[exp].add(int(rm.group(1)))

all_exps = sorted(set(exp_admitted) | set(exp_abandoned) | set(exp_rounds))

for exp_num in all_exps:
    if exp_admitted[exp_num] == 0 and exp_abandoned[exp_num] == 0:
        print(f"FAIL: experiment{exp_num} admitted 0 candidates")
        fails += 1

n_abandoned_total = sum(exp_abandoned.values())
if n_abandoned_total > 0:
    for exp_num in all_exps:
        n_abn = exp_abandoned[exp_num]
        n_rnd = len(exp_rounds[exp_num])
        if n_abn > 0:
            if n_rnd > 0 and n_abn > n_rnd // 2:
                print(f"FAIL: experiment{exp_num}: {n_abn} of {n_rnd} rounds abandoned (threshold: >{n_rnd // 2})")
                fails += 1
            else:
                print(f"WARN: experiment{exp_num}: {n_abn} round(s) abandoned (retried, then skipped)")
                warns += 1

total_admitted = sum(exp_admitted.values())
total_rejected = sum(1 for e in entries if e.get("outcome") == "rejected")
print(f"admitted={total_admitted} rejected={total_rejected} abandoned_rounds={n_abandoned_total} experiments={len(all_exps)}")
sys.exit(fails)
PYEOF

if [[ -n "$_PY3" ]]; then
  bad_ledger=0; n_ledger=0
  # Live copies
  for LEDGER in "$W"/run*/*/repo/_runs/*/experiment*/model_loop/attempted_hypotheses.jsonl; do
    [[ -f "$LEDGER" ]] || continue
    n_ledger=$((n_ledger + 1))
    result=$("$_PY3" "$_LEDGER_CHECKER" "$LEDGER" 2>/dev/null) \
      || { bad_ledger=$((bad_ledger + 1)); say "      $LEDGER: $result"; }
  done
  # Archived runs: extract each ledger and check
  for TAR in "$W"/run*/*/agent_runs.tar.gz; do
    [[ -f "$TAR" ]] || continue
    while IFS= read -r MEMBER; do
      n_ledger=$((n_ledger + 1))
      TMPLEDGER=$(mktemp /tmp/ledger_XXXXXX.jsonl)
      tar xzOf "$TAR" "$MEMBER" > "$TMPLEDGER" 2>/dev/null || true
      result=$("$_PY3" "$_LEDGER_CHECKER" "$TMPLEDGER" 2>/dev/null) \
        || { bad_ledger=$((bad_ledger + 1)); say "      $TAR :: $MEMBER: $result"; }
      rm -f "$TMPLEDGER"
    done < <(tar tzf "$TAR" 2>/dev/null | grep -E "model_loop/attempted_hypotheses\.jsonl$" | grep -v "cognitive_models")
  done
  rm -f "$_LEDGER_CHECKER"
  if [[ "$n_ledger" == "0" ]]; then say "- [info] no ledger files found for admission check"
  elif [[ "$bad_ledger" == "0" ]]; then say "- [ok]   candidate admission: all $n_ledger ledger(s) show healthy admission"
  else say "- [FAIL] $bad_ledger of $n_ledger ledger(s) show candidate write failures"; fails=$((fails + 1)); fi
else
  say "- [skip] candidate admission check (python3 not found)"
fi

# 11. Critique presence: every round's history.json entry records the critique
#     status it ran with (P35: "critiqued" with the agent's statistic counts,
#     "no_critique", or "disabled"). A finished run in which NO round of an
#     experiment produced a critique is flagged as a WARNING, not a failure —
#     the loop is designed to proceed without a critique, and a false failure
#     here has cancelled a gated arm before. The point is visibility: the old
#     fallback battery made every archived round look critiqued (one
#     pipeline-written statistic, the marginal choice rate) while the critique
#     agent had never once produced a statistic.
_CRITIQUE_CHECKER=$(mktemp /tmp/check_critique_XXXXXX.py)
cat > "$_CRITIQUE_CHECKER" <<'PYEOF'
"""Summarise the per-round critique status of one inner-loop history.json.

Prints one line of counts. Exit 0: at least one round was critiqued (or the
run had no rounds). Exit 1: rounds exist and none produced a critique. Exit 2:
the rounds predate the status record (nothing to check). Exit 3: a status
value outside the vocabulary — a pipeline bug, reported as a failure.
"""
import json, sys

KNOWN = ("critiqued", "no_critique", "disabled")
history = json.load(open(sys.argv[1]))
rounds = [e for e in history if e.get("iteration") is not None]
counts = {k: 0 for k in KNOWN}
unrecorded = 0
unknown = []
for e in rounds:
    critique = e.get("critique")
    if critique is None:
        unrecorded += 1
        continue
    status = critique.get("status")
    if status in counts:
        counts[status] += 1
    else:
        unknown.append(repr(status))
print(
    f"rounds={len(rounds)} " + " ".join(f"{k}={v}" for k, v in counts.items())
    + f" unrecorded={unrecorded}"
)
if unknown:
    print(f"unknown critique status: {', '.join(unknown)}")
    sys.exit(3)
if not rounds or counts["critiqued"] > 0:
    sys.exit(0)
if unrecorded == len(rounds):
    sys.exit(2)
sys.exit(1)
PYEOF

if [[ -n "$_PY3" ]]; then
  n_hist=0; n_nocrit=0; n_unrec=0; n_badcrit=0
  _check_history_critique() {  # $1 = path to a history.json, $2 = label for the verdict
    local result rc
    n_hist=$((n_hist + 1))
    result=$("$_PY3" "$_CRITIQUE_CHECKER" "$1" 2>&1); rc=$?
    case "$rc" in
      0) ;;
      1) n_nocrit=$((n_nocrit + 1)); say "      no critique in any round: $2 ($result)";;
      2) n_unrec=$((n_unrec + 1));;
      *) n_badcrit=$((n_badcrit + 1)); say "      unreadable critique status: $2 ($result)";;
    esac
  }
  for H in "$W"/run*/*/repo/_runs/*/experiment*/model_loop/history.json; do
    [[ -f "$H" ]] || continue
    _check_history_critique "$H" "$H"
  done
  for TAR in "$W"/run*/*/agent_runs.tar.gz; do
    [[ -f "$TAR" ]] || continue
    while IFS= read -r MEMBER; do
      TMPHIST=$(mktemp /tmp/history_XXXXXX.json)
      tar xzOf "$TAR" "$MEMBER" > "$TMPHIST" 2>/dev/null || true
      _check_history_critique "$TMPHIST" "$TAR :: $MEMBER"
      rm -f "$TMPHIST"
    done < <(tar tzf "$TAR" 2>/dev/null | grep -E "experiment[0-9]+/model_loop/history\.json$")
  done
  rm -f "$_CRITIQUE_CHECKER"
  if [[ "$n_hist" == "0" ]]; then say "- [info] no inner-loop history.json found for the critique check"
  elif [[ "$n_badcrit" -gt 0 ]]; then say "- [FAIL] $n_badcrit of $n_hist history.json(s) carry an unreadable critique status"; fails=$((fails + 1))
  elif [[ "$n_nocrit" -gt 0 ]]; then say "- [WARN] $n_nocrit of $n_hist experiment(s) produced no critique in any round — CriticAL contributed nothing there (a warning, not a failure)"
  elif [[ "$n_unrec" == "$n_hist" ]]; then say "- [info] critique status not recorded (run predates the per-round record)"
  else say "- [ok]   critique: every experiment ($n_hist) had a round with agent-written statistics"; fi
else
  rm -f "$_CRITIQUE_CHECKER"
  say "- [skip] critique check (python3 not found)"
fi

# 12. Incumbent changes: the primary metric for improving the loop. A cell's
#     scoring steps are its experiments' history.json entries in order; the
#     incumbent at a step is its best_model (the model the loop exports and
#     carries). A finished cell in which the incumbent NEVER changed is flagged
#     as a WARNING, not a failure — zero is the true value of every archived
#     motif_stack cell (0 changes over 27 steps) and must not block a run. The
#     per-cell line also counts steps at which the incumbent was a discovered
#     model, i.e. not among the models scored at experiment 1's seed step.
#     Mirrors src/subjective_randomness/incumbent.py (stdlib only here).
_INCUMBENT_CHECKER=$(mktemp /tmp/check_incumbent_XXXXXX.py)
cat > "$_INCUMBENT_CHECKER" <<'PYEOF'
"""Count incumbent changes over one cell's history.json files (experiment order).

Prints one line of counts. Exit 0: the incumbent changed at least once (or
there are fewer than two steps). Exit 1: two or more steps and no change.
Exit 3: a history that is not a whole experiment record (no seed step, no
scored models, an entry without best_model) — a pipeline bug, reported as a
failure.
"""
import json, sys

steps = []
starting = None
for exp_num, path in enumerate(sys.argv[1:], start=1):
    history = json.load(open(path))
    if not history:
        print(f"experiment {exp_num}: empty history")
        sys.exit(3)
    if exp_num == 1:
        first = history[0]
        if first.get("step") != 0 or first.get("iteration") is not None or not first.get("posteriors"):
            print("experiment 1 does not open with a seed step that scored models")
            sys.exit(3)
        starting = set(first["posteriors"])
    for entry in history:
        if "best_model" not in entry:
            print(f"experiment {exp_num} step {entry.get('step')!r}: no best_model")
            sys.exit(3)
        steps.append(entry["best_model"])
changes = sum(1 for a, b in zip(steps, steps[1:]) if a != b)
discovered = sum(1 for m in steps if m not in starting)
print(f"steps={len(steps)} changes={changes} discovered_steps={discovered} final={steps[-1]}")
sys.exit(0 if changes > 0 or len(steps) < 2 else 1)
PYEOF

if [[ -n "$_PY3" ]]; then
  n_inc_cells=0; n_frozen=0; n_badinc=0
  _check_cell_incumbent() {  # $1 = cell label, $2.. = history.json paths in experiment order
    local label="$1" result rc; shift
    n_inc_cells=$((n_inc_cells + 1))
    result=$("$_PY3" "$_INCUMBENT_CHECKER" "$@" 2>&1); rc=$?
    case "$rc" in
      0) say "      $label: $result";;
      1) n_frozen=$((n_frozen + 1)); say "      $label: $result  <- incumbent never changed";;
      *) n_badinc=$((n_badinc + 1)); say "      $label: unreadable history ($result)";;
    esac
  }
  for CELL in "$W"/run*/*/; do
    CELL="${CELL%/}"
    label="$(basename "$(dirname "$CELL")")/$(basename "$CELL")"
    TAR="$CELL/agent_runs.tar.gz"
    if [[ -f "$TAR" ]]; then
      # Archived cell: pull each experiment's history.json into a temp dir,
      # numerically ordered (experiment10 after experiment9).
      members=$(tar tzf "$TAR" 2>/dev/null | grep -E "(^|/)experiment[0-9]+/model_loop/history\.json$" | sort -V)
      [[ -n "$members" ]] || continue
      TMPINC=$(mktemp -d /tmp/incumbent_XXXXXX)
      paths=()
      while IFS= read -r MEMBER; do
        out="$TMPINC/$(echo "$MEMBER" | tr '/' '_')"
        tar xzOf "$TAR" "$MEMBER" > "$out" 2>/dev/null || true
        paths+=("$out")
      done <<< "$members"
      _check_cell_incumbent "$label" "${paths[@]}"
      rm -rf "$TMPINC"
    else
      # Kept repo copy (KEEP_REPO_COPY=1): read the live run tree.
      live=$(ls "$CELL"/repo/_runs/*/experiment*/model_loop/history.json 2>/dev/null | sort -V)
      [[ -n "$live" ]] || continue
      paths=()
      while IFS= read -r H; do paths+=("$H"); done <<< "$live"
      _check_cell_incumbent "$label" "${paths[@]}"
    fi
  done
  rm -f "$_INCUMBENT_CHECKER"
  if [[ "$n_inc_cells" == "0" ]]; then say "- [info] no inner-loop history.json found for the incumbent check"
  elif [[ "$n_badinc" -gt 0 ]]; then say "- [FAIL] $n_badinc of $n_inc_cells cell(s) have an unreadable history for the incumbent check"; fails=$((fails + 1))
  elif [[ "$n_frozen" -gt 0 ]]; then say "- [WARN] $n_frozen of $n_inc_cells cell(s) never changed incumbent — the exported best model was the same at every scoring step (a warning, not a failure; the plan's baseline is 0 changes)"
  else say "- [ok]   incumbent: changed at least once in every cell ($n_inc_cells)"; fi
else
  rm -f "$_INCUMBENT_CHECKER"
  say "- [skip] incumbent check (python3 not found)"
fi

# 13. Recovery, if any cell finished.
say ""
say "## Recovery (final-step pearson r per cell)"
say '```'
for T in "$W"/run*/*/holdout.csv; do
  [[ -f "$T" ]] || continue
  cell=$(basename "$(dirname "$(dirname "$T")")")/$(basename "$(dirname "$T")")
  awk -F, 'NR>1{r=$7; b=$6} END{printf "%-42s final r=%-8s best=%s\n", cell, r, b}' cell="$cell" "$T" | tee -a "$V"
done
say '```'
say ""

# 14. Print config, code SHA, feature mode, and harness/agent paths.
say "## Run info"
for CFG in "$W"/run*/*/config.yaml; do
  [[ -f "$CFG" ]] || continue
  say "- config: \`$CFG\`"
  grep -E "raw_features|pool_models_dir|seed_models_dir" "$CFG" 2>/dev/null | sed 's/^/  /' | tee -a "$V"
done
# The code the sweep staged, and any cell recorded on other code (the array
# refuses to resume one, so this should never list anything).
say "- code: $(cat "$W/code_commit" 2>/dev/null || echo "unknown (no code_commit)")"
for CC in "$W"/run*/*/code_commit; do
  [[ -f "$CC" ]] || continue
  [[ "$(cat "$CC")" == "$(cat "$W/code_commit" 2>/dev/null)" ]] \
    || say "      cell on other code: $(dirname "$CC") ($(cat "$CC"))"
done
HARNESS_ROOT="$W/harness_repo"
if [[ -d "$HARNESS_ROOT" ]]; then say "- harness-root: \`$HARNESS_ROOT\`"
else say "- harness-root: not found (pre-P10 run or cleaned up)"; fi
n_agent_trees=$(ls -d "$W"/run*/*/repo 2>/dev/null | wc -l)
say "- agent-root trees: $n_agent_trees live (archived copies not counted)"
say ""

if [[ "$fails" == "0" ]]; then say "**VERDICT: all checks passed.**"; else say "**VERDICT: $fails check(s) FAILED — see above.**"; fi
exit "$fails"

# Regression controls for this script itself (it has produced false failures
# twice, and each one cancelled the gated arm):
#   good run, must exit 0:
#     WORK_ROOT=$SCRATCH/auto-psych/holdout_raw_features_smoke4 bash "$0"
#   run whose seeds were dropped, must exit non-zero:
#     WORK_ROOT=$SCRATCH/auto-psych/holdout_raw_features_smoke2 bash "$0"
