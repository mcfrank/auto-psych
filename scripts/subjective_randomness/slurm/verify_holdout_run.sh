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

# 1. Every task must have reached its final line.
n_done=0; n_logs=0
for L in "$W"/slurm_logs/holdout_recovery_*.out; do
  [[ -f "$L" ]] || continue
  n_logs=$((n_logs + 1))
  grep -q "\[task .*\] done ->" "$L" && n_done=$((n_done + 1))
done
if [[ "$n_logs" -gt 0 && "$n_done" == "$n_logs" ]]; then say "- [ok]   all $n_logs task(s) finished"
else say "- [FAIL] $n_done of $n_logs task(s) finished"; fails=$((fails + 1)); fi

# 2. THE check for this arm: a raw seed that cannot bind to raw rows is
#    [drop]ped, not fatal, so the run would quietly proceed with fewer models.
n_drop=$(grep -h "\[drop\]" "$W"/slurm_logs/holdout_recovery_*.out 2>/dev/null | wc -l)
if [[ "$n_drop" == "0" ]]; then say "- [ok]   no model dropped (every raw seed bound to raw rows)"
else
  say "- [FAIL] $n_drop dropped model(s) — a seed could not bind:"
  grep -h "\[drop\]" "$W"/slurm_logs/holdout_recovery_*.out 2>/dev/null | sort -u | head -5 | sed 's/^/      /' | tee -a "$V"
  fails=$((fails + 1))
fi

# 3. No traceback, and specifically no feature-column collision.
n_err=$(grep -hE "Traceback|collides with" "$W"/slurm_logs/holdout_recovery_*.out 2>/dev/null | wc -l | tr -d ' ')
if [[ "${n_err:-0}" -eq 0 ]]; then say "- [ok]   no traceback or column collision"
else
  say "- [FAIL] ${n_err} error line(s):"
  grep -hE "Traceback|collides with" "$W"/slurm_logs/holdout_recovery_*.out 2>/dev/null | head -3 | sed 's/^/      /' | tee -a "$V"
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
#    Isolation here is by data, not by import (docs/raw_features_arm.md).
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

# 7. Every design/screened_out.json must be empty ([]).
bad_screen=0; n_screen=0
for SO in "$W"/run*/*/repo/_runs/*/experiment*/design/screened_out.json; do
  [[ -f "$SO" ]] || continue
  n_screen=$((n_screen + 1))
  [[ "$(cat "$SO" | tr -d '[:space:]')" == "[]" ]] \
    || { bad_screen=$((bad_screen + 1)); say "      non-empty: $SO"; }
done
for TAR in "$W"/run*/*/agent_runs.tar.gz; do
  [[ -f "$TAR" ]] || continue
  while IFS= read -r MEMBER; do
    n_screen=$((n_screen + 1))
    [[ "$(tar xzOf "$TAR" "$MEMBER" 2>/dev/null | tr -d '[:space:]')" == "[]" ]] \
      || { bad_screen=$((bad_screen + 1)); say "      non-empty: $TAR :: $MEMBER"; }
  done < <(tar tzf "$TAR" 2>/dev/null | grep -E "design/screened_out\.json$")
done
if [[ "$n_screen" -gt 0 && "$bad_screen" == "0" ]]; then say "- [ok]   all $n_screen screened_out.json(s) are empty"
elif [[ "$bad_screen" -gt 0 ]]; then say "- [FAIL] $bad_screen of $n_screen screened_out.json(s) are non-empty"; fails=$((fails + 1)); fi

# 8. Agent-tree isolation: forbidden paths from agent_tree.exclude must not
#    appear in the agent's repo copy (whether live or archived).
_TREE_FORBIDDEN_ANCHORED=(
  "src/subjective_randomness/features.py"
  "src/subjective_randomness/sequence_stats.py"
  "src/subjective_randomness/stimulus_design.py"
  "src/subjective_randomness/model_recovery.py"
  "src/subjective_randomness/pymc_recover.py"
)
_TREE_FORBIDDEN_UNANCHORED=(ground_truth_models.py evaluate_recovery.py preprocess.py)
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
    done < <(find "$TREPO" -name "$fn" -not -path "*/_runs/*" 2>/dev/null)
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

# 10. Candidate admission: at least one admitted per experiment, no round
#     whose rejections are all "no candidate.py written".
_LEDGER_CHECKER=$(mktemp /tmp/check_ledger_XXXXXX.py)
cat > "$_LEDGER_CHECKER" <<'PYEOF'
"""Check candidate admission health from ledger files.

Reads attempted_hypotheses.jsonl, checks that each experiment admitted at
least one candidate, and that no round's rejections are all 'no candidate.py
written'. Prints a summary and exits non-zero on failure.
"""
import json, sys, re
from collections import defaultdict

fails = 0
ledger_path = sys.argv[1]
lines = open(ledger_path).read().strip().splitlines()
if not lines:
    sys.exit(0)

entries = [json.loads(line) for line in lines if line.strip()]

# Group by experiment (from context field like "experiment2 round 0 candidate 1 lens 0")
exp_admitted = defaultdict(int)
rounds = defaultdict(list)
for e in entries:
    ctx = e.get("context", "")
    m = re.search(r"experiment(\d+)", ctx)
    exp = int(m.group(1)) if m else 0
    if e["outcome"] == "admitted":
        exp_admitted[exp] += 1
    rm = re.search(r"round (\d+)", ctx)
    if rm:
        round_key = (exp, int(rm.group(1)))
        rounds[round_key].append(e)

# Check: at least one admitted per experiment
for exp_num in sorted(exp_admitted.keys()):
    if exp_admitted[exp_num] == 0:
        print(f"FAIL: experiment{exp_num} admitted 0 candidates")
        fails += 1

# Check: no round where ALL rejections are "no candidate.py written"
for (exp, rnd), entries_list in sorted(rounds.items()):
    admitted = [e for e in entries_list if e["outcome"] == "admitted"]
    rejected = [e for e in entries_list if e["outcome"] == "rejected"]
    if not admitted and rejected:
        all_no_file = all("no candidate.py written" in e.get("detail", "") for e in rejected)
        if all_no_file:
            print(f"FAIL: experiment{exp} round {rnd}: all {len(rejected)} rejections are 'no candidate.py written'")
            fails += 1

# Summary
total_admitted = sum(exp_admitted.values())
total_rejected = sum(1 for e in entries if e.get("outcome") == "rejected")
print(f"admitted={total_admitted} rejected={total_rejected} experiments={len(exp_admitted)}")
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

# 11. Recovery, if any cell finished.
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

# 11. Print config, code SHA, feature mode, and harness/agent paths.
say "## Run info"
for CFG in "$W"/run*/*/config.yaml; do
  [[ -f "$CFG" ]] || continue
  say "- config: \`$CFG\`"
  grep -E "raw_features|pool_models_dir|seed_models_dir" "$CFG" 2>/dev/null | sed 's/^/  /' | tee -a "$V"
done
CODE_SHA=$(cd "$W"/run*/*/repo 2>/dev/null && git rev-parse HEAD 2>/dev/null || echo "unknown")
say "- code SHA: $CODE_SHA"
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
