#!/bin/bash
# Verify a finished raw-features run: the checks that separate "this arm is
# invalid" from "this arm gave a disappointing number". Writes VERDICT.md into
# the run root and exits non-zero if any check fails, so a Slurm --mail-type
# tells you which happened without reading anything.
#
# Run it directly, or gate it on the run's last job so it needs no polling:
#   sbatch --dependency=afterany:<analysis_id> --mail-type=END,FAIL \
#          --export=ALL,WORK_ROOT=<root> ... verify_raw_features_run.sh
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

# 4. The agents' CSV must carry the raw columns and nothing else.
bad_csv=0; n_csv=0
while IFS= read -r CSV; do
  n_csv=$((n_csv + 1))
  [[ "$(head -1 "$CSV" | tr -d '\r')" == "$RAW_HEADER" ]] \
    || { bad_csv=$((bad_csv + 1)); say "      not raw: $CSV"; }
done < <(ls "$W"/run*/*/repo/_runs/*/experiment*/data/responses.csv 2>/dev/null)
# A task that SUCCEEDS deletes its repo copy and tars the run tree, so on a
# clean run the loop above finds nothing. Read the archive in that case,
# otherwise this check silently verifies nothing (as it did on smoke 2).
if [[ "$n_csv" == "0" ]]; then
  for TAR in "$W"/run*/*/agent_runs.tar.gz; do
    [[ -f "$TAR" ]] || continue
    while IFS= read -r MEMBER; do
      n_csv=$((n_csv + 1))
      [[ "$(tar xzOf "$TAR" "$MEMBER" 2>/dev/null | head -1 | tr -d '\r')" == "$RAW_HEADER" ]] \
        || { bad_csv=$((bad_csv + 1)); say "      not raw: $TAR :: $MEMBER"; }
    done < <(tar tzf "$TAR" 2>/dev/null | grep -E "experiment[0-9]+/data/responses\.csv$")
  done
fi
if [[ "$n_csv" == "0" ]]; then say "- [warn] no agent-facing responses.csv found (repo copies removed on success?)"
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

# 6. Recovery, if any cell finished.
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
if [[ "$fails" == "0" ]]; then say "**VERDICT: all checks passed.**"; else say "**VERDICT: $fails check(s) FAILED — see above.**"; fi
exit "$fails"

# Regression controls for this script itself (it has produced false failures
# twice, and each one cancelled the gated arm):
#   good run, must exit 0:
#     WORK_ROOT=$SCRATCH/auto-psych/holdout_raw_features_smoke4 bash "$0"
#   run whose seeds were dropped, must exit non-zero:
#     WORK_ROOT=$SCRATCH/auto-psych/holdout_raw_features_smoke2 bash "$0"
