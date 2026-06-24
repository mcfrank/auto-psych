#!/bin/bash
# Re-score finished holdout-recovery runs on ONE common exhaustive eval pool.
#
# Across the held-out model-recovery and inner-loop-ablation studies (and their
# impossible-theory variants), some runs measured recovery on a SAMPLED held-out
# pool (e.g. 500 pairs at lengths 6 & 8) and others on the EXHAUSTIVE pool (every
# distinct unordered pair over all sequences up to length 8). That makes their
# RMSE / Pearson-r numbers incomparable. This submits two chained jobs:
#
#   1. setup - build the reanalysis venv once under $WORK_ROOT
#   2. array - one task per finished holdout.json, re-scored on the exhaustive
#              pool (lengths 1..8) IN PLACE
#
# Reanalysis hits each run's MCMC cache (no agents, no resampling) and overwrites
# that run's holdout.{json,csv,png}. The impossible runs were already exhaustive,
# so re-scoring them is ~idempotent; the sampled-pool runs are brought onto the
# same pool. AFTERWARD, re-run your test-retest / combined collectors (e.g.
# collect_holdout_results.py, collect_no_inner_loop_results.sh) to roll the
# exhaustive numbers up into data/results.
#
# IMPORTANT: only run this once the studies' run arrays have FINISHED — it
# rewrites every holdout.json it finds, and a run that is still in flight has no
# holdout.json yet (so it would be silently skipped).
#
# Usage:
#   # sweep the given study roots on $SCRATCH (dirs and/or explicit holdout.json):
#   bash scripts/subjective_randomness/slurm/submit_reanalyze_exhaustive.sh \
#     $SCRATCH/auto-psych/holdout_test_retest_v2 \
#     $SCRATCH/auto-psych/holdout_test_retest_no_inner_loop_v2 \
#     $SCRATCH/auto-psych/impossible_holdout_test_retest_v2 \
#     $SCRATCH/auto-psych/impossible_holdout_test_retest_no_inner_loop
#
#   # dry run — just build the manifest and print what WOULD be re-scored:
#   DRY_RUN=1 bash scripts/.../submit_reanalyze_exhaustive.sh <roots...>
#
# Sharing one venv across several independent (e.g. per-study, dependency-gated)
# sweeps — build it ONCE, then point each sweep at it so they don't race on the
# build:
#   RB=$SCRATCH/auto-psych/reanalyze_exhaustive
#   # 1. build the shared venv once:
#   WORK_ROOT=$RB UV_PROJECT_ENVIRONMENT=$RB/venv \
#     bash scripts/.../submit_reanalyze_exhaustive.sh --setup-only
#   # 2. each sweep reuses it (own WORK_ROOT for an isolated manifest/logs):
#   WORK_ROOT=$RB/holdout UV_PROJECT_ENVIRONMENT=$RB/venv \
#     bash scripts/.../submit_reanalyze_exhaustive.sh --skip-setup <root>
#
# Flags: --setup-only (build the venv and exit), --skip-setup (reuse an existing
# venv, no setup job).
# Env knobs: WORK_ROOT, UV_PROJECT_ENVIRONMENT, MAX_PARALLEL, REANALYZE_LENGTHS,
# REANALYZE_PREDICT_MAX_DRAWS, REANALYZE_MIN_REMAINING (forwarded via --export=ALL).
set -euo pipefail

HOLDOUT_SLURM_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HOLDOUT_SLURM_DIR

SCRATCH_BASE="${SCRATCH:-$GROUP_SCRATCH}/auto-psych"

# Parse flags out of the positional args; everything else is a study root.
#   --setup-only  build the reanalysis venv (only) and exit. Run this ONCE up
#                 front when you intend to launch several --skip-setup sweeps.
#   --skip-setup  do NOT submit a setup job; reuse an already-built venv (fail
#                 loudly if it is missing). Lets independent per-study launchers
#                 share one venv (via UV_PROJECT_ENVIRONMENT) instead of racing
#                 to rebuild it into the same path.
SKIP_SETUP="" ; SETUP_ONLY="" ; ROOTS=()
for arg in "$@"; do
  case "$arg" in
    --skip-setup) SKIP_SETUP=1 ;;
    --setup-only) SETUP_ONLY=1 ;;
    --*)          echo "ERROR: unknown flag: $arg" >&2; exit 1 ;;
    *)            ROOTS+=("$arg") ;;
  esac
done

# Reanalysis WORK_ROOT: holds the manifest + slurm logs (and, by default, the
# venv at $WORK_ROOT/venv). Kept OFF the study roots so it never collides with a
# re-run's own WORK_ROOT. Set UV_PROJECT_ENVIRONMENT to point several per-study
# WORK_ROOTs at ONE shared venv.
export WORK_ROOT="${WORK_ROOT:-$SCRATCH_BASE/reanalyze_exhaustive}"
[[ -n "${UV_PROJECT_ENVIRONMENT:-}" ]] && export UV_PROJECT_ENVIRONMENT
VENV_PY="${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}/bin/python"
mkdir -p "$WORK_ROOT"
LOGDIR="$WORK_ROOT/slurm_logs"
mkdir -p "$LOGDIR"

cd "$HOLDOUT_SLURM_DIR"

# --setup-only: build the venv (no manifest, no roots needed) and exit.
if [[ -n "$SETUP_ONLY" ]]; then
  setup_id=$(sbatch --parsable --export=ALL \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    reanalyze_setup.sbatch)
  echo "submitted setup-only job: $setup_id  (venv -> ${VENV_PY%/bin/python})"
  echo "once it completes, run --skip-setup sweeps with the SAME"
  echo "  UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-$WORK_ROOT/venv}"
  exit 0
fi

# Study roots to sweep. Pass dirs and/or explicit holdout.json paths as args; the
# four study families have no single canonical WORK_ROOT (the run wrappers use
# suffixes like _v2 / _full), so there is no safe default — require them.
if [[ "${#ROOTS[@]}" -eq 0 ]]; then
  echo "ERROR: pass at least one study root dir (or holdout.json) to re-score." >&2
  echo "       e.g. $SCRATCH_BASE/holdout_test_retest_v2 ..." >&2
  echo "       existing study roots under $SCRATCH_BASE:" >&2
  ls -d "$SCRATCH_BASE"/*holdout* 2>/dev/null | sed 's/^/         /' >&2 || true
  exit 1
fi

# Build the manifest: one holdout.json path per line. Each find is SCOPED to a
# given root (never a filesystem-wide scan) — depth 3 reaches run<R>/<gt>/holdout.json.
MANIFEST="$WORK_ROOT/manifest.txt"
: > "$MANIFEST"
for root in "${ROOTS[@]}"; do
  if [[ -f "$root" && "$root" == *holdout.json ]]; then
    echo "$root" >> "$MANIFEST"
    echo "  [file] $root"
    continue
  fi
  if [[ ! -d "$root" ]]; then
    echo "  [skip] root not found: $root" >&2
    continue
  fi
  before=$(wc -l < "$MANIFEST")
  find "$root" -mindepth 1 -maxdepth 4 -name holdout.json -type f 2>/dev/null \
    | sort >> "$MANIFEST"
  after=$(wc -l < "$MANIFEST")
  echo "  [root] $root -> $((after - before)) holdout.json"
done

TOTAL=$(wc -l < "$MANIFEST")
[[ "$TOTAL" -gt 0 ]] || { echo "ERROR: no holdout.json found under the given roots." >&2; exit 1; }
export REANALYZE_MANIFEST="$MANIFEST"
echo
echo "manifest: $MANIFEST  ($TOTAL runs to re-score)"

if [[ -n "${DRY_RUN:-}" ]]; then
  echo ">>> DRY_RUN: not submitting. Manifest contents:"
  cat "$MANIFEST"
  exit 0
fi

MAX_PARALLEL="${MAX_PARALLEL:-10}"
# Pass-through reanalysis knobs only if the caller set them (defaults live in the
# array sbatch / CLI).
[[ -n "${REANALYZE_LENGTHS:-}"           ]] && export REANALYZE_LENGTHS
[[ -n "${REANALYZE_PREDICT_MAX_DRAWS:-}" ]] && export REANALYZE_PREDICT_MAX_DRAWS
[[ -n "${REANALYZE_MIN_REMAINING:-}"     ]] && export REANALYZE_MIN_REMAINING
[[ -n "${REPO:-}"                        ]] && export REPO

if [[ -n "$SKIP_SETUP" ]]; then
  # Reuse a venv built by an earlier --setup-only run; fail loudly if missing.
  if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: --skip-setup but no reanalysis venv at $VENV_PY." >&2
    echo "       build it once first, with the SAME WORK_ROOT/UV_PROJECT_ENVIRONMENT:" >&2
    echo "         bash $(basename "$0") --setup-only" >&2
    exit 1
  fi
  array_id=$(sbatch --parsable --export=ALL \
    --array="1-${TOTAL}%${MAX_PARALLEL}" \
    --output="$LOGDIR/%x_%A_%a.out" --error="$LOGDIR/%x_%A_%a.out" \
    reanalyze_exhaustive_array.sbatch)
  echo "submitted array job: $array_id (1-$TOTAL%$MAX_PARALLEL; reusing venv $VENV_PY)"
else
  setup_id=$(sbatch --parsable --export=ALL \
    --output="$LOGDIR/%x_%j.out" --error="$LOGDIR/%x_%j.out" \
    reanalyze_setup.sbatch)
  echo "submitted setup job: $setup_id"

  array_id=$(sbatch --parsable --dependency=afterok:"$setup_id" --export=ALL \
    --array="1-${TOTAL}%${MAX_PARALLEL}" \
    --output="$LOGDIR/%x_%A_%a.out" --error="$LOGDIR/%x_%A_%a.out" \
    reanalyze_exhaustive_array.sbatch)
  echo "submitted array job: $array_id (1-$TOTAL%$MAX_PARALLEL)"
fi

echo
echo "watch with:  squeue --me"
echo "logs in:     $LOGDIR"
echo "re-scored runs are listed in: $MANIFEST"
echo "when the array finishes, re-run your collectors to roll up data/results."
