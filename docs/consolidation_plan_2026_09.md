# Consolidation plan, September 2026 — executable version

*Written 2026-09-16 by Claude (Fable 5.1) after reviewing `merge_plan_2026_09.md`
(Claude) and `merge_plan_2026_09_amended.md` (Codex) and verifying their claims
against the code and the archived runs. This is the plan an implementing agent
follows, phase by phase. A driver (`scripts/consolidation/`) runs one agent
session per phase in a Slurm job; the agent reads this file in full every
session and executes exactly the phase it is told.*

---

## 0. What this consolidates and why

Five lines of work exist, none merged (see `merge_plan_2026_09.md` §1). The
decision, after review:

1. **Base on campaign iteration 3** (`ba8b2de`). It carries iterations 1–2's
   correctness fixes (held-out label leak, export by ELPD rank, PSIS-LOO
   exemption) plus the live-set carry-forward and the attempted-hypotheses
   ledger.
2. **Merge current `main`** (infrastructure: the `MissingStimulusColumns`
   fail-loud screen `884728e`, campaign/panel tooling, docs).
3. **Restore the user-authored leakage audit** that iteration 3 reverted, by
   reverting the revert.
4. **Merge arm C as opt-in machinery.** Arm C's own run was **not a raw-features
   run** (§0.1), so its recovery numbers decide nothing; its machinery (raw
   seed sets, collision rule, pool selection, manifest-entry scrub, verifier)
   is wanted.
5. **Make raw mode genuinely raw end to end**, then verify it with a smoke run
   (P2, P7, P8). After the first smoke round the user decided (2026-09-16,
   see the amendment before P9) that **raw is the only mode** and that the
   feature code must be **unreadable** by agents: P9 removes the featurized
   pipeline, P10 isolates the agents' tree and gates candidate imports, P11–
   P12 smoke it again, then the job launches a **5-repeat recovery sweep**
   (5 repeats × 4 ground truths = 20 cells), submits the RMSE evaluation and
   writes `RESULTS.md` (P13–P15).
6. **Honest metrics**: probability RMSE primary, expected Bernoulli KL regret
   secondary, Pearson r descriptive; per-repeat paired differences, no
   stimulus bootstrap. Add an offline all-admitted-models oracle diagnostic
   with a *retention* bucket (§0.2).
7. **Reimplement**, not cherry-pick, the two defensible halves of iterations 4
   and 5: the `elpd_diff ± dse` race presentation (without the per-stimulus
   residual table) and the seven-lens rotation (without repair-on-rejection).

### 0.1 Verified facts the plan rests on

- **Arm C re-featurized behind its own back.** On `6ed41ea`,
  `run_inner_model_loop_programmatic` in `src/pipelines/outer_loop/orchestrator.py`
  takes no `raw_features` parameter and always calls `_load_project_featurizer`
  then `_write_feature_csv`. In the archived run
  (`$ARMC_RUN_ROOT/run1/falk_konold_dp/agent_runs.tar.gz`),
  `experiment1/data/responses.csv` has 5 columns and
  `experiment1/model_loop/responses.csv` has 59; every candidate `CONTEXT.md`
  lists the engineered columns. Its verifier only checked `data/responses.csv`.
  The 20 `[drop]` lines are candidates written against supplied columns that
  could not bind raw design rows.
- **"Best model the loop ever held" was really "best recorded incumbent".**
  `evaluate_trajectory` persists only the per-step `best_model` and the BMA.
- **Iteration 3 changed the design prior.** Commit `9764f50` replaced the
  stacking-weight registry with a **uniform prior over the carried set** and
  removed the prune weight floor (prune on `elpd_diff > 2·dse` alone).
  `CLAUDE.md` on `main` still documents stacking weights. Both plans inherited
  this silently; it is kept, and documented (P6).
- **The held-out model's manifest entry was readable in every sweep before arm
  C.** `main`'s `holdout_recovery_array.sbatch` deletes the held-out `.py` from
  the registry and the pool and stubs the family twin, but leaves the
  `models_manifest.yaml` entry (name + rationale). Arm C's array script removes
  that entry (`remove_manifest_entry.py`). Merging arm C therefore closes this
  channel for every future sweep, featurized or raw. This is a deliberate
  confound against iteration 3's numbers; record it, do not undo it.
- **Lens bug.** On iteration 3 the lens is `hints[candidate_idx % len(hints)]`
  with three candidates, so lenses 3–6 never fire (72/72/72/0/0/0/0 in the
  iteration 4 sweep).
- **Metrics.** Brier regret against a known probability is `(p−q)²`, i.e. MSE,
  so it is not a second metric. Expected Bernoulli KL regret
  `q·log(q/p) + (1−q)·log((1−q)/(1−p))` is the calibration-sensitive one.
  Matched seeds do **not** give identical data past experiment 1 (adaptive
  design); call them *matched-seed cells*.
- **A retention failure exists that is neither discovery nor selection.** In
  iteration 4, `run2/falk_konold_dp` found the held-out DP model (r 0.9999) in
  experiment 1 and lost it at experiment 2 when the fit gate rejected an
  unclipped sigmoid on a new design pair (campaign journal, 2026-09-11).

### 0.2 Three failure buckets the diagnostics must separate

- **discovery**: no strong model was ever admitted;
- **selection**: a strong admitted model existed but was never the incumbent;
- **retention**: a model that *was* the incumbent was later dropped or pruned.

---

## 1. Frozen inputs

The driver records these in `$WORK_ROOT/consolidation.env` at submit time and
verifies every SHA when it prepares the clone. If any differs, the job stops
before the agent starts.

| Input | Ref in the clone | Expected SHA |
|---|---|---|
| campaign iteration 3 (**base**) | `refs/consolidation/iter3` | `ba8b2de66c45305b66066545ed8cf30c5cad208f` |
| campaign iteration 4 (reference only) | `refs/consolidation/iter4` | `470e187ed2beded050c267830380d14f1e02d27c` |
| campaign iteration 5 (reference only) | `refs/consolidation/iter5` | `2d450e21f20c5748b870e479f859766fd47b5b69` |
| arm C | `refs/consolidation/armc` | `6ed41ea6d8994473eab0b1b79959d9ec706eb340` |
| `main` at submit time | `refs/consolidation/main` | `$MAIN_SHA` (from `consolidation.env`) |
| leakage-audit patch (fallback only) | `$LEAKAGE_PATCH` | sha256 `cf3be3eed6373dda6192e3729971c47fd2884afc44651e511cc686b7b0b2764c` |
| arm C archived run (read-only evidence) | `$ARMC_RUN_ROOT` | — |
| iteration 3 sweep (3 repeats, seeds 101–103; comparison) | `$ITER3_SWEEP` | — |
| iteration 2 sweep (5 repeats, seeds 101–105; comparison) | `$ITER2_SWEEP` | — |
| pre-campaign baseline sweep (5 repeats, seeds 101–105; comparison) | `$BASELINE_SWEEP` | — |
| sweep knobs for P13 | `$SWEEP_N_REPEATS`, `$SWEEP_BASE_SEED`, `$SWEEP_MAX_PARALLEL` (`$SWEEP_ARMS` is obsolete: one pipeline) | from `consolidation.env` |

Ancestry (verified): iterations 4 and 5 descend from iteration 3; arm C
descends from iteration 2 (`bd7f032`), so it does **not** contain iteration 3.

---

## 2. Where the agent works and what it may do

- **Clone:** `$REPO`, branch `consolidate/2026-09`, created by the driver at
  `ba8b2de` with the refs above already fetched. Work and commit here. Never
  push. Never touch `$SOURCE_REPO` (the user's checkout).
- **Progress dir:** `$WORK_ROOT/progress/` — the phase markers (§3) and
  baseline files live here, outside the repo.
- **Python:** `$VENV_PY` (pymc/pytensor/arviz stack, pandas, pytest, h5netcdf).
  You are on a Slurm compute node: running Python and the fast test suite here
  is fine. Never write under `$HOME`.
- **Fast suite:**
  `$VENV_PY -m pytest -q -m "not slow" -p no:cacheprovider --continue-on-collection-errors -rf`
  Some modules cannot be collected in this venv (flask, pydantic, plotnine,
  langchain, browser libs) and two tie-break numerics fail on this stack; that
  is why P0 records the exact **failing node IDs** as the baseline. Compare by
  node ID, never by count.
- **Static checks:** `git diff --check`; `$VENV_PY -m compileall -q src scripts`;
  `$VENV_PY -m pytest -q tests/test_python_sources_compile.py`.
- **Slurm:** you may `sbatch` only in P7 and P11 (smoke cells), P13 (the
  recovery sweep) and P14 (the evaluation job) — nothing else. Never
  `scancel`, never `scontrol`, never poll or sleep-wait on a job. The driver
  handles waiting by requeueing itself after the jobs a phase submitted.
- **TDD, as the repo requires:** every behaviour change starts from a failing
  test. Run the relevant tests after each green step; run the fast suite before
  each phase's commit.
- **Fail loudly.** No silent defaults, no fallbacks, no `except: pass`.
- **Reference reads:** `git show refs/consolidation/iter4:<path>` and
  `git show refs/consolidation/iter5:<path>` in the clone show iteration 4/5
  code without checking it out. `git show refs/consolidation/iter4` shows the
  commit's diff.
- **Do not** bring over `incumbent_fit.py`, `AdmissionOutcome`, candidate
  repair, carried-model repair, or the `candidate_repairs` knob.
- **Do not** modify the ground-truth registry models, the family twins, or the
  held-out parameters. P3 touches the evaluation reporting path by the user's
  explicit instruction; it must leave every existing `pearson_r`/`rmse` value
  unchanged (regression-tested).

---

## 3. Phase contract (enforced by the driver)

Phases run in order: **P0 … P15**. Each session is told its phase. When the
phase's acceptance checks pass:

1. commit everything on `consolidate/2026-09` (tree clean:
   `git status --porcelain` prints nothing);
2. write `$WORK_ROOT/progress/P<k>.done` with, on the first line,
   `commit: <full sha of HEAD>` and below it a short summary: what changed,
   the tests run and their results, and any deviation from this plan.

If a **stop condition** (§7) is hit, or the phase cannot be completed as
specified, write `$WORK_ROOT/progress/P<k>.blocked` with the reason and what
you found, commit any clean partial work, and stop. Do not improvise around a
stop condition.

The driver validates the marker (HEAD matches, tree clean, phase-specific
files exist). An invalid marker gets one repair session with the problems
injected; a second failure blocks the job.

**Re-opening an earlier phase.** A verdict phase may send an earlier phase
back for another round by writing `$WORK_ROOT/progress/P<j>.retry<k>` (one
line on why) and deleting `P<j>.done`. In that case do **not** write your
own `.done` marker: end the session. The driver sees the new retry marker,
re-runs `P<j>`, and then runs your phase again from scratch.

---

## 4. Phases

### P0 — Baseline on the untouched base

**Goal:** the exact fast-suite failure set at `ba8b2de`, so every later phase
is compared by node ID.

**Do:**
1. `git rev-parse HEAD` must be `ba8b2de66c45305b66066545ed8cf30c5cad208f`;
   `git status --porcelain` empty. If not, write `P0.blocked`.
2. Run the fast suite. From the `-rf` short summary, write the sorted failing
   node IDs, one per line, to `$WORK_ROOT/progress/baseline_failing_tests.txt`,
   and the collection errors (module paths) to
   `$WORK_ROOT/progress/baseline_collection_errors.txt`.
3. Record `$VENV_PY --version`, `pymc.__version__`, `arviz.__version__` in the
   done file.

**Accept:** both files exist; no code changes; `P0.done` has `commit: ba8b2de…`.

### P1 — Integrate: merge `main`, restore the audit, merge arm C

Three commits, tests after each. Resolve conflicts by **keeping both sides**;
never by picking one.

**Do:**
1. `git merge --no-ff refs/consolidation/main`. Codex's preflight found this
   clean against `6ec013e`; the frozen `main` is later, so re-check. Run the
   fast suite; new failing node IDs relative to the baseline must be fixed or
   explained in the done file. Targeted: `tests/test_eig_selection.py`,
   `tests/test_recovery_improvement_units.py`, anything named `screen`.
2. `git revert --no-edit ba8b2de` (it is a single-purpose revert of the audit:
   107 lines in `holdout_recovery.py`, 138 in its test file). Commit message
   must say the restored audit is user-authored and protected. If the revert
   does not apply cleanly, `git apply --3way $LEAKAGE_PATCH` (verify the sha256
   first) and commit with the same message. Run
   `tests/test_subjective_randomness_holdout_recovery.py -k leakage`: the
   restored tests (`any_csv_generating_model`, `any_manifest_gt_named`,
   per-model `pm.Data` columns, loop-output manifests ignored) must pass.
3. `git merge --no-ff refs/consolidation/armc`. Expected textual conflict:
   `CLAUDE.md` only (keep both paragraphs). **Semantic** conflicts are the
   risk: arm C (branched from iteration 2) and iteration 3 both edited
   `src/pipelines/inner_loop/pymc_orchestrator.py` (`_prune_losers`, export,
   history) and `src/subjective_randomness/holdout_recovery.py`. Rules for
   reconciling: iteration 3's behaviour is the base (prune on
   `elpd_diff > 2·dse` among reliable rows, no weight floor; uniform design
   prior; ledger; carried models unprotected); arm C's additions are kept
   (`loo_reliability.py`, the same-value collision rule
   `_same_feature_value`/`PROTECTED_ROW_COLUMNS` in `pymc_inference.py`,
   `pool_models_dir`, raw seed sets, `remove_manifest_entry.py` and the array
   script's manifest scrub, the verifier). Run: `tests/test_prune_losers.py`,
   `tests/test_carry_live_set.py`, `tests/test_hypothesis_ledger.py`,
   `tests/test_pymc_inner_loop_ledger.py`, `tests/test_loo_reliability.py`,
   `tests/test_model_compare_se.py`, `tests/test_raw_features_seeds.py`,
   `tests/test_model_manifest.py`, `tests/test_remove_manifest_entry_cli.py`,
   `tests/test_pymc_inference_extensible_features.py`,
   `tests/test_subjective_randomness_holdout_recovery.py`, then the fast suite.
4. Do **not** edit `holdout_recovery_faithful.yaml` or any launcher default.
   Raw mode stays opt-in.

**Accept:** three merge/revert commits; fast-suite failing set ⊆ baseline (any
new failure fixed, or explained and accepted only if it is an environment
collection error). If reconciling the arm C merge takes more than ~2 hours of
session time, write `P1.blocked` with the exact failing tests.

### P2 — Make raw mode genuinely raw end to end

**Goal:** in a `raw_features` run, no agent-facing artefact carries an
engineered column, and any model that cannot bind raw rows fails by name.

**Tests first** (all new, red before green):
- `tests/test_raw_inner_loop.py`
  - `run_inner_model_loop_programmatic(exp_dir, raw_features=True, ...)` with
    `run_pymc_inner_loop` monkeypatched to a recorder: given a raw 5-column
    `data/responses.csv`, the written `model_loop/responses.csv` header is
    exactly `sequence_a,sequence_b,participant_id,trial_index,chose_left`.
  - with `raw_features=False` the featurized columns are present (existing
    behaviour, now pinned).
  - `raw_features=True` with a `data/responses.csv` that already carries
    engineered columns raises (a raw run must not inherit featurized data).
  - the candidate `CONTEXT.md` written from a raw CSV names only those five
    columns, says only `chose_left` is numeric, and says a `compute_features`
    or `prepare_observed` hook is **required**; it contains none of
    `rep_motifs`, `occ_n20`, `multiscale_imbalance`, `precomputed`.
- `tests/test_raw_config_validation.py`: a raw config whose `pool_models_dir`
  or `seed_models_dir` holds a model that cannot bind a raw row raises at
  config resolution, naming the model and the missing columns; the raw seed
  dirs pass.
- `tests/test_verify_raw_features_run.py` (runs the bash verifier via
  `subprocess`): a synthetic run tree reproducing arm C's bug — raw
  `experiment1/data/responses.csv`, 59-column
  `experiment1/model_loop/responses.csv`, a `CONTEXT.md` listing engineered
  columns — makes the verifier exit non-zero and name the offending file; a
  clean raw tree passes; a tree with **no** candidate-facing CSV fails.

**Do:**
1. Define `RAW_RESPONSE_COLUMNS` once in the pipeline
   (`src/pipelines/outer_loop/featurizer.py` is the natural home) and make
   `holdout_recovery.py` import it from there (it already imports from the
   orchestrator; one more import from the pipeline is in keeping with the
   existing thin coupling).
2. `run_inner_model_loop_programmatic(..., raw_features: bool = False)`. When
   true: do not load the featurizer; write `model_loop/responses.csv` with
   `_write_feature_csv(rows, None, ...)`; then read the header back and raise
   if it is not exactly `RAW_RESPONSE_COLUMNS`. Also raise if the pooled rows
   carry any column outside that set.
3. Thread `raw_features` from `holdout_recovery.py`
   (`_run_holdout_recovery_resolved` → the inner-loop call) and from every
   other caller of `run_inner_model_loop_programmatic` that has a raw-features
   notion (grep; others keep the default).
4. Candidate context (`_write_candidate_context` in `pymc_orchestrator.py`):
   derive the feature columns as `header − RAW_RESPONSE_COLUMNS`. If empty,
   the prose must state there are no precomputed feature columns and that the
   model must compute its own via `compute_features`/`prepare_observed`. Do
   the same for the critique context (grep for where `CRITIQUE_CONTEXT.md` is
   written) and for any prompt text in `src/pipelines/inner_loop/prompts/`
   that assumes precomputed columns.
5. Config validation in `run_holdout_recovery_from_config`: when
   `raw_features` is true, `pool_models_dir` must be set, and every model in
   both manifests must bind a raw stimulus row. Use the same binding path the
   design screen uses (`make_stim_data` / `MissingStimulusColumns` in
   `src/models/pymc_inference.py`) on a row like
   `{"sequence_a": "HHT", "sequence_b": "THT", "chose_left": 0, "participant_id": 0, "trial_index": 0}`.
   Raise `ValueError` naming the model and the missing columns. Capability,
   not directory-name suffix, is the check.
6. Export/carry portability: where the inner loop exports models into
   `cognitive_models/` (`_export_inner_loop_models` on iteration 3) and where
   the outer loop carries them forward, determine raw mode from the
   `model_loop/responses.csv` header (single source of truth) and, in raw
   mode, verify each exported/carried model binds a raw row; raise naming the
   model. (`884728e` already raises at design time; this fails earlier, with
   the model's name, before a design is even attempted.)
7. Strengthen `scripts/subjective_randomness/slurm/verify_raw_features_run.sh`:
   check every `experiment*/data/responses.csv` **and**
   `experiment*/model_loop/responses.csv` (raw trees under
   `run<r>/<gt>/repo/_runs/<gt>/` and archived `agent_runs.tar.gz`, via
   `tar tzf`/`tar xzf -O`); check candidate `CONTEXT.md` column lists; fail
   if no candidate-facing CSV was found; require zero `[drop]` lines and every
   `design/screened_out.json` equal to `[]`; fail if any candidate source
   imports the project featurizer (`src.subjective_randomness.features`,
   `preprocess`); confirm all configured cells completed; print the resolved
   config, code SHA and feature mode into `VERDICT.md`.

**Accept:** the new tests pass; `tests/test_raw_features_seeds.py`,
`tests/test_carry_live_set.py`, `tests/test_inner_loop_carryforward_screen.py`
(if present), `tests/test_subjective_randomness_holdout_recovery.py` pass; fast
suite failing set ⊆ baseline.

### P3 — Honest metrics and offline diagnostics

**Goal:** RMSE primary, KL regret secondary, Pearson descriptive, per-cell
paired comparison on a common pool, and the three-bucket oracle diagnostic.
Existing `pearson_r`/`rmse` values must not change.

**Tests first:**
- `tests/test_recovery_metrics.py` for a new pure module
  `src/subjective_randomness/recovery_metrics.py`:
  `kl_regret(q, p, eps=1e-9)`, `bias(q, p)`, `calibration(q, p) -> (slope,
  intercept)` (OLS of p on q), `rmse(q, p)`. Analytic cases: `p == q` gives
  0/0/(1,0); `p = 1 − q` gives slope −1; values at exactly 0 and 1 are clipped
  and finite; `q` and `p` of different length raise.
- A regression test that `evaluate_trajectory`'s `pearson_r` and `rmse` on a
  small fixture equal the values the current code produces (capture them in
  the red step).
- Reader tests: `trajectory_tidy_rows`, `holdout_test_retest.py`'s row reader
  and `src/recovery_improvement/digest.py` accept a legacy `holdout.csv`
  without the new columns and report them as missing (`n/a`), never fabricate.
- `compare_matched_cells` test on two tiny synthetic run roots: the common
  pool excludes the **union** of both cells' trained pairs; output has one row
  per cell pair with both cells' metrics and the delta.
- `oracle_admitted_models` test on a synthetic run root with three admitted
  models per step and a ledger: reports oracle-best, incumbent, final, gap
  per step, and lists a "lost incumbent" (a model that was `best_model` at an
  earlier step and appears with outcome `dropped` or `pruned` in
  `attempted_hypotheses.jsonl`).

**Do:**
1. `evaluate_trajectory` rows gain `kl_regret`, `bias`, `calib_slope`,
   `calib_intercept` and the `_bma` variants; `holdout.json` gains
   `metrics_version: 2`; `TRAJECTORY_COLUMNS` updated; keep every existing
   column and value.
2. `scripts/subjective_randomness/holdout_test_retest.py` and the analysis
   sbatch: emit `rmse` and `kl_regret` summaries **in addition to** the
   Pearson ones (keyed per metric in `test_retest.json`; extra columns in
   `test_retest.csv`); the digest's comparison table shows RMSE first, Pearson
   after. Do not silently change the meaning of an existing key.
3. New CLI `scripts/subjective_randomness/compare_matched_cells.py` (tyro):
   `--sweep-a <root> --sweep-b <root> --out <dir>` (and `--cell-a/--cell-b`
   for one pair). For every `run<r>/<gt>` present in both: build the
   exhaustive pool over lengths 1..8 minus the union of both cells' trained
   pairs (`collect_trained_pairs`), re-score both cells via
   `reevaluate_trajectories` (extend it with an `extra_excluded_pairs`
   argument; do **not** overwrite `holdout.json`), and write `paired.csv` /
   `paired.json`: per cell pair the final-step RMSE, KL regret, Pearson r for
   A and B and the deltas; then per ground truth the mean, median, min and max
   delta across repeats. No bootstrap over stimuli. Cells whose archives cannot
   be reconstructed are listed, not skipped silently.
4. New CLI `scripts/subjective_randomness/oracle_admitted_models.py` (tyro):
   `--result <holdout.json> [--steps final|all]`. For each step (default:
   final step of each experiment), score every model under
   `experiment<k>/model_loop/models/*.py` and `models/pruned/*.py` on the
   held-out pool via the cell's `mcmc_cache` (cache hits; refits are a loud
   error only if the cache is missing), and write `oracle.json`/`oracle.csv`
   next to `holdout.json` with, per step: oracle-best model and RMSE, the
   incumbent's RMSE, the final selected model's RMSE, the oracle−incumbent gap,
   and the retention list from `attempted_hypotheses.jsonl` (models with a
   `dropped`/`pruned` outcome that were `best_model` at an earlier step).
   Trees archived as `agent_runs.tar.gz` are extracted into a temp dir under
   the cell.
5. These CLIs are user-side evaluation tools. Do not reference them from any
   agent-facing prompt or context.
6. New CLI `scripts/subjective_randomness/recovery_report.py` (tyro):
   `--sweep <root> --label <name> [--compare <label>=<paired dir>]...
   [--oracle-glob <glob>] --out <markdown path>`. Reads every cell's
   `holdout.csv` final step (`rmse`, `kl_regret`, `pearson_r`, `best_model`;
   legacy files without the new columns show `n/a`, never a fabricated
   value) and writes, as markdown plus a CSV beside it: per ground truth the
   final-step RMSE mean ± sd, median, min, max and n over repeats, the same
   for KL regret, Pearson r as descriptive, and the best model per cell. For
   each `--compare` dir it reads that dir's `paired.csv` and adds per-cell
   rows and per-ground-truth mean/median/min/max deltas. For the oracle
   files it adds, per ground truth, the number of cells whose oracle−incumbent
   RMSE gap exceeds 0.02 and the number of lost-incumbent events. A missing
   compare dir or sweep root is an error. Test on a synthetic sweep (2 ground
   truths × 3 repeats) with exact expected means, a legacy-CSV cell, and a
   missing compare dir.

**Accept:** new tests pass; the regression test proves old values unchanged;
`tests/test_subjective_randomness_holdout_recovery.py` and the reporting tests
pass; fast suite failing set ⊆ baseline.

### P4 — Reimplement the race presentation (iteration 4's defensible half)

**Goal:** candidates see a real race, not a rounded softmax.

**Tests first** — `tests/test_existing_hypotheses_race.py`: given a posterior
dict with an `az.compare` table (`rank`, `elpd_diff`, `dse`, reliability flag —
look at what iteration 3 already persists in `model_posterior.json` /
`history.json` and at `_select_export_model`), the written
`existing_hypotheses.md` lists models best-first with rank, `elpd_diff ± dse`,
a verdict ("not clearly separated at ≈2·dse" vs "clearly behind"), and the
PSIS-LOO reliability; a model with no comparison row is shown as "no
comparison row" (no crash); the text contains no softmax posterior, no
per-stimulus residuals, no parameter values, no source code.

**Do:** rewrite `_write_existing_hypotheses` in `pymc_orchestrator.py`
accordingly; update the sentence in `src/pipelines/inner_loop/prompts/pymc_theory.md`
that describes the file. Consult
`git show refs/consolidation/iter4:src/pipelines/inner_loop/pymc_orchestrator.py`
lines around `_write_existing_hypotheses` for the wording, but write your own
code and tests. Do **not** add `incumbent_fit.py` or any residual table.

**Accept:** new tests pass; `tests/test_parallel_candidates.py`,
`tests/test_pymc_inner_loop_history.py`, `tests/test_pymc_inner_loop_critique.py`,
`tests/test_pymc_inner_loop_ledger.py`, `tests/test_candidate_context_features.py`
pass; fast suite failing set ⊆ baseline.

### P5 — Reimplement lens rotation (iteration 5's defensible half)

**Goal:** all seven exploration lenses fire over a standard run.

**Tests first** — `tests/test_lens_rotation.py` (write fresh; iteration 5's
file may be consulted but this one must not depend on any `incumbent_fit`
fixture): (a) pure schedule: `_lens_index(offset, iteration, candidate_count,
candidate_idx, n_lenses) == (offset + iteration·candidate_count +
candidate_idx) % n_lenses`, and over 3 experiments × 2 rounds × 3 candidates
all seven default lenses fire with counts differing by at most 1; (b) the lens
text in a candidate's `CANDIDATE_BRIEF.md` matches the lens index the ledger
records for that slot; (c) `run_inner_model_loop_programmatic` passes
`lens_offset = _lens_offset(exp_num, ...)` derived from `exp_dir.name`
(`experiment2` with 2 rounds × 3 candidates starts at offset 6); (d) an empty
lens battery raises.

**Do:** add `_lens_offset` and `_lens_index` as pure functions in
`pymc_orchestrator.py`; `run_pymc_inner_loop(..., lens_offset: int = 0)`;
replace the `candidate_idx % len(hints)` site(s); record the lens index in the
ledger entry for the slot (the ledger already has a lens field). Thread
`lens_offset` from the outer loop.

**Accept:** new tests pass; the P4 test list passes; fast suite failing set ⊆
baseline. `tests/test_candidate_repair.py` and
`tests/test_carried_model_repair.py` must **not** exist.

### P6 — Documentation, decision record, full checks

**Do:**
1. `CLAUDE.md`: the registry paragraph must describe iteration 3's uniform
   design prior over the carried set and the dse-only prune rule; add the
   raw-mode paragraph (two seed sets, `raw_features`, the verifier) if the arm
   C merge did not already; keep the screening paragraph from `main`.
2. `docs/consolidation_decision_record.md`: inputs and SHAs, the decisions in
   §0, the arm C finding with the 5-vs-59-column evidence, the metric protocol
   (§8), the gate (§9), the manifest-scrub confound, and what remains open
   (residual-table causal test, repair behaviour, process-level import
   isolation).
3. Run: `git diff --check`; `$VENV_PY -m compileall -q src scripts`;
   `tests/test_python_sources_compile.py`; the full fast suite. Write the
   failing node IDs to `$WORK_ROOT/progress/P6_failing_tests.txt` and the
   diff against the baseline (added / removed) into the done file.

**Accept:** no new failing node IDs beyond the baseline; docs committed.

### P7 — Submit the two smoke chains (the only phase that may `sbatch`)

**Goal:** one featurized and one true-raw smoke cell from the consolidated
commit. Cheap: SMOKE settings (one task, tiny MCMC), but **two experiments and
one candidate round**, so carry-forward and raw design binding of carried
candidates are exercised (that is exactly where arm C failed).

**Do**, from `$REPO` (its `.secrets` is a copy of the user's, put there by the
driver; `REPO` must point at the clone so the array copies the consolidated
tree):

```bash
cd "$REPO"
export REPO="$REPO"
SMOKE=1 N_EXPERIMENTS=2 INNER_LOOP_ITERATIONS=1 BASE_SEED=100 \
  WORK_ROOT="$WORK_ROOT/smoke_featurized" \
  bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
SMOKE=1 N_EXPERIMENTS=2 INNER_LOOP_ITERATIONS=1 BASE_SEED=100 \
  WORK_ROOT="$WORK_ROOT/smoke_raw" \
  bash scripts/subjective_randomness/slurm/run_raw_features_arm.sh
# verifier, gated on the raw chain's analysis job:
sbatch --dependency=afterany:<raw analysis id> --job-name=verify_raw_smoke \
  --partition=normal --time=00:20:00 --cpus-per-task=1 --mem=4GB \
  --output="$WORK_ROOT/smoke_raw/slurm_logs/verify_%j.out" \
  --export=ALL,WORK_ROOT="$WORK_ROOT/smoke_raw",ALL_JOB_IDS="<setup>,<array>,<analysis>" \
  scripts/subjective_randomness/slurm/verify_raw_features_run.sh
```

Read the submit scripts before running them and adjust only if a knob name
has changed. If a submission fails, fix the cause in the repo (commit) and
resubmit; if it fails twice, write `P7.blocked`. On a **retry round** (a
`progress/P7.retry*` marker exists, written by P8), use fresh work roots
`$WORK_ROOT/smoke_featurized_round<k>` and `$WORK_ROOT/smoke_raw_round<k>`
and overwrite `smoke_jobs.json` with the new ids and roots.

Write `$WORK_ROOT/progress/smoke_jobs.json`:

```json
{"featurized": {"work_root": "...", "job_ids": ["<setup>", "<array>", "<analysis>"]},
 "raw":        {"work_root": "...", "job_ids": ["<setup>", "<array>", "<analysis>", "<verify>"]}}
```

**Accept:** `smoke_jobs.json` exists with numeric job ids; tree clean;
`P7.done`. Do not wait for the jobs: end the session. The driver requeues
itself `afterany` those ids and starts P8 when they have finished.

### P8 — Smoke verdict and handoff (no sweeps are launched)

**Preconditions (driver-checked):** every id in `smoke_jobs.json` has left the
queue.

**Do:**
1. Read both chains' `slurm_logs/`, `run1/<gt>/holdout.{json,csv}`, the
   experiment trees, and `smoke_raw/VERDICT.md`.
2. Judge the **raw** smoke against all of: both CSV layers and every candidate
   `CONTEXT.md` raw-only; all seeds and carried candidates bound to raw design
   rows (no `[drop]`, every `screened_out.json` empty); no candidate imports
   the featurizer; `any_csv_generating_model` and `any_manifest_gt_named`
   present in `holdout.json` (and false); the new metric columns present in
   `holdout.csv`; lenses 0–5 appear across the 6 candidate briefs (lens 6 is
   covered by the deterministic test); the run reached final evaluation with
   no traceback. Judge the **featurized** smoke on: completes, no traceback,
   metrics present, leakage fields present.
3. If a criterion fails because of a defect you can fix with confidence
   (a wrong path in the verifier, a missing field), fix it with a test, commit,
   delete `progress/P7.done` and write `progress/P7.retry` with one line on
   why, and **do not write `P8.done`** (end the session); the driver will re-run P7 (at most twice in total) and then this phase again.
   Otherwise record the failure.
4. Write `$WORK_ROOT/VERDICT.md`: one line per criterion, pass/fail, evidence
   path.
5. Write `$WORK_ROOT/HANDOFF.md` for the user: the final commit SHA; how to
   fetch the branch into `main` (`git fetch $REPO consolidate/2026-09`); the
   verdict summary; what P9 is about to launch (arms, repeats, cost); and
   the commands for a featurized arm from the same commit (§9) for the user
   to run later if they want the raw-vs-featurized gate.
6. State clearly in `VERDICT.md` whether the **raw smoke passed every
   criterion**: P9 launches the raw sweep only on a pass.

**Accept:** `VERDICT.md` and `HANDOFF.md` exist; tree clean; `P8.done`.

### Amendment of 2026-09-16 (after the first smoke round)

The smoke verdict showed two things (see `VERDICT.md`, `P9.blocked` of the
first attempt, and `docs/consolidation_decision_record.md`):

1. The restored audit caught the held-out model's name in the *other* feature
   regime's manifests; fixed in `74df77c`.
2. An agent-written candidate (`encoding_difficulty`) imported `parse_motifs`
   from `src/subjective_randomness/features.py` inside its `compute_features`
   hook. That function *is* the held-out `falk_konold_dp` decision variable.
   The raw arm was therefore not measuring discovery. Nothing enforced the
   brief's "numpy, pymc, pytensor only" rule, and the feature library was
   readable in the agents' tree.

The user's decision: **raw is the only mode, and the feature code must be
unreadable by agents, not merely un-imported.** The featurized pipeline is
removed. Phases P9–P15 below replace the earlier P9–P11 (sweep, evaluation,
results), which move to P13–P15. `SWEEP_ARMS` in the inputs is obsolete: there
is one pipeline now.

### P9 — Raw is the only mode

**Goal:** one data path. Agents receive `sequence_a, sequence_b,
participant_id, trial_index, chose_left` and nothing else, everywhere: the
holdout harness, the live outer loop, the design stage. No featurizer exists
in `src/pipelines/`. The raw seed sets are *the* seed sets.

**Tests first:**
- a test that `run_inner_model_loop_programmatic`, `run_design_programmatic`
  and the collect stage no longer accept a featurizer / `raw_features`
  argument and always write raw rows (adapt `tests/test_raw_inner_loop.py`);
- a test that `src/pipelines/`, `src/models/`, `src/critique/`,
  `src/subjective_randomness/holdout_recovery.py`, the seed sets and
  `src/subjective_randomness/model_families/` import nothing from
  `src.subjective_randomness.features` — check by AST over those trees
  **and** by importing each entry module in a subprocess with
  `sys.modules["src.subjective_randomness.features"] = None`;
- the seed-parity test pinned to a frozen fixture: a small CSV of stimuli
  with the expected `compute_features` outputs of each raw seed (generate it
  once from the current seeds in the red step, commit it), so the seeds are
  regression-tested without the feature library;
- the existing raw tests (`test_raw_config_validation`,
  `test_verify_raw_features_run`, `test_raw_features_seeds`) adapted so "raw"
  is the default and only behaviour.

**Do:**
1. Delete the featurized seed sets and promote the raw ones:
   `src/pipelines/outer_loop/projects/subjective_randomness/seed_models/` ←
   the contents of `seed_models_raw/`; `src/subjective_randomness/pymc_model_families/`
   ← the contents of `pymc_model_families_raw/`; remove the `_raw` dirs; fix
   every reference (configs, launchers, `holdout_recovery_array.sbatch`,
   `POOL_MODELS_REL`, docs, tests). The manifest mirror test keeps asserting
   the two manifests agree.
2. Delete `src/pipelines/outer_loop/projects/subjective_randomness/preprocess.py`,
   `scripts/subjective_randomness/preprocess.py`, and the featurizer concept
   from the pipeline: `Featurizer`/`load_featurizer` and every
   `featurize_path` / `featurize` parameter in `featurizer.py`, `collect.py`,
   `eig.py`, `orchestrator.py`, `inner_loop/run.py`, `runtime/config.py`.
   `featurizer.py` keeps only `RAW_RESPONSE_COLUMNS` (rename the module if
   its name no longer fits). Remove the `raw_features` flag everywhere: the
   raw behaviour is unconditional. Remove `pool_models_dir` only if the
   project pool is now unambiguous; otherwise keep it.
3. Holdout harness: evaluation rows and ground-truth generation are raw rows
   (`feature_rows` → raw rows; every model computes its own features through
   its hook, as `_augment_rows_with_features` already does at predict time).
   The same-value collision rule can go; keep the "a model redefines a
   harness column" error for the five raw columns. Delete
   `holdout_recovery_raw_features.yaml`, `run_raw_features_arm.sh`,
   `compare_raw_features_arm.py`; `holdout_recovery_faithful.yaml` is the
   raw config now (say so in its header).
4. Family twins: `model_families/common.py` must not import
   `src.subjective_randomness.features`; inline the helpers it needs. The
   research-library modules that only user-side analysis uses
   (`features.py`, `sequence_stats.py`, `model_recovery.py`,
   `pymc_recover.py`, the analysis scripts) stay, but nothing that runs in a
   holdout task or the live pipeline may import them.
5. Delete the featurized-mode tests (`test_featurizer_parity.py`,
   `test_outer_loop_featurizer_resolution.py`, `test_subjective_randomness_preprocess.py`,
   featurized cases elsewhere) rather than keeping dead branches alive.
6. Docs: `CLAUDE.md` (one data path; the two-seed-sets paragraph goes),
   `README.md`, `scripts/subjective_randomness/README.md`, `docs/raw_features_arm.md`
   (mark superseded: raw is the only mode).

**Accept:** the new tests pass; the fast suite's failing set ⊆ baseline
**minus** the deleted tests (list the deleted node IDs in the done file);
`git grep -n "raw_features\|featurize_stimulus\|load_featurizer\|preprocess.py"`
over `src/pipelines src/models src/critique src/subjective_randomness/holdout_recovery.py scripts/subjective_randomness/slurm`
returns nothing but comments that explain the history.

### P10 — The agents' tree contains no feature code; imports are gated

**Goal:** in every array task, the tree the agents can read holds no code
that computes stimulus features except the *visible* seeds' own hooks, and
a candidate that imports anything outside an allowlist is rejected at
admission with the reason in the ledger.

**Tests first:**
- `tests/test_agent_tree_isolation.py`: build the agent tree exactly as the
  array task does (factor the rsync exclude list into a file both the sbatch
  and the test read, e.g. `scripts/subjective_randomness/slurm/agent_tree.exclude`)
  into a temp dir from the repo, and assert: none of
  `src/subjective_randomness/features.py`, `sequence_stats.py`,
  `stimulus_design.py`, `model_recovery.py`, `pymc_recover.py`,
  `model_families/`, any `preprocess.py`, `ground_truth_models.py`,
  `evaluate_recovery.py`, `gt.txt`, the holdout configs, or the literature
  test files exist in it; and a source grep for
  `def parse_motifs|def featurize_stimulus|def multiscale_local_imbalance|def occurrence_probability|def periodicity_score`
  matches only files under the visible seed directories.
- `tests/test_candidate_import_gate.py`: `_admit_candidate` rejects a
  `candidate.py` whose AST contains an `import`/`from` (at any depth, inside
  functions included) of a module not in the allowlist — with a ledger
  entry `rejected: forbidden import <module>` — and admits one that uses
  only the allowlist. Allowlist: `numpy`, `pymc`, `pytensor`, `arviz`,
  `scipy`, `math`, `itertools`, `functools`, `collections`, `re`, `typing`,
  `dataclasses`, `statistics`, `operator`. Same gate for the critique
  agent's `test_statistic` code.
- a test that the candidate brief and the critique brief state the
  allowlist and say every helper must be written in the file itself.

**Do:**
1. **Separate the harness from the agents' tree.** The array task keeps a
   full checkout for the *harness process* (e.g. `$WORK_ROOT/harness_repo`,
   staged once by the setup job, read-only) and a scrubbed **agent tree**
   (`$RUN_REPO`, rsync with the exclude file, the held-out seed removed, the
   manifests scrubbed) that holds `_runs/` and only what agents need. The
   harness runs from the full checkout with the run root pointing into the
   agent tree; every place the inner loop uses `here()` / `REPO_ROOT` to
   decide what the candidate or critique agents may see (`--add-dir`,
   opencode `external_directory`, the agents' cwd and `PYTHONPATH`) must take
   an explicit **agent root** instead. If the candidate brief's self-check
   snippet needs `src/models/pymc_inference.py` importable from the agent
   tree, include the minimum modules it needs in the tree (they must pass
   the isolation test) or replace the snippet with a check the harness
   runs. If this separation proves more invasive than a session allows,
   the fallback is: build the exhaustive evaluation pool in the setup job
   (pristine) and load it from `eval_stimuli.json`, read `DEFAULT_PARAMS`
   from the pristine family snapshot by parsing only, generate ground-truth
   responses from the raw PyMC registry, and make the harness import none of
   the excluded modules inside the copy — with the same isolation test as
   the acceptance criterion. Record which route you took in the done file.
2. The rsync exclude file, used by the array sbatch and the test.
   The opencode deny globs get the same paths.
3. The admission gate for candidates and the critique's test statistic:
   static AST check before any import of the file; ledger entry; the briefs
   updated (`pymc_theory.md` and the candidate/critique context writers).
4. The verifier (`verify_raw_features_run.sh`, rename to
   `verify_holdout_run.sh` if you like): assert the forbidden paths are
   absent from every task's agent tree, assert no admitted model or
   candidate imports outside the allowlist, keep the raw-CSV, CONTEXT, drop,
   `screened_out.json` and completion checks, and print the agent-root and
   harness-root paths it checked.

**Accept:** new tests pass; fast suite failing set ⊆ (baseline minus deleted
tests); `git grep` of `here()` and `REPO_ROOT` in `src/pipelines/inner_loop/`
and `src/runtime/coding_agent.py` shows no remaining use that decides agent
visibility.

### P11 — Submit the smoke cell (may `sbatch`)

One cheap raw cell from the consolidated commit, two experiments, one
candidate round, exactly as the earlier P7 raw smoke but through the
standard launcher (there is no arm any more):

```bash
cd "$REPO"; export REPO="$REPO"
SMOKE=1 N_EXPERIMENTS=2 INNER_LOOP_ITERATIONS=1 BASE_SEED=100 \
  WORK_ROOT="$WORK_ROOT/smoke_round<k>" \
  bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
# verifier afterany the analysis job, as in P7
```

`<k>` is this phase's round (`progress/P11.retry*` markers count the
re-opens; the first round is `smoke_round1`). Write
`progress/isolation_smoke_jobs.json` as
`{"raw": {"work_root": "...", "job_ids": ["<setup>", "<array>", "<analysis>", "<verify>"]}}`.
Do not wait.

**Accept:** the jobs file exists with numeric ids; tree clean; `P11.done`.

### P12 — Smoke verdict

**Preconditions (driver-checked):** the smoke jobs have left the queue.

Judge the cell against all of: both CSV layers and every candidate
`CONTEXT.md` raw-only; no `[drop]`, every `screened_out.json` empty; the
verifier passes, including the **agent-tree isolation check**; no admitted
model or candidate imports outside the allowlist, and any candidate that
tried is in the ledger as `rejected: forbidden import`; the leakage-audit
fields present and false; the new metric columns present; lenses 0–5 across
the six briefs; final evaluation reached with no traceback.

If a criterion fails because of a defect you can fix with confidence: fix it
with a test, commit, write `progress/P11.retry<k>` with one line on why,
delete `progress/P11.done`, and **do not write `P12.done`** — end the session.
The driver re-runs P11 and then this phase again (at most three smoke rounds).
Otherwise write `$WORK_ROOT/VERDICT.md` (one line per criterion, pass/fail,
evidence path) and `$WORK_ROOT/HANDOFF.md` (final commit, how to fetch the
branch, what P13 is about to launch and its cost), then `P12.done`.

**Accept:** `VERDICT.md` and `HANDOFF.md` exist; tree clean; `P12.done`.

### P13 — Launch the 5-repeat recovery sweep (may `sbatch`)

**Preconditions:** `VERDICT.md` says the smoke passed every criterion;
otherwise write `P13.blocked` — a 20-cell sweep on an unverified pipeline
is the user's call.

```bash
cd "$REPO"; export REPO="$REPO"
N_REPEATS=$SWEEP_N_REPEATS BASE_SEED=$SWEEP_BASE_SEED MAX_PARALLEL=$SWEEP_MAX_PARALLEL \
  WORK_ROOT="$WORK_ROOT/sweep" \
  bash scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh
# verifier afterany the analysis job
```

Do not raise `MAX_PARALLEL`. Write `progress/sweep_jobs.json` as
`{"raw": {"work_root": ".../sweep", "job_ids": ["<setup>", "<array>", "<analysis>", "<verify>"]}}`.
Record the expected cost (about one earlier sweep's Gemini spend) and wall
time (about a day) in the done file. Do not wait.

**Accept:** `sweep_jobs.json` exists with numeric ids; tree clean; `P13.done`.

### P14 — Submit the RMSE evaluation job (may `sbatch`)

**Preconditions (driver-checked):** every sweep job has left the queue.

1. If `scripts/subjective_randomness/recovery_report.py` does not exist in
   the clone, add it to the P3 item 6 specification, tests first, and commit.
2. Add `scripts/subjective_randomness/slurm/evaluate_recovery_sweep.sbatch`
   (commit it): `--partition=normal --time=1-00:00:00 --cpus-per-task=8
   --mem=32GB`, caches off `$HOME`, cwd `$REPO`, Python `$VENV_PY`; each step
   fails loudly:
   a. the verifier on `$WORK_ROOT/sweep`;
   b. `compare_matched_cells.py --sweep-a $ITER2_SWEEP --sweep-b $WORK_ROOT/sweep --out $WORK_ROOT/analysis/vs_iter2`,
      the same against `$ITER3_SWEEP` (`vs_iter3`) and `$BASELINE_SWEEP`
      (`vs_baseline`). Those sweeps were featurized runs; their cells are
      re-scored on the common pool with their own archived models, so the
      comparison is valid as a reference, and confounded as an ablation;
   c. `oracle_admitted_models.py --result <cell>/holdout.json --steps final`
      for every `run<r>/<gt>` cell;
   d. `recovery_report.py --sweep $WORK_ROOT/sweep --label consolidated_raw
      --compare vs_iter2=… --compare vs_iter3=… --compare vs_baseline=…
      --oracle-glob '$WORK_ROOT/sweep/run*/*/oracle.json'
      --out $WORK_ROOT/analysis/RESULTS_auto.md`.
   Comparison cells whose archives cannot be re-scored are listed, never
   skipped silently.
3. Submit it (`--output=$WORK_ROOT/analysis/evaluate_%j.out`,
   `--export=ALL,WORK_ROOT=…,REPO=…,VENV_PY=…` plus the sweep roots); write
   `progress/analysis_jobs.json` as
   `{"analysis": {"work_root": "$WORK_ROOT/analysis", "job_ids": ["<id>"]}}`.
   Do not wait.

**Accept:** the sbatch is committed; `analysis_jobs.json` exists; tree clean;
`P14.done`.

### P15 — Results: the RMSE evaluation report

**Preconditions (driver-checked):** the evaluation job has left the queue.

1. Read `$WORK_ROOT/analysis/evaluate_<id>.out` and every output under
   `$WORK_ROOT/analysis/`, the sweep's `test_retest.{json,csv}`, the
   verifier's `VERDICT.md`, and the per-cell `holdout.json` leakage fields.
2. If the evaluation job failed on a defect you can fix with confidence, fix
   it with a test, commit, write `progress/P14.retry` with one line on why,
   delete `progress/P14.done`, and do not write `P15.done`; the driver
   re-runs P14 (at most twice in total) and comes back here. Otherwise
   record the failure and report what did complete.
3. Write `$WORK_ROOT/RESULTS.md` — the RMSE evaluation the user asked for:
   - **Engineering gate:** cells completed (of 20), verifier verdict
     including isolation, `[drop]` count, non-empty `screened_out.json`
     count, forbidden-import rejections from the ledgers, leakage-audit
     fields, per-task spend from the logs.
   - **Recovery table (primary):** per ground truth, final-step RMSE mean ±
     sd, median, min, max over the 5 repeats; KL regret alongside; Pearson r
     descriptive; the best model per cell.
   - **Matched-seed deltas** against iteration 2 (5 repeats), iteration 3
     (3) and the pre-campaign baseline (5), per cell and per ground truth,
     each labelled as **confounded** (those runs supplied features; this one
     does not, plus every loop change and the manifest scrub).
   - **Three-bucket diagnostic:** per ground truth, cells with an
     oracle−incumbent gap above 0.02 (selection), cells whose oracle-best is
     itself poor (discovery), lost-incumbent events (retention); name them.
   - **Interpretation:** what improved and what did not against the 0.02
     practical margin at n = 5, no significance claims; which ground truth
     drives the mean; what to run next.
   Update `HANDOFF.md` with a pointer to `RESULTS.md` and the final commit.

**Accept:** `RESULTS.md` exists; tree clean; `P15.done`.

## 5. Test-baseline discipline

`baseline_failing_tests.txt` (P0) is the reference. After every phase, the
failing set must be a subset of it. A failure that appears and is caused by
the phase's change is fixed in that phase. A failure that appears and is an
environment collection error is recorded in the done file. Nothing else is
acceptable.

## 6. Commit discipline

One commit per numbered behavioural step where practical; merge commits with
`--no-ff`; messages say what and why, and name the plan phase (`[P2]`). Never
amend a merge. Never rewrite history.

## 7. Stop conditions (write `P<k>.blocked`)

- a frozen input's SHA differs from §1;
- a supposedly raw agent-facing artefact carries an engineered column after P2;
- a raw model reaches design only to fail binding after P2;
- the fast suite gains a failing node ID you cannot attribute and fix;
- the arm C merge cannot be reconciled with iteration 3 within the P1 budget;
- a submit script needs more than a knob-name change to run;
- P13 would launch the sweep although the smoke did not pass every
  criterion in `VERDICT.md`;
- after P10, any file that computes stimulus features (other than the
  visible seeds' own hooks) is present in an agent-visible tree, or a
  candidate's forbidden import reaches the fitting stage instead of the
  admission gate;
- anything that would require touching `$SOURCE_REPO`, pushing, or cancelling
  a job.

## 8. Metric protocol (for the decision record and the handoff)

For ground-truth probabilities `q_i` and recovered `p_i` on the common pool:

- **primary**: `RMSE = sqrt(mean((p_i − q_i)²))`;
- **secondary**: `KL regret = mean(q_i·log(q_i/p_i) + (1−q_i)·log((1−q_i)/(1−p_i)))`
  with `p`, `q` clipped to `[1e−9, 1−1e−9]`;
- **descriptive**: Pearson r, bias `mean(p_i − q_i)`, calibration slope and
  intercept (OLS of p on q).

The repeat is the stochastic unit. Report every matched-seed paired delta, then
mean/median/min/max per ground truth. No stimulus bootstrap. Predeclared
practical margin: **0.02 RMSE**. Changing it after seeing results needs a
written amendment.

## 9. Comparisons against the earlier, featurized sweeps

There is no featurized arm any more (P9). The earlier sweeps (iteration 2,
iteration 3, the pre-campaign baseline) supplied engineered feature columns
to their agents, so a matched-seed comparison against them measures
*everything* that changed at once: the representation now has to be
discovered, the loop changes of iterations 1–5 as consolidated, and the
manifest scrub. Report those deltas as reference points with that label,
never as an ablation of any single change. The practical margin (0.02 RMSE)
still applies to reading them.

If a single-change ablation is ever wanted, run it as two arms from one
commit with matched seeds and compare with `compare_matched_cells.py`; the
gate would be: no stable ground truth (`falk_konold_dp`,
`finite_experience_occurrence`, `motif_stack`) worse than +0.02 mean RMSE
and none worse in every paired repeat, with `local_representativeness`
reported cell by cell and never deciding alone.

**Diagnostics:** if the incumbent regresses but the oracle-best does not,
investigate selection; if both regress, discovery; if a lost-incumbent entry
explains it, retention.
