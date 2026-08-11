# Codex Review Implementation Notes

This document records the changes made after reviewing the
`fix-tests-and-refactor` branch. The implementation focused on correctness,
fail-loud behavior, reproducibility, and keeping research-code boundaries easy
to audit.

## Executable integrity and CI

- Fixed the syntax error in
  `scripts/subjective_randomness/holdout_test_retest.py`.
- Added `tests/test_python_sources_compile.py`, which compiles every Python file
  under `src/`, `scripts/`, and `analysis/`. This catches syntax errors in
  standalone scripts that pytest would not otherwise import.
- Added `.github/workflows/tests.yml` to install locked dependencies, run the
  compilation test, and run the non-slow test suite on pushes and pull requests.

## Strict model manifests

- Strengthened `src/models/model_manifest.py` and
  `src/models/theorist/loader.py` so manifests must contain a valid `models`
  list with non-empty, unique model names.
- A manifest that names a missing `<model>.py` file now raises an attributed
  `FileNotFoundError`; loaders and EIG no longer silently omit the missing
  model and renormalize over the survivors.
- Added regression coverage for malformed manifests, duplicates, invalid names,
  and missing model files.

## Inference safety

- Model pruning is skipped when any PSIS-LOO comparison row is marked
  unreliable. This prevents an unreliable estimate from moving a candidate out
  of the active model set.
- Export refuses to create `best_model.py` when the selected model's LOO result
  is missing or unreliable.
- Sampling diagnostics are rerun for both on-disk and in-process cached fits, so
  loading a cached suspect fit cannot make its warning disappear from the run
  log.
- Small-fixture integration tests explicitly neutralize their stochastic
  Pareto-k flags only where those tests exercise unrelated export and cache
  wiring. Dedicated deterministic tests verify that production export and
  pruning remain blocked.

## Participant collection

- Participant backends may declare a `max_concurrency`. Hosted API participants
  permit concurrent calls, while the shared local Hugging Face model is limited
  to one call at a time because generation consumes shared model state and
  PyTorch RNG state.
- LLM collection now writes `collection_stats.json` and requires exactly one
  valid response for every participant/stimulus pair. Errors or unparseable
  replies cause the partial dataset to be rejected before `responses.csv` is
  written.
- Collected `chose_left` values must be finite binary values (`0` or `1`); values
  such as fractions, out-of-range numbers, NaN, or labels are rejected.

## Probability and registry contracts

- Added `src/models/probability.py` as the shared validator for scalar
  probabilities, categorical distributions, and probability arrays.
- Ground-truth generation no longer defaults a missing `left` probability to
  `0.5`. Model distributions must contain exactly the expected response keys,
  contain finite values in `[0, 1]`, and sum to one.
- PyMC prior- and posterior-predictive probabilities are checked for finite
  values in `[0, 1]` before use.
- Registry roots, theory names, theory weights, and reserved mass are now
  validated. Negative, non-numeric, NaN, or infinite weights fail loudly.
- EIG loads explicit registries through the canonical registry loader and raises
  if an explicitly requested registry does not exist.

## Project assets and loader consolidation

- `problem_definition_path`, `references_dir`, and `project_prompts_dir` now all
  resolve through the declared `PROJECT_ASSETS_DIR`.
- Checked-in inputs for `number_game`, `think_aloud_game24`, and
  `subjective_randomness` were consolidated under
  `src/pipelines/outer_loop/projects/`. Their contents were not rewritten.
- Byte-identical subjective-randomness reference duplicates were removed from
  the root `projects/` tree. The root tree is reserved for generated run output.
- Removed the duplicate permissive ground-truth loader from the outer-loop
  orchestrator; it now imports the canonical fail-loud loader from
  `src/models/project/ground_truth.py`.

## Dead-code cleanup

- Removed the orphaned `_server_reachable`, `_start_experiment_server`,
  `_run_one_participant_browser`, and `_rows_from_trial_data` helpers after
  confirming they had no production callers.
- Removed their obsolete tests and updated `REFACTOR_NOTES.md` accordingly.

## Verification

The final tree was checked with:

- Full pytest suite: **992 passed, 3 skipped, 13 warnings**.
- Ruff over every touched Python file: passed.
- Repository-wide Python compilation test: passed.
- `bash -n` over shell scripts: passed.
- `uv lock --check`: passed.
- `git diff --check`: passed.

The remaining warnings are ArviZ Pareto-k warnings from deliberately small MCMC
fixtures. They are retained as visible warnings; production model export now
blocks the corresponding unreliable selected result.

## Post-review adjustments (2026-08-09)

A follow-up review of this implementation accepted it with four adjustments:

1. **Registry docs matched to the new contract.** `_load_model_weights` now
   raises on a missing registry file, but `prompts/2_design.md` and the
   CONTEXT.md text in `orchestrator.py` still promised "absent registry =
   uniform". Both now say: the orchestrator creates `model_registry.yaml`
   before design, an *empty* registry means uniform, a *missing* path is an
   error. (In orchestrated runs `init_registry` always precedes design, so the
   raise cannot fire there.)
2. **Incomplete LLM collections are retried and preserved, not just refused.**
   Each participant trial now gets one retry on a transient backend error or
   an uncommitted reply (`_TRIAL_ATTEMPTS` in `collect.py`; the side
   assignment is drawn before the attempts, so counterbalancing is
   unaffected). If the collection still comes up short, the rows that were
   collected — paid LLM output — are saved to `responses_rejected.csv` before
   the completeness gate raises, instead of being discarded with only the
   counts.
3. **Pruning reliability gate narrowed.** Any single unreliable LOO row used
   to disable all pruning, so one flaky agent-written candidate could make the
   active model set grow-only for the rest of a run. Now: an unreliable rank-0
   baseline still blocks all pruning (every `elpd_diff` is measured against
   it), and a model whose own row is unreliable is never pruned on it — but a
   reliable clear loser is pruned even when an unreliable bystander exists.
4. **A refused export still writes `model_posterior.json`.** The reliability
   gate in `_export` now fires after the posterior + comparison record is on
   disk (it is a diagnostic record, not an endorsement) and continues to block
   `best_model.py`. A refused run can therefore be diagnosed from its own
   results directory.

Verified after the adjustments: full pytest suite (fast + slow) green, ruff
clean on every touched file.
