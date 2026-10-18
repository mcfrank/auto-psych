# Cleanup backlog (audit of 2026-10-16, branch `number-game`)

Code that can be removed or minimized without changing behaviour, found by
three read-only audits and spot-checked with grep. **To do after the merges
with `main`**: line numbers will have moved, so re-check each item with
`grep -rn <name> src scripts tests` (and `*.sh`, `*.sbatch`) before deleting.

Out of scope (user, 2026-10-16): the Gemini/opencode backend, Sherlock
support and the live Prolific/Firebase path stay. Dead code *inside* them is
fair game.

Order of work: Tier 1 by area, one commit per area with its tests; Tier 3
alongside, file by file; Tier 2 item by item once decided. Full suite once at
the end.

## Verified plan (2026-10-18, against `main` after the merges)

Re-checked item by item against the merged tree, `src/rsa/` and `scripts/rsa/`
included, counting any script run by hand (README / usage docstring / `python
-m` in docs) as a caller. Branch `cleanup`; one commit per step, each with the
listed tests; Tier 3 comments trimmed in the same files; full suite once at the
end. Must stay (other code uses them): `run_coding_agent`'s `env` and
`sandbox` (rsa passes `env=`, `sandbox=False`); `convergence_problems` (rsa's
numpy fits); `collect._results_request` / `_poll_prolific_until_target` /
`_pause_unfilled_study` / `_PROLIFIC_MAX_WAIT_SEC` and
`critique_round._format_critiques_md` / `_usable_test_statistics` (rsa imports
them by name); `pymc_orchestrator._compare` (tests monkeypatch it there); the
`likelihood.py` CLI (documented in its docstring); `reporting.aggregate_holdout_trajectories`
(`scripts/subjective_randomness/plot_holdout_curves.py`, run by hand).

1. **Dead modules and names** (no caller anywhere): `deployment/firestore.py` +
   test; `src/models/theorist/` + `src/models/__init__.py` re-exports +
   `project/__init__.py` (and `synthetic_data`'s unreachable branch, which uses
   it); `holdout_data.validate_raw_pool_models` + `test_raw_config_validation.py`,
   `strip_to_raw_columns`; `leakage_audit._distinctive_param_names`;
   `observability.append_validation_failure`; `prolific.list_submissions`
   (move its pagination test to `list_studies`); `config.py`'s five dead names;
   `common.softplus`, `n_switches`, `bernoulli_log_prob`; `_ScenarioState.marginal_gains`;
   `task_for_columns`; `number_game.data.load_cells`; the bare-string manifest
   shim; `registry`'s `"probabilities"` key and `normalize_theories`;
   `manifest.experiment_number_from_dir` (`build_manifest` takes an int);
   `prior_predict_p_left_batch`, `expected_information_gain_prior_pymc`;
   `model_zoo._admit_candidate`; unread result keys (`run_pymc_inner_loop`'s
   `live_models` / `ledger_path`, `_export`'s `*_path` and `comparison`).
2. **Never-passed parameters**: `eig`'s `fit_draws/tune/chains` (check
   `test_eig_pymc` time) and `random_seed`; `spawn_cc_agent`'s `prompt_key`,
   `timeout_secs`, `model`, `write_context(extra=)`; `run_inner_model_loop_programmatic`'s
   `complexity_prior_const`; `build_eval_stimuli`'s `extra_excluded_pairs`;
   `run_coding_agent`'s `extra_args`, `on_summary`, `wait_out_usage_limits`;
   `participants.py` / `smoke.py` / `llm.py` dead kwargs (update
   `scripts/smoke_open_participant.py`); `pruned=[]`; `n_critique_replicates`,
   `fit_time_limit_sec`.
3. **Fail loudly instead of falling back**: `orchestrator.py`'s `return []`
   paths and the no-loadable-models `rows = []` (769-773, new); `project_id or …`;
   the repeated ground-truth and config checks; the matplotlib ImportError skip
   in `holdout_test_retest.py`; the redundant `collect.py` ImportError guard;
   `scoring` / `model_zoo` / `candidate_agent` empty-comparison fallbacks;
   unreachable validator branches; internal-boundary re-validation (EIG weights,
   `_validated_p`'s range check, worker-count guards, `bad_k_tolerance`).
4. **Deduplicate**: `run.py`'s 30 arguments × 6 (pass `Args`); `_drop_*_models`
   tail (empty check optional for the CV one); score → compare → record (4
   copies; keep `_compare` looked up in `pymc_orchestrator`); experiment-number
   parsing; `extract_observed` / `make_stim_data` dtype loop (keep the plain
   `ValueError`); `prior_predict_p_left` over `_draws`; `mean_log_likelihood` /
   `log_likelihood`; `posterior.py`'s not-converged block; `check_candidate`'s
   gates (admission also needs `hypothesis.md`); `collect.py`'s serial browser
   loop and `/results` fetch; firebase polling loops; `model_zoo._untrusted` /
   `scoring._excluded`; `DATA_CSV` (6 copies; put it in `number_game/data.py`);
   `FittedModel.convergence_problems()` method; `plot_holdout_recovery` /
   `reanalyze_holdout_exhaustive` output block (`format_number` prints "n/a"
   where `_fmt` printed "undefined").
5. **Merge the impossible-model CLI into `holdout_recovery.py`** (both
   impossible sbatch files, `impossible_models/__init__.py`, docs, ~8 tests).
6. **Tier 2, as decided**, then Tier 3 leftovers; full suite.

Bugs found in passing (fix separately): `run.py` receives `agent_timeout_sec`
but never passes it to `spawn_cc_agent`, so `3_implement` always gets 900 s;
`results_collection.summarize_model_posterior` picks the posterior argmax;
`eig._write_output` crashes on random picks.

Tier 2 revisions after the re-check: keep the pooled aggregation (only the
plotnine figure half, ~650 lines, and the `plotnine` dependency can go); keep
the exhaustive-pool reanalysis chain and `validate_lazy_eig.py` (documented
tools).

## Tier 1: safe (no behaviour change), ~2,000 lines incl. tests

### Pipelines (`src/pipelines/`)
- `outer_loop/run.py:103-581`: ~30 arguments spelled out six times
  (`_run_agent`, `_run_experiment`, `_run_experiment_stages`, `main`). Pass the
  `Args` object plus the few resolved values (backend, `fit_kwargs`, hints,
  `run_label`, `participant_model`). ~130 lines; ~10 tests call these directly.
- `outer_loop/deployment/firestore.py` (90) + `tests/test_deployment_firestore.py`
  (38): nothing imports `validate_submit_payload` / `responses_to_csv`.
- `inner_loop/model_zoo.py` `_admit_candidate`: test-only forward to
  `_admit_candidate_with_reason(...).admitted` (7 test files). ~28.
- `outer_loop/collect.py:388-447`: the serial browser loop in
  `_collect_from_firebase` duplicates `_run_one_participant_firebase`; a pool of
  one replaces it. Also the redundant `ImportError` guard at 213-215. ~60.
- `model_loop_runner.py:367-474` `run_inner_model_loop_programmatic`:
  `complexity_prior_const` never passed; the "None ⇒ inherit" `extra` dict over
  9 kwargs could be direct pass-through. ~30.
- `inner_loop/scoring.py` `_export` and `_best_exportable_model:74`: fallbacks
  for an empty comparison table never run (`compare_table` always returns a row
  per model); same in `_prune_losers` and `_write_existing_hypotheses`; returned
  keys `model_posterior_path`, `best_model_path`, `report_path`, `comparison`
  are never read. ~55; some tests pass `{}`.
- `check_candidate.py:101-159` repeats admission's cheap gates
  (`model_zoo._cheap_gate_rejection`, `model_zoo.py:1514-1537`): share them so
  the self-check cannot drift from admission. ~30.
- `model_zoo.py` `_drop_*_models` (three helpers, ~358/424/637) end with the
  same print / ledger "dropped" / raise-if-empty / rewrite-manifest: one helper.
  ~25.
- `pymc_orchestrator.py` (~358/710/760): the score → compare → record-history
  block appears three times; `_score` / `_compare` only forward (keep `_compare`'s
  name: tests monkeypatch it). ~25.
- `synthetic_data.py`: `_generate_from_pymc_models` and `_generate_from_models`
  share a participant/trial loop; the `cognitive_models_dir` branch (324-342) is
  unreachable (the only caller always passes `model_registry`). ~45.
- `critique_round.py`: `_persist_critique_results` reruns every statistic on the
  observed data (its "no usable statistic" raise can't fire); two identical
  `no_critique` dicts; a final internal `critiques.md` existence check. ~25.
- `outer_loop/eig.py`: `fit_draws` / `fit_tune` / `fit_chains` never passed;
  `random_seed` always equals `seed` (`orchestrator.py:597, 611`). ~15.
- `orchestrator.py:380` `spawn_cc_agent`: `prompt_key`, `timeout_secs` never
  passed; `model` only by `tests/test_agent_model_threading.py`;
  `write_context(extra=...)` passed by no one. ~15.
- Experiment number parsed from the directory name five ways
  (`orchestrator.py:118`, `:691` — silently defaults to 1 —,
  `model_loop_runner.py:62, 356`, `deployment/manifest.py:36`): one helper. ~15.
- `orchestrator.py`: fallbacks that hide errors, against fail-loudly —
  `_collect_llm_participant_programmatic` (494-501) swallows the participant
  model's error and returns `[]`; two other `return []` paths; the ground-truth
  name check (748-758) repeats `run.py:724-731`; `project_id or
  exp_dir.parent.name`; re-validation of `config.json` (679-689). ~35.
- `model_loop_runner.py:267-288`: raw-binding check in `_export_inner_loop_models`
  repeats the admission data contract (and silently skips when
  `participant_id` is hidden). ~22.
- `orchestrator_validators.py`: `best_model or max(posteriors)` fallback (313),
  zoo-name check in `_validate_model_set` (88-102, stale "repair loop"
  comment), unreachable `else (True, "No validator")`. ~25.
- Live collection duplicates: `/results` fetch/parse in `collect.py:288-299`
  and `450-464`; token read `collect.py:533` vs `firebase.results_token()`; two
  HTTP polling loops `firebase.py:70-138` and `579-610`. ~40.
- Small dead pieces (~55): `llm.py:52-58` `batch_dir` branch and the `state`
  param through `browser_steering.py:209`; `participants.py:136-144` four
  never-passed kwargs; `smoke.py:121-128` `instructions_html` / `debrief_html`;
  `manifest.py` re-checks of CLI-constrained values, double JSON serialisation,
  `manifest_dir` wrapper; `simulated_n_participants` / `or 1` fallback
  (`collect.py:256-260`, `orchestrator.py:739`); never-overridden
  `fit_time_limit_sec` / `time_limit_sec` / `n_critique_replicates`; dead
  `pruned=[]` history field; duplicate predicates `model_zoo._untrusted` and
  `scoring._excluded`; `test_only` hint fallbacks `candidate_agent.py:587-589`.
- Stale `sys.path.insert` inside functions (`orchestrator.py:653`,
  `model_loop_runner.py:594, 627`, `orchestrator_validators.py:69`) and
  "Extracted from…" headers (`model_loop_runner.py` claims re-exports that
  don't exist; `model_zoo.py`, `scoring.py`, `candidate_agent.py`). ~15.

### Models, comparison, critique, registry, tasks, number game
- `src/models/theorist/` package + `src/models/__init__.py` re-exports +
  `src/models/project/__init__.py`: serve only the unreachable
  `synthetic_data._generate_from_models` fallback (`tests/test_collect_pymc.py:62`);
  `get_model_names_from_manifest` duplicates
  `model_manifest.read_loadable_model_names`; `get_ground_truth_model_names` has
  no callers; the re-exports load eagerly on every `src.models.*` import. ~200.
- `pymc_inference.py`: `prior_predict_p_left` (196) is
  `prior_predict_p_left_draws` (234) + mean; `prior_predict_p_left_batch` (285)
  and `expected_information_gain_prior_pymc` (356) are test-only. ~70.
- `model_comparison/likelihood.py` CLI (`Args` / `main`): only
  `tests/test_cli_parsing.py:133`; `log_likelihood` is a 3-line forward used once
  (`posterior.py:361`). ~50.
- Convergence: `convergence_problems(idata, names)`, `convergence_problems_of`,
  `FittedModel.convergence_problems()`, `convergence_diagnostics_of` — four ways
  to one answer; make it two `FittedModel` methods. ~20.
- `data_binding.py`: `extract_observed` (340) copies `make_stim_data`'s (297)
  dtype-casting loop; share the loop, keep `extract_observed`'s `ValueError`
  (`model_zoo.py:828` classifies by it). ~15.
- `posterior.py`: the "did not converge" row + warning block is copied in
  `compare_table` and `_cv_compare_table`. ~12.
- Internal-boundary validation (~40): `validate_theory_weights` /
  `validate_probability` inside `eig_from_prior_means` / `_model_prior`;
  `_validated_p` (`eig_selection.py:94`) re-checks `_validated_p_left_draws`;
  `workers < 1`, `time_limit_sec <= 0`, `fit_workers < 1`, `default_fit_workers`'
  empty check, `_fit_executor`'s compile-dir count, `loo_diagnostics`'
  `bad_k_tolerance` range.
- Dead or test-only (~25): `_ScenarioState.marginal_gains`
  (`eig_selection.py:302`), `task_for_columns` (`tasks/__init__.py:104`),
  `load_cells` (`number_game/data.py:51`), the `pytensor < 2.31` import fallback
  (`model_loading.py:179`; lock pins 2.38.3), the bare-string manifest entry shim
  (`model_manifest.py:68-73`).
- `DATA_CSV` defined five times (`number_game/dataset.py`, three
  `scripts/number_game/*.py`, `tests/test_number_game_replication.py`): import it.
- `registry/io.py`: `normalize_theories` test-only; `get_model_weights`
  duplicates `load_registry(p)["theories"]`; the `"probabilities"` legacy key
  (38-39) appears nowhere.

### Research library, runtime, viewer, scripts
- `holdout_data.validate_raw_pool_models` (`holdout_data.py:239`) never called
  (its docstring says "at config resolution") + `tests/test_raw_config_validation.py`
  (147). Or wire it back in if the check is wanted. ~175.
- `holdout_data.strip_to_raw_columns` (220) test-only. ~17 + tests.
- `leakage_audit._distinctive_param_names` (25): logic inlined at line 160. ~17.
- `run_coding_agent` (`coding_agent.py:677`): every caller passes
  `stock=True, sandbox=True` and none passes `env`, `on_summary`, `extra_args`,
  `wait_out_usage_limits` (their justifications refer to tools removed in
  64ab8c6c). Drop `extra_args`, `wait_out_usage_limits`, `on_summary`. ~35.
- `runtime/config.py` dead names: `DEFAULT_SIMULATED_N_PARTICIPANTS`,
  `DEFAULT_MAX_VALIDATION_RETRIES`, `SECRETS_PATH`, `run_dir_for_state`,
  `agent_dir` (docstring still says `1_theory`); `problem_definition_path` /
  `references_dir` test-only. ~25.
- `observability.append_validation_failure` (31): no callers. ~14.
- `prolific.list_submissions` (188): test-only (collection uses
  `get_submission_counts`). ~38.
- `holdout_eval.build_eval_stimuli`'s `extra_excluded_pairs` (104, through 918):
  never passed. ~10.
- `holdout_test_retest.py:566-573`: skips the figure if matplotlib won't import
  (a hard dependency): an error-hiding fallback. ~4.
- `model_families/common.py`: `softplus` (69) unused; `n_switches` (146),
  `bernoulli_log_prob` (306) test-only. ~11.
- `plot_holdout_recovery.py:61-120` and `reanalyze_holdout_exhaustive.py:120-195`
  duplicate one output block (and a local `_fmt` duplicating
  `runtime/formatting.format_number`). ~50 if the reanalysis chain stays.
- `scripts/subjective_randomness/impossible_holdout_recovery.py` (206) is a fork
  of `holdout_recovery.py`, and `run_impossible_holdout_recovery_from_config`
  (`holdout_recovery.py:932-987`) only reads `gt_models_dir` / explicit
  `gt_models` from the config; one CLI can switch on `gt_models_dir`. ~250;
  update the impossible array sbatch (line 214).

## Tier 2: needs a decision, ~4,500 lines

- **Pooled-trajectory plotting** `reporting.py:193-1028`
  (`aggregate_holdout_trajectories`, `aggregate_tidy_rows`,
  `aggregate_exclusion_lines`, `holdout_combined_frames`,
  `holdout_trajectories_ggplot`, `plot_holdout_trajectories_combined`): no
  caller outside tests since `scripts/analysis/` was removed (328a67bd,
  8d29ad1c); `plotnine` is used only here. ~835 + ~470 of tests
  (`test_subjective_randomness_holdout_combined.py`, the aggregate tests of
  `test_honest_aggregates.py` — keep its `sweep_cells` / incumbent tests).
  CLAUDE.md and `docs/modeling_loop.md:1459` still describe it. Keep only if
  plotting from notebooks.
- **Impossible-model Slurm chain** (`impossible_holdout_setup.sbatch`,
  `impossible_holdout_recovery_array.sbatch`,
  `submit_impossible_holdout_test_retest.sh`, `run_impossible*_test_retest.sh`):
  a parallel fork of the faithful chain; merge like
  `scripts/number_game/slurm/submit_recovery.sh` (20 lines of overrides). ~450.
  Real merging work; it is the literature sweep's control.
- **Exhaustive-pool reanalysis chain** (`reanalyze_holdout_exhaustive.py`,
  `slurm/reanalyze_setup.sbatch`, `slurm/reanalyze_exhaustive_array.sbatch`,
  `slurm/submit_reanalyze_exhaustive.sh`): re-scores old sampled-pool runs; both
  current configs use the exhaustive pool. Also frees `reevaluate_trajectories`'
  `eval_pool_override` / `gt_models_dir` (`holdout_eval.py:911`). ~530; docs in
  `README.md` and `slurm/README.md:248-310`.
- **`validate_lazy_eig.py`** (498) + test (73): one-off validation; CLAUDE.md,
  `eig.py:292`, `eig_selection.py:56` cite it as evidence. ~570.
- **Pure-Python twins** `model_families/` (only `DEFAULT_PARAMS` is read in
  production, `holdout_data._family_default_params:84`; the rest is a test
  oracle for ~22 test files; stale package docstring) ~600, and the old
  featurizer **`features.py`** (415; test oracle only). Trade-off: lose the
  independent cross-check tests.
- **Assets**: `projects/think_aloud_game24/` (~430; no registered task, so it
  can't run; its prompt is for the removed theorist stage),
  `seed_models/archive_hero_run_2026_07/` (~440; README says kept for
  reproducibility), `projects/subjective_randomness/evaluate_recovery.py` (303;
  no script runs it, legacy `inner_loop_model` shims).
- **Old-run-format compatibility** (~100): bare-list `starting_models.json`
  (`model_loop_runner.py:132-193`), "model_loop without recorded input"
  (`model_loop_runner.py:535-543`), the `/results` `project_id` / `run_id`
  fallback (`collect.py:320-326, 526-529` + JS), `model_posterior.json` written
  twice in `scoring._export`, viewer legacy parsing (`viewer/scan.py:98-147`:
  `theory_report.md`, `inner_loop_model` heuristic, `design_rationale.md`).
  Only needed to resume or view pre-October runs.
- **Two concurrent-fit mechanisms** in `pymc_inference.py` (the
  `ProcessPoolExecutor` pool, ~1853-2021, vs per-request time-limited children,
  ~1670-1850): `_fit_outcomes` could use the second with no limit. ~150;
  cancellation under `stop_on_failure` differs, and the pool's fingerprint
  cross-check would go.
- **Complexity prior**: always 0 (`DEFAULT_COMPLEXITY_PRIOR_CONST = 0.0`, no
  config sets it); `model_complexity`, `posterior.py:60` and the
  `complexity_prior_const` plumbing. ~45.
- **`posterior.py` CLI** + `_pool_response_csvs` (~90): only a README usage
  example (`pymc_model_families/README.md:49`).
- **Registry**: `reserved_for_new` written (always 0.0) and never read. ~20.
- **`eig.py` CLI** (`Args`, `main`, `_write_output`, ~80): no script; tests drive it.
- **Sampling-warning thresholds**: `_warn_sampling_diagnostics` warns at any
  divergence / R-hat > 1.01 while the gate uses `MAX_*`; merging into
  `convergence_problems` changes log output only. ~45.
- **`scripts/sanitize_prompt_listings.py`** (100) + test (63): targets
  `paper-source/prompts`, not in the repo.
- **Root configs** `km_pilot.yaml`, `full_run_sr-oct26.yaml`: tracked,
  unreferenced; keep as run records?

## Tier 3: history-narrating comments and docstrings, ~1,500–2,500 lines

Dated rationale ("used to…", "until 2026-09-30…", incident write-ups) to trim to
current behaviour. Worst: sbatch headers (`holdout_recovery_array.sbatch`,
`slurm/_env.sh`, the `run_*_test_retest.sh` pairs; shell is ~37% comments),
`pymc_inference.py` (`_fit_process_caches`, compile-slot block,
`refit_settings`, `FitInfrastructureFailure`, `model_logp_is_finite`),
`model_zoo.py` (~20 markers), `eig.py:286-314`, `model_loop_runner`'s
`update_registry_from_interpretation` / `_export_inner_loop_models`,
`critique_round.py`, `holdout_recovery.py`, `coding_agent.py`, `ppc.py`,
`mcmc_defaults.py`. Wrong docstrings: `critique/__init__.py` ("no
multiple-comparisons correction" — BH is reported), `pymc_inference.py` module
("preprocessed responses CSV"), `model_families/__init__.py`,
`inner_loop/run.py:114` (`DEFAULT_CANDIDATE_HINTS` is no longer the default).

## Bugs found along the way (fix regardless)

- `results_collection.py:316` `summarize_model_posterior` picks the best model
  by rounded-posterior argmax — the rule CLAUDE.md says is wrong (use the best
  ELPD rank among reliable models). Affects `SUMMARY.md` only.
- `eig.py` `_write_output` would crash on a design with random picks (their
  `eig` is `None`).
