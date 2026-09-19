# Code tour

For a reader who knows the science (Bayesian model comparison, subjective
randomness) and has not seen the code. See `CLAUDE.md` for architecture rules,
conventions and the full mental model.

## The two loops in brief

**Outer loop** (`src/pipelines/outer_loop/`). One pass per experiment: design
stimuli via greedy max-EIG selection, optionally implement a jsPsych experiment,
collect responses (simulated or live), run the inner model loop, export the
surviving model set, and write a uniform-prior registry for the next experiment's
design. State crosses experiment boundaries through on-disk artifacts only:
carried model `.py` files, the `models_manifest.yaml`, the attempted-hypotheses
ledger, and `model_registry.yaml`.

**Inner loop** (`src/pipelines/inner_loop/`). The only place new hypotheses enter
the system. Seeds the model zoo from the previous experiment's carried set (or the
project's seed models for experiment 1). Then, for each round: optionally critique
the incumbent via posterior-predictive checks, spawn parallel candidate agents
(each steered by a rotating exploration "lens"), admit candidates that pass the
import gate → logp → MCMC-fit → finite-ELPD → novelty gates, score all survivors
by ELPD-LOO, prune non-protected losers that are statistically distinguishable
from the best, and export the live set. Every candidate slot and every prune is
recorded in the hypothesis ledger.

## On-disk artifacts that carry state

| Artifact | Written by | Read by | Format |
|---|---|---|---|
| `cognitive_models/` (model `.py` files + manifest + ledger) | outer-loop export | next experiment's seeding | `.py`, YAML, JSONL |
| `model_registry.yaml` | `update_registry_from_interpretation` | next experiment's EIG design | YAML (`{theories: {name: prob}}`) |
| `model_loop/responses.csv` | outer-loop orchestrator | inner loop | CSV (5 raw columns) |
| `model_loop/models/` | inner-loop seed + admission | inner-loop scoring, pruning, export | `.py` files + manifest |
| `attempted_hypotheses.jsonl` | hypothesis ledger | candidate briefs (as `attempted_hypotheses.md`) | JSONL |
| `design/stimuli.json` | EIG design | collect stage | JSON |
| `data/responses.csv` | collect stage | outer-loop orchestrator | CSV (5 raw columns) |
| `history.json` | inner-loop history recorder | holdout evaluation, viewer | JSON |
| `model_posterior.json` | model comparison | inner loop, viewer | JSON |

## Walk-through: one holdout cell

Starting from `scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh`:

1. **Setup job** (`holdout_setup.sbatch`): syncs the uv environment on scratch,
   stages a pristine snapshot of the ground-truth PyMC models and family twins
   (read-only for all array tasks), and stages a full harness repo copy.

2. **Array task** (`holdout_recovery_array.sbatch`): one per (repeat, ground truth).
   - Builds the **agent tree**: `rsync` with `agent_tree.exclude` strips feature
     code, research library modules, GT-recipe files, holdout configs, and build
     artifacts. Physically removes the held-out model's `.py` and manifest entry
     from both the registry (`pymc_model_families/`) and the live seed pool
     (`seed_models/`). Stubs the pure-Python family twin. Hardens opencode
     read/glob/grep denies.
   - Invokes `scripts/subjective_randomness/holdout_recovery.py` from the full
     **harness repo** (can import everything the agents must not see) with
     `--agent-root` pointing at the scrubbed tree.
   - Archives the agent's `_runs/` tree as `agent_runs.tar.gz` and deletes the
     repo copy to reclaim inodes.

3. **`holdout_recovery.py`** CLI → `src/subjective_randomness/holdout_recovery.py`:
   - Resolves config, validates seed-model binding, sets up the evaluation pool
     (exhaustive over lengths 1–8 minus trained pairs).
   - For each experiment (`n_experiments`, typically 4):
     - **Generate responses**: ground-truth model produces `chose_left` for each
       stimulus pair. Written as `data/responses.csv` (5 raw columns only; the
       `generating_model` tag is stripped before the agents see it).
     - **Design**: `run_design_programmatic` → `eig.design_exhaustive` → greedy
       max-EIG stimulus selection using the model registry as the prior.
     - **Inner loop**: `run_inner_model_loop_programmatic` → seeds, fits, compares,
       optionally spawns candidate agents, admits, prunes, exports. The carried
       set flows into the next experiment.
     - **Evaluate trajectory**: at each inner-loop step and after the final export,
       score the incumbent and the BMA on the held-out pool. Writes `holdout.json`
       with RMSE, KL regret, Pearson r, bias, calibration, per step.

4. **Analysis job** (`holdout_test_retest.py`): across all cells, computes
   test-retest summaries, writes `test_retest.json` and `test_retest.csv`.

5. **Verifier** (`verify_holdout_run.sh`): checks every agent-facing CSV carries
   only raw columns, every `CONTEXT.md` names only raw columns, no candidate
   imports the feature library, no `[drop]` lines, every `screened_out.json` is
   empty, the agent tree contains no forbidden paths, and all configured cells
   completed.

## Walk-through: one inner-loop round

Starting from `run_pymc_inner_loop` in `pymc_orchestrator.py`:

1. **Seed**: copy seed models into `model_loop/models/`, continue the carried
   ledger. Drop unfittable seeds (log warning, never silent).

2. **Score**: fit all models via `pymc_inference.fit_model` (cached on model
   sha × data sha × sampler signature), compute `model_posterior` (softmax
   ELPD + `az.compare` table), select `_best_exportable_model` by ELPD-LOO
   rank among PSIS-LOO-reliable models.

3. **Critique** (if enabled): write `CRITIQUE_CONTEXT.md` with the incumbent's
   hypothesis and the candidate response columns. Spawn a critique agent to
   propose `test_statistic(df) -> float` functions. Run the PPC harness
   (`critique/ppc.py`): score each statistic against posterior-predictive
   replicates, compute two-sided empirical p with BH-FDR correction. Write
   `critiques.md` for the next candidate round.

4. **Candidate round** (repeated `max_iterations` times):
   - Write `CANDIDATE_BRIEF.md` with the exploration lens, existing hypotheses
     (as an `elpd_diff ± dse` race, not softmax), attempted hypotheses (from
     the ledger), critique results, and `CONTEXT.md` with the raw column names.
   - Spawn `candidate_count` agents in parallel (each gets a distinct lens via
     `_lens_index`). Each writes `candidate.py`, `hypothesis.md`, `model_name.txt`.
   - **Admit sequentially**: `_admit_candidate` checks the import gate
     (`import_gate.py` — AST-based allowlist), loads the model, verifies logp is
     finite, runs a real MCMC fit, checks ELPD-LOO is finite, and checks novelty
     (posterior-mean RMSE ≥ 0.02 from every admitted model). Each outcome is
     recorded in the ledger.
   - **Score** all models again; **prune** non-protected losers
     (`elpd_diff > dse_multiplier × dse` among reliable rows).

5. **Export**: `_export` copies the surviving zoo (protected seeds + all
   non-pruned models) into `inner_loop_model/`, then the outer loop's
   `_export_inner_loop_models` builds `cognitive_models/` from it. The ledger
   is copied beside the manifest so the next experiment inherits it.

## Fail-loud checks and what each protects

| Check | Module | Protects against |
|---|---|---|
| `MissingStimulusColumns` (raises on missing feature columns) | `data_binding.py` | Models dropped from design because the data lacks their columns |
| `NON_STIMULUS_COLUMNS` screening (allows `participant_id`, `trial_index` drops only) | `eig.py` | Silent hypothesis-set shrinkage at design time |
| `screened_out.json` written even when empty | `eig.py` | Invisible drops — the verifier asserts `[]` for raw runs |
| Import gate (AST allowlist) | `import_gate.py` | Candidates importing feature code or other forbidden modules |
| Agent-tree isolation (`agent_tree.exclude` + rsync) | `holdout_recovery_array.sbatch` | Agents reading ground-truth recipes, feature library, or holdout configs |
| Manifest scrub (held-out model's name + rationale removed) | `remove_manifest_entry.py` | Agents seeing the held-out model's identity |
| `generating_model` column stripped | `holdout_recovery.py` | Response data leaking which model generated it |
| Leakage audit (`any_csv_generating_model`, `any_manifest_gt_named`) | `leakage_audit.py` | Held-out label present in any agent-visible artifact |
| PSIS-LOO reliability (tolerating exempt constant-logp trials) | `loo_reliability.py` | arviz's blanket k>0.7 flag discarding winners on clipped trials |
| Best-by-ELPD-rank export (not softmax argmax) | `scoring.py` | Exporting a far-behind model that reads 0.0 in the rounded posterior |
| Novelty RMSE gate (0.02) | `model_zoo.py` | Re-skinned duplicates wasting candidate slots |
| Config-time raw-column binding check | `holdout_recovery.py` | A seed model that cannot bind raw rows reaching design and failing there |

## Module reference

### Inner loop

| Module | Lines | Purpose |
|---|---|---|
| `src/pipelines/inner_loop/pymc_orchestrator.py` | 344 | Top-level orchestrator: seed → score → critique → candidate → prune loop |
| `src/pipelines/inner_loop/model_zoo.py` | 690 | Model zoo: seeding, admission gates, pruning, novelty check, lens rotation |
| `src/pipelines/inner_loop/candidate_agent.py` | 387 | Candidate brief generation and agent spawning |
| `src/pipelines/inner_loop/scoring.py` | 339 | ELPD-LOO scoring, best-model selection, history, export artifacts |
| `src/pipelines/inner_loop/hypothesis_ledger.py` | 185 | Append-only JSONL ledger of attempted hypotheses |
| `src/pipelines/inner_loop/run.py` | 171 | CLI entry point for the inner loop |
| `src/pipelines/inner_loop/import_gate.py` | 62 | AST-based import allowlist for candidate/critique code |

### Outer loop

| Module | Lines | Purpose |
|---|---|---|
| `src/pipelines/outer_loop/run.py` | 744 | CLI entry point for the outer loop |
| `src/pipelines/outer_loop/orchestrator.py` | 716 | Seeding, carry-forward, stage dispatch, context writing, agent spawning |
| `src/pipelines/outer_loop/collect.py` | 454 | Data collection: live Prolific, Firebase retrieval, QC |
| `src/pipelines/outer_loop/eig.py` | 405 | EIG-based design: screen models, enumerate stimuli, greedy selection |
| `src/pipelines/outer_loop/model_loop_runner.py` | 379 | Inner-loop integration: pool responses, protect seeds, export, registry |
| `src/pipelines/outer_loop/synthetic_data.py` | 352 | Synthetic data: LLM-as-participant and model-based response generation |
| `src/pipelines/outer_loop/orchestrator_validators.py` | 316 | Validators for outer-loop stage outputs |
| `src/pipelines/outer_loop/browser_steering.py` | 315 | Playwright browser automation for jsPsych experiments |
| `src/pipelines/outer_loop/participants.py` | 232 | LLM participant backends (closed API, open Hugging Face) |
| `src/pipelines/outer_loop/columns.py` | 36 | `RAW_RESPONSE_COLUMNS` constant and CSV writer |
| `src/pipelines/outer_loop/deployment/` | ~1560 | Firebase, Prolific, local, smoke deployment |

### Models and inference

| Module | Lines | Purpose |
|---|---|---|
| `src/models/pymc_inference.py` | 860 | PyMC fitting, caching, prior/posterior prediction, EIG, diagnostics |
| `src/models/data_binding.py` | 334 | CSV → `pm.set_data` dict; `compute_features` and `prepare_observed` hooks |
| `src/models/eig_selection.py` | 327 | Greedy max-EIG stimulus set selection |
| `src/models/model_loading.py` | 224 | Load agent-written `.py` models, attach hooks, per-process cache |
| `src/models/loo_reliability.py` | 165 | PSIS-LOO reliability with exempt-trial logic |
| `src/models/model_manifest.py` | 138 | `models_manifest.yaml` parser and writer |
| `src/models/probability.py` | 65 | Probability validation helpers |
| `src/models/mcmc_defaults.py` | 28 | Single source of MCMC sampler defaults |
| `src/model_comparison/posterior.py` | 353 | Bayesian model posterior via softmax ELPD + `az.compare` |
| `src/model_comparison/likelihood.py` | 113 | ELPD-LOO computation for a single model |
| `src/critique/ppc.py` | 578 | CriticAL posterior-predictive check |
| `src/registry/io.py` | 138 | `model_registry.yaml` I/O and validation |

### Runtime

| Module | Lines | Purpose |
|---|---|---|
| `src/runtime/coding_agent.py` | 561 | Backend-agnostic agent launcher (claude, opencode, codex) |
| `src/runtime/token_usage.py` | 189 | LLM token-usage accounting |
| `src/runtime/config.py` | 78 | `REPO_ROOT`, path resolution, secrets |

### Holdout recovery harness

| Module | Lines | Purpose |
|---|---|---|
| `src/subjective_randomness/holdout_recovery.py` | 693 | Holdout recovery orchestrator: generate data, run loops, dispatch eval |
| `src/subjective_randomness/holdout_eval.py` | 649 | Trajectory evaluation: fit, predict, score against held-out pool |
| `src/subjective_randomness/holdout_data.py` | 289 | Ground-truth data generation, parameter resolution, pool validation |
| `src/subjective_randomness/leakage_audit.py` | 215 | Audit agent-written models for ground-truth leakage |
| `src/subjective_randomness/recovery_metrics.py` | 65 | RMSE, KL regret, bias, calibration |
| `src/subjective_randomness/cell_archive.py` | 92 | Resolve a finished cell's run tree, extracting `agent_runs.tar.gz` when it is not on disk |
| `src/subjective_randomness/recovery_ceiling.py` | 265 | Recovery ceiling: refit the held-out ground truth on a cell's own data to bound achievable RMSE |
| `src/subjective_randomness/tidy.py` | 88 | `trajectory_tidy_rows` for holdout results |
| `src/subjective_randomness/config.py` | 37 | Shared config/path helpers |
| `src/pipelines/outer_loop/projects/subjective_randomness/evaluate_recovery.py` | 294 | Evaluation pool builder, `feature_rows` |
| `src/pipelines/outer_loop/projects/subjective_randomness/ground_truth_models.py` | 125 | Ground-truth model registry |
| `src/pipelines/outer_loop/projects/subjective_randomness/seed_models/` | ~820 | 4 seed models + archive models + manifest |
| `src/subjective_randomness/pymc_model_families/` | ~1460 | 11 PyMC model families + manifest (frozen recovery registry) |
| `src/subjective_randomness/model_families/` | ~1520 | 11 pure-Python model family implementations |
| `src/subjective_randomness/impossible_models/` | ~140 | 4 impossible-theory ground-truth generators |

### Holdout harness scripts and Slurm launchers

| Script | Purpose |
|---|---|
| `scripts/subjective_randomness/holdout_recovery.py` | CLI wrapper for holdout recovery |
| `scripts/subjective_randomness/holdout_test_retest.py` | Test-retest analysis of holdout results |
| `scripts/subjective_randomness/compare_matched_cells.py` | Paired comparison of matched-seed cells |
| `scripts/subjective_randomness/oracle_admitted_models.py` | Oracle-best diagnostic (**known broken**: scores 0 steps on archived cells) |
| `scripts/subjective_randomness/recovery_report.py` | Per-sweep RMSE tables (**known bug**: repeats one lost-incumbent total under every ground truth) |
| `scripts/subjective_randomness/recovery_ceiling.py` | Recovery-ceiling CLI over a finished sweep |
| `scripts/subjective_randomness/slurm/recovery_ceiling.sbatch` | Ceiling job; runs its end-to-end test as a gate first |
| `scripts/subjective_randomness/remove_manifest_entry.py` | Remove a model from manifest |
| `scripts/subjective_randomness/slurm/submit_holdout_test_retest.sh` | Slurm launcher: chains setup → array → analysis |
| `scripts/subjective_randomness/slurm/holdout_recovery_array.sbatch` | Array task: one (repeat, ground truth) cell |
| `scripts/subjective_randomness/slurm/holdout_setup.sbatch` | Setup: stage GT snapshots, sync venv |
| `scripts/subjective_randomness/slurm/verify_holdout_run.sh` | Post-run verifier (raw columns, isolation, completion) |
| `scripts/subjective_randomness/slurm/agent_tree.exclude` | rsync exclude list for agent-tree isolation |

### Research library (standalone, thin coupling to pipeline)

| Module | Lines | Purpose |
|---|---|---|
| `src/subjective_randomness/features.py` | 415 | Feature engineering (user-side analysis only) |
| `src/subjective_randomness/sequence_stats.py` | 434 | Sequence statistics (runs, motifs, etc.) |
| `src/subjective_randomness/stimulus_design.py` | 867 | Stimulus pool generation and analysis |
| `src/subjective_randomness/model_recovery.py` | 444 | Closed-ended model recovery |
| `src/subjective_randomness/adaptive_recovery.py` | 731 | Sequential Bayesian optimal design recovery |
| `src/subjective_randomness/reporting.py` | 1118 | Recovery reporting and figure generation |

### Support (viewer, monitor, campaign tooling)

| Module | Lines | Purpose |
|---|---|---|
| `src/viewer/` | ~1050 | Flask SPA run explorer and static-site freezer |
| `src/monitor/` | ~660 | Live dashboard for in-progress human studies |
| `src/recovery_improvement/` | ~2220 | Autonomous recovery-improvement campaign driver |
| `src/consolidation/driver.py` | 318 | Consolidation plan driver |
