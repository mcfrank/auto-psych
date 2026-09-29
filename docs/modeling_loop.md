# The modeling loop, step by step: one holdout-recovery cell

Describes the code at commit `555a6d0` (branch `consolidate/2026-09`).
`file:line` references were taken at `4de536c`; `555a6d0` fixed the two
crashes this document had found (the fit pool and a refit, the refinement
menu and a cap retirement), so lines in `pymc_inference.py` and `model_zoo.py`
after those spots are off by a few.

This is a reference for reviewing what the code does in one holdout-recovery
cell, from the Slurm array task to the recovery metrics. It was written by
reading the code. Where a docstring, `README.md` or `CLAUDE.md` says something
different, this document follows the code, and the difference is listed in the
last section.

Concrete values come from
`scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml` and the
defaults it does not override.
`docs/` is excluded from every agent tree (`agent_tree.exclude`), so this file
does not reach the agents.

---

## 0. Overview and call chain

```
submit_holdout_test_retest.sh   (submits setup → array → retry → analysis)
holdout_setup.sbatch            (once per sweep: stage harness_repo, agent_src, gt_models_src, gt_family_src; record code_commit)
holdout_recovery_array.sbatch   (one task = one (repeat, ground truth) cell)
 └─ scripts/subjective_randomness/holdout_recovery.py : main
     └─ src/subjective_randomness/holdout_recovery.py : run_holdout_recovery_from_config
         └─ _run_holdout_recovery_resolved            (one GT, because --gt-model is passed)
             ├─ run_holdout_experiments               (experiments 1..3)
             │   for each experiment:
             │   ├─ model set: seed_experiment_models_from_project | carry_forward_cognitive_models
             │   ├─ design:    orchestrator.run_design_programmatic → eig.design_exhaustive
             │   ├─ collect:   holdout_data.generate_responses (GT, fixed params, counterbalanced)
             │   └─ inner loop: model_loop_runner.run_inner_model_loop_programmatic
             │                  → inner_loop.pymc_orchestrator.run_pymc_inner_loop
             │                  → _export_inner_loop_models, update_registry_from_interpretation
             ├─ build_eval_stimuli, evaluate_trajectory (logs eval_exclusions.jsonl)
             ├─ annotate_incumbents / summarise_incumbents
             ├─ leakage_check
             ├─ seed_baseline_correlation, fitted_seed_baseline_correlation
             └─ write trajectory.json (outside the agent tree)
holdout_retry.sbatch            (after the array: resume failed tasks)
holdout_analysis.sbatch         (after the array: summary, warnings, MISSING_CELLS.txt)
```

The Implement stage (`3_implement`, a jsPsych agent) is not run in the holdout
harness. No theory or design agent exists. The only LLM agents are the inner
loop's critique agent and candidate agents.

### Effective settings for one cell (faithful config under the array)

| Setting | Value | Where it is set |
| --- | --- | --- |
| Ground truths (one per array task) | `falk_konold_dp`, `motif_stack`, `finite_experience_occurrence`, `local_representativeness` | config `gt_models`; sbatch `GTS` (sbatch:66) |
| GT parameters | family `DEFAULT_PARAMS` (config values are `null`) | `holdout_data.resolve_generating_params` |
| Experiments per cell | 3 | config `n_experiments` |
| Synthetic participants per experiment | 40 | config `n_participants` |
| Stimuli per experiment | 64, all chosen by EIG (40-response EIG until its noise floor, then single-response EIG fill); 0 random | config `design: {n_eig: 64, n_random: 0}` |
| Design pair lengths | 2..8, same-length pairs only | `run_design_programmatic` default `lengths` (orchestrator.py:430), `design_exhaustive` (eig.py:253) |
| Cell seed | `BASE_SEED + REPEAT` (default 0 + r) | sbatch:83, passed as `--seed`; **overrides config `seed: 7`** |
| Per-purpose seeds | `derive_seed(cell seed, GT, experiment, purpose)` | holdout_recovery.py:294 (§8) |
| Inner-loop rounds per experiment | 5 | config `inner_loop.max_iterations` |
| Candidate slots per round | 6 (3 explore, 2 refine incumbent, 1 refine chosen) | config `candidate_count`; `model_zoo.slot_roles` |
| Critique test statistics requested | 8 | config `n_critique_proposals` |
| Critique replicates | 1000 | `CRITIQUE_PPC_REPLICATES` (critique_round.py:49) |
| Novelty RMSE threshold | 0.002 | `DEFAULT_NOVELTY_RMSE_THRESHOLD` (model_zoo.py:571); not set in config |
| Pruning | once per experiment, at the end: `elpd_diff > 2.0 · dse_clustered` | `DEFAULT_PRUNE_DSE_MULTIPLIER` (model_zoo.py:560), `_prune_losers` (model_zoo.py:625) |
| Live-set cap | 8 models, seeds included | `MAX_LIVE_MODELS` (model_zoo.py:717) |
| Agent backend / model | `opencode` / `google/gemini-3.1-pro-preview` | sbatch `--backend ${AGENT_BACKEND:-opencode}` (sbatch:372); config `agent.model` unless `AGENT_MODEL` is set |
| Agent timeout | 1800 s per agent attempt | config `agent.timeout_sec` |
| Production MCMC | 2000 draws, 1000 tune, 4 chains, target_accept 0.8 (a model's declared value is a floor), max_treedepth 10, seed 42 | config `fit` + `_FIT_DEFAULTS` (pymc_inference.py:350), `resolve_fit_settings` (pymc_inference.py:421) |
| Convergence gate | ≤ 0.1% divergent transitions, R-hat ≤ 1.05, bulk ESS ≥ 100; one refit at target_accept 0.95 on failure | mcmc_defaults.py:39-44 |
| Eval pool | exhaustive same-length pairs, lengths 1..8, minus trained pairs; ≤500 posterior draws per prediction | config `eval_pool` |

---

## 1. Setup and holdout

### 1.1 Array task → (repeat, ground truth)

`holdout_recovery_array.sbatch:80-83`: with `G = 4` ground truths, task `T`
maps to `REPEAT = (T-1)/G + 1`, `GT = GTS[(T-1) % G]`, `SEED = BASE_SEED + REPEAT`.
All four ground truths of one repeat get the same cell seed; the seeds actually
used are derived from it together with the GT name (§8). Resources: 16 CPUs,
64 GB, 1 day, partition `normal` (sbatch:2-6).

**One job per cell** (sbatch:95-102, `cell_lock.sh`). The task writes its job
id to `RUN_DIR/.cell_lock`. If the lock names another job that `squeue` still
lists, the task exits 0 without touching the cell. A lock whose holder is no
longer queued is stale and is taken over. The lock is removed on exit.

**Automatic resume.** `submit_holdout_test_retest.sh` submits
`holdout_retry.sbatch` with `--dependency=afterany` on the array. It reads the
array's end states from `sacct` (`cell_status.py retry-plan`) and resubmits,
through the staged submit script, every task that ended `FAILED`, `TIMEOUT`,
`NODE_FAIL`, `PREEMPTED` or `BOOT_FAIL` (same memory) or `OUT_OF_MEMORY` (with
`--mem=128G` on that group's array only). Cancelled tasks are left alone, and
so is a task whose cell already has its `holdout.json`. The retry array keeps
the `%MAX_PARALLEL` cap. Each resubmission chains its own retry job, up to
`MAX_RETRY_ROUNDS` (default 2), and the resubmitted tasks run with `--resume`
(they always do). `holdout_analysis.sbatch` writes
`$WORK_ROOT/MISSING_CELLS.txt` listing every expected cell without a
`holdout.json`.

**One code for the whole sweep.** The setup job stages the code once:
`harness_repo` (which also holds the Slurm scripts every later job sources),
`agent_src` (the repo scrubbed with `agent_tree.exclude`, from which each cell
builds its agent tree) and the GT snapshots, then writes
`$WORK_ROOT/code_commit` (`code_commit.sh`: the commit, plus a hash of any
uncommitted changes). A retry skips the setup job and runs the staged scripts.
A later setup on the same `WORK_ROOT` re-stages nothing and fails if the
checkout's code differs. Each cell writes `RUN_DIR/code_commit` when it starts
and refuses to resume on other code (or, with earlier work and no record, at
all). A cell that already has `holdout.json` exits at once, so a resubmission
cannot overwrite its archive or its leak record; an existing
`agent_runs.tar.gz` is never overwritten.

Directories:

| Path | Contents | Seen by agents? |
| --- | --- | --- |
| `$WORK_ROOT/run<r>/<gt>/` (`RUN_DIR`) | `holdout.{json,csv,png}`, `trajectory.json`, `eval_exclusions.jsonl`, `mcmc_cache/`, `agent_runs.tar.gz`, `gt_name_mentions.txt` (only if any), `agent_activity.md`, `.cell_lock`, `repo` → the agent tree | no |
| `$AGENT_TREES_ROOT/<sha256(RUN_DIR)[:16]>/` (`AGENT_DIR`) | `repo/` (agent tree), `mcmc_cache` → `RUN_DIR/mcmc_cache`, `venv` → the shared venv, `.xdg/` | only `repo/` (see §1.5) |
| `AGENT_DIR/repo/_runs/cell_1/` | the run tree: `experiment1..3/`, `agent_notes/`, `eval_stimuli.json` | yes (read-only, except own dirs) |
| `$WORK_ROOT/harness_repo` | full repo copy staged once per sweep; the harness process and every job after setup run from here | no |
| `$WORK_ROOT/agent_src` | the repo scrubbed with `agent_tree.exclude`, staged once per sweep; each cell's agent tree is built from it | no |
| `$WORK_ROOT/code_commit`, `RUN_DIR/code_commit` | the staged code's identity; the code a cell started on | no |
| `$WORK_ROOT/gt_models_src`, `gt_family_src` | pristine copies of `pymc_model_families/` and `model_families/` | no |

The run directory is named `cell_<i>`, not after the ground truth
(holdout_recovery.py:565), because every path in the run tree appears in agent prompts.

### 1.2 Building the agent tree

In order (sbatch:134-318):

1. `rsync -a --delete --delete-excluded --filter='P /_runs/***' --exclude-from=agent_tree.exclude $WORK_ROOT/agent_src/ $RUN_REPO/`
   (sbatch:147-148). `--delete-excluded` also cleans a resumed cell's tree
   built before an exclusion was added; the filter protects the run tree. The
   exclude list drops `.git`, `.venv`, `data`, caches, `node_modules`, `*.nc`,
   `CLAUDE.md`, `AGENTS.md`, `ground_truth_models.py`, `evaluate_recovery.py`,
   `gt.txt`, two literature test files, `preprocess.py`, `.secrets`,
   `.secrets.example`, and the whole of `/docs/`, `/tests/`, `/scripts/`,
   `/analysis/`, `/diagrams/`, `/README.md`, `/SUMMARY.md`,
   `/HERO_RUN_DESIDERATA.md` and `/src/subjective_randomness/` (which contains
   `model_families/`, `pymc_model_families/`, `features.py`,
   `stimulus_design.py`, …). From each project directory it also drops
   `references/`, `instruction_literature.md`, `problem_definition.md` and
   `seed_models/archive_*/`.
2. Delete `<gt>.py` from the registry path and from the live seed pool
   `src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`
   (sbatch:157-160). The registry path is already absent.
3. Replace `model_families/<gt>.py` with a stub, but only if the file exists
   (sbatch:163-189). It never does, because step 1 excluded
   `src/subjective_randomness/`, so this step has no effect.
4. Write `opencode.json` read/glob/grep deny rules (seed_models, both family
   dirs, holdout configs, `ground_truth_models.py`, `evaluate_recovery.py`,
   `gt.txt`) (sbatch:204-227). Only opencode reads this file.
5. Remove the GT's entry from both manifests (`remove_manifest_entry.py`); a
   manifest directory absent from the tree (the registry) is skipped. Then
   remove it from every other `models_manifest.yaml` outside `_runs/` that
   mentions it (sbatch:239-270).
6. Delete every file whose name contains the GT's name, outside `_runs/`
   (sbatch:289-292).
7. Verify: both family directories are absent, and no `*<gt>*.py` remains.
   Otherwise exit 1 (sbatch:299-312).
8. **GT-name scan** (`scan_gt_name.sh --before-agents`, sbatch:317): `grep -rIl`
   for the GT name over every text file in the tree except `_runs/`. Any hit
   stops the cell (exit 1) before an agent runs.
9. Remove from the agents' environment every variable whose value contains the
   GT name, `$WORK_ROOT` or `$REPO`, plus `GT_MODELS` and `CONFIG`
   (sbatch:350-364). That is the harness's environment; each sandboxed agent
   then gets only an allowlist of it (`agent_sandbox.agent_environment`), so
   no `SLURM_*` variable (`SLURM_ARRAY_TASK_ID` maps to the GT), no retry
   variable and no `.secrets` key but its own backend's login reaches it.

The harness then runs from `harness_repo` with `--agent-root $RUN_REPO`,
`--results-root $RUN_REPO/_runs`, `--summary-root $WORK_ROOT/run<r>`,
`--cache-dir $AGENT_DIR/mcmc_cache`, `--gt-models-dir $WORK_ROOT/gt_models_src`,
`--gt-family-dir $WORK_ROOT/gt_family_src`, and `--resume` (sbatch:366-384).

After a successful run: `scan_gt_name.sh --after-run` over `_runs/` writes
`gt_name_mentions.txt` if an agent wrote the GT name (warn-only, sbatch:391),
and `agent_activity_report.py` writes `agent_activity.md` listing the URLs
agents fetched and any paths outside their own directory (warn-only,
sbatch:395-397). The analysis job collects both into
`GT_NAME_WARNINGS.txt` / `OUTSIDE_PATH_WARNINGS.txt`.

What the agent tree still contains: `src/` other than `src/subjective_randomness/`
(so `src/pipelines/`, `src/models/`, `src/runtime/`, …), minus the files above;
the three non-held-out seed models in `seed_models/` with their scrubbed
manifest; top-level files and directories the list does not name (e.g.
`pyproject.toml`, `functions/`, `templates/`, `opencode.json`); and the run tree.

### 1.3 The ground-truth generator and its parameters

- **Generator:** the GT's PyMC model file from the pristine snapshot
  `gt_models_src/<gt>.py`, a copy of `src/subjective_randomness/pymc_model_families/<gt>.py`
  (`run_holdout_experiments` uses `gt_models_dir`, holdout_recovery.py:172).
- **Parameters:** config `gt_models: {name: null}` means the pure-Python
  family's `DEFAULT_PARAMS`. `resolve_generating_params`
  (holdout_data.py:86) calls `_family_default_params` (holdout_data.py:72),
  which reads `gt_family_src/<gt>.py` by **parsing** the `DEFAULT_PARAMS`
  literal with `ast` (`_default_params_from_file`, holdout_data.py:54), without
  importing it. If no pristine directory is given, it imports
  `src.subjective_randomness.model_families.<gt>`.
- **Where the values live** (cited by line, not copied here):

  | GT | `DEFAULT_PARAMS` | Free parameters in the PyMC model |
  | --- | --- | --- |
  | `falk_konold_dp` | `model_families/falk_konold_dp.py:38` | `beta`, `side_bias` |
  | `finite_experience_occurrence` | `model_families/finite_experience_occurrence.py:43` | `beta`, `side_bias` |
  | `local_representativeness` | `model_families/local_representativeness.py:40` | `theta_alt`, `alt_weight`, `periodic_share`, `beta`, `side_bias` |
  | `motif_stack` | `model_families/motif_stack.py:43` | `delta`, `alpha`, `repetition_weight`, `mirror_share`, `complement_share`, `beta`, `side_bias` |

  `side_bias` is 0 in all four. `_require_exact_params` (holdout_data.py:132)
  fails unless the parameter names match the PyMC model's free RVs exactly.
- **GT choice probabilities:** `p_left_fixed_params` (holdout_data.py:145).
  It binds raw rows (`sequence_a`, `sequence_b`, dummy `chose_left=0`) through
  the model's own `compute_features`/`prepare_observed` hook, fixes every free
  RV with `pm.do(model, params)`, and draws one prior-predictive sample of the
  `p_left` Deterministic. That sample is deterministic.

### 1.4 The seed model set

Experiment 1 is seeded from the harness checkout's live pool
`src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`, which
the harness process sees unscrubbed. `seed_exclusion` (holdout_data.py:35)
withholds the GT by manifest name, so the three other literature models are
seeded. The pool manifest mirrors the registry manifest in
`pymc_model_families/models_manifest.yaml`, and a test
(`tests/test_model_manifest.py:76`) asserts the model files are byte-identical.

The same three models are the **protected** set: they are never pruned or
retired by the cap and are always carried forward (`_protected_seed_names`,
model_loop_runner.py:81). The protected set is the pool manifest intersected
with the experiment's `cognitive_models/`.

### 1.5 What agents can see: the sandbox

Every loop agent (critique and candidates) is launched with `sandbox=True`
(critique_round.py:338, candidate_agent.py:741). `run_coding_agent`
(coding_agent.py:603) wraps the CLI in bubblewrap through `sandbox_command`
(agent_sandbox.py:47):

| Mount | Mode | What |
| --- | --- | --- |
| `cwd` = the agent tree (`--agent-root`) | read-only | the whole scrubbed repo copy, including `_runs/cell_1/` (all experiments, other slots' dirs, the zoo) |
| `allowed_dirs` | read-only | candidate: its dir, `model_loop/models`, `model_loop/`; critique: its dir, the zoo, `model_loop/` (all already inside the tree) |
| `writable_dirs` + the agent's own dir + `memory_dir` | read-write | candidate: `candidate_<i>[_retry_1|_repair_1]/`; critique: `iter_<i>/critique/`; both: `_runs/cell_1/agent_notes/` |
| `<agent dir>/scratch` | read-write, mounted at `/tmp` | kept after the run |
| `<agent dir>/.home` | read-write, mounted at `$HOME` | deleted after the agent exits (`remove_private_home`) |
| `/usr`, `/etc`, `/share/software`, the venv and its base interpreter, the CLI install | read-only | system software |

The sandbox uses a private PID namespace (`--unshare-pid`), so the harness's
`ps` arguments are not visible. The network is shared. The environment is an
allowlist (`agent_environment`: `AGENT_ENV_NAMES` — system basics, locale,
XDG, the compiler toolchain PyTensor needs, thread caps, network/TLS settings
— plus the backend's own login and configuration, `BACKEND_ENV`: provider
keys and `OPENCODE_*` for opencode, `CLAUDE_CODE_*`/`ANTHROPIC_*` for claude,
`CODEX_*`/`OPENAI_API_KEY` for codex). Every other `.secrets` key (Prolific,
Firebase, the results token), every `SLURM_*` variable and the sweep's own
variables are withheld; `_env.sh` still exports `.secrets` into the harness.
Logins: claude needs `CLAUDE_CODE_OAUTH_TOKEN`, codex gets
a private `CODEX_HOME` holding only `auth.json`, and opencode reads its
provider key from the environment (`_login`, agent_sandbox.py:217). For
opencode, `external_directory` is set to `allow` because the sandbox itself
does the confining (agent_sandbox.py:123-131).

The MCMC cache (`$AGENT_DIR/mcmc_cache`) is outside the agent tree, so agents
cannot see it. Its path does appear in the critique context text.

`stock=True` makes a Claude agent run without the user's configuration.
`memory_dir` is the run's `agent_notes/`, shared by every agent of this cell
across rounds and experiments (it is used as an auto-memory directory by the
claude backend only).

Agent-written Python (candidates, critique statistics) runs **inside the
harness process**, not the sandbox, so the code gate (§5.9) forbids file reads
and interpreter escapes as well as imports outside the allowlist.

---

## 2. Model set per experiment

`run_holdout_experiments` (holdout_recovery.py:117):

- **Experiment 1:** `ensure_experiment_dirs` creates `cognitive_models/`,
  `design/`, `experiment/`, `data/`, `model_loop/`. `init_registry` writes an
  empty `model_registry.yaml` (model_loop_runner.py:354).
  `seed_experiment_models_from_project` (orchestrator.py:82) copies the kept
  seed files and writes a manifest without the excluded entry. It is a no-op
  if a manifest already exists, which is what makes `--resume` safe.
- **Experiments ≥ 2:** `carry_forward_cognitive_models` (orchestrator.py:147)
  copies the previous experiment's `cognitive_models/*.py`, its manifest and
  `attempted_hypotheses.jsonl`. It raises on a missing file.
- After either path, the `"models"` validator loads every model.

**Registry (the design prior).** After each inner loop,
`update_registry_from_interpretation` (model_loop_runner.py:364) writes
`model_registry.yaml` as `{theories: {name: 1/n for every model in cognitive_models/}, reserved_for_new: 0.0}`.
Experiment k+1's design reads experiment k's registry. The carried set is
copied verbatim into experiment k+1's `cognitive_models/`, so the design prior
is always uniform over exactly the models it scores. In experiment 1 no
registry is passed, and the prior is also uniform. The `az.compare` stacking
weights appear only in `model_posterior.json`.

---

## 3. Stimulus design

Entry: `run_design_programmatic` (orchestrator.py:422) →
`eig.design_exhaustive` (eig.py:188) with `n_select = 64` (config `n_eig`),
`n_random = 0`, `lengths = (2,3,4,5,6,7,8)`, `n_samples = 200`,
`n_scenarios = 1000`, `n_responses = 40` (the participant count), and
`seed = random_seed = derive_seed(cell seed, GT, exp_num, "design")`
(holdout_recovery.py:226-231). Output: `design/stimuli.json` and
`design/screened_out.json`.

### 3.1 Candidate pair space

`enumerate_all_pairs(lengths, same_length_only=True)`
(stimulus_design.py:114, called at eig.py:253):

- All `2^L` H/T strings for each L in 2..8, pooled in order (L ascending,
  then `itertools.product("HT", repeat=L)` order).
- Every **unordered** pair of two **distinct** strings
  (`itertools.combinations`), kept only if both have the **same length**.
- Each pair appears once, with `sequence_a` being the string that comes first
  in enumeration order. For equal lengths this is the lexicographically
  smaller string with H < T. This is the orientation written to
  `stimuli.json` and the one the EIG scores; data collection then
  counterbalances left/right per participant and trial (§4).
- Size: Σ_{L=2..8} C(2^L, 2) = 6 + 28 + 120 + 496 + 2,016 + 8,128 + 32,640 =
  **43,434 pairs**. Length-8 pairs are 75% of the pool.

### 3.2 Screening (`_screen_usable_models`, eig.py:62)

Each model named in `cognitive_models/models_manifest.yaml` that has a `.py`
file is probed with `make_stim_data(model, [rows[0]])`. The probe row is the
first pool pair (`HH` vs `HT`) with only `sequence_a`, `sequence_b` and
`chose_left=0`. Outcomes:

- `BROKEN_MODEL_CODE_ERRORS` (ImportError, SyntaxError, NameError,
  AttributeError, IndentationError): **raise**.
- `MissingStimulusColumns` where the missing columns are only
  `participant_id`/`trial_index` (`NON_STIMULUS_COLUMNS`, data_binding.py:244):
  the model is **dropped** from the design and recorded.
- `MissingStimulusColumns` naming any other column: **raise**.
- Any other exception: the model is **dropped** and recorded with the error
  (eig.py:115-119).
- No usable model: raise.

After the predictive draws (§3.3), a model whose `p_left` is undefined (NaN
or outside [0, 1]) on any pool pair (`InvalidPredictions`, prior or
posterior) is also left out of this design, printed as `[screen] EIG: …`
and recorded with `invalid_pairs` (the number of affected pairs) and a
reason naming up to five of them. If no model is left, the design raises.
Such a model used to crash the design, identically on every retry.

`design/screened_out.json` is always written as a list of
`{model, missing, reason}` (plus `invalid_pairs` for an undefined `p_left`);
it is empty when nothing was dropped. The model
prior is renormalised over the surviving models (§3.4).

### 3.3 Predictive draws: which distribution, how many

For every usable model m, the design builds an array `p[m]` of shape
`(D_m, N)` with `N = 43,434`: `p[m][d, j]` is `p_left` for pair j (in its
enumeration orientation) under parameter draw d.

- **Experiment 1: prior predictive.** `prior_predict_p_left_draws`
  (pymc_inference.py:185) binds all 43,434 rows at once and runs
  `pm.sample_prior_predictive(draws=200, var_names=["p_left"], random_seed=<design seed>)`.
  D_m = 200 draws from each model's **prior** over its parameters.
- **Experiments k ≥ 2: posterior predictive given all data so far.**
  `_posterior_p_left_draws` (eig.py:137) calls `fit_model` for each model, one
  at a time, on `experiment{k-1}/model_loop/responses.csv`, which is the
  cumulative file holding experiments 1..k−1 (2,560 · (k−1) rows). Settings:
  `DESIGN_TWIN_DRAWS = 500`, `DESIGN_TWIN_TUNE = 500`, `DESIGN_TWIN_CHAINS = 2`
  (mcmc_defaults.py:26-28), and `target_accept` = the model's own
  `SAMPLER_SETTINGS` value if it declares one, else
  `DESIGN_TWIN_TARGET_ACCEPT = 0.9` (mcmc_defaults.py:32, eig.py:177-179). The
  config's `fit` block does not reach the design. Other settings: `cores` 4
  (2 chain processes), `random_seed` 42, max_treedepth 10. A fit that fails
  the convergence gate (§5.3) is refit once at target_accept 0.95 and that fit
  is used; if the refit also fails, the model is still scored (the design does
  not check the gate itself). The posterior is thinned by
  `_thin_posterior(max_draws=200)` (pymc_inference.py:466) to 100 evenly
  spaced draws per chain, and `pm.sample_posterior_predictive` of `p_left`
  (`predict_p_left_draws`, pymc_inference.py:510, seed = design seed) gives
  D_m = 200.
- `p_left` is deterministic given the parameters, so each "draw" is one
  parameter vector's choice probabilities. Parameter uncertainty enters **only**
  through the D_m draws, as a within-model mixture.
- **Participant structure.** The EIG treats every selected stimulus as
  answered by 40 respondents who share one parameter vector, so it scores the
  count of "left" choices out of 40 (§3.5). This matches data generation (no
  individual differences, §4), except that the EIG assumes every response is
  to the enumeration orientation, whereas the data counterbalance left/right.
  The two agree when a model's `p_left(b, a) = 1 − p_left(a, b)`, which holds
  at `side_bias = 0` but not for prior or posterior draws with a nonzero
  `side_bias`.
- Probabilities are clipped to [1e-12, 1 − 1e-12] (`_P_CLIP`, eig_selection.py:51).

### 3.4 Model prior π

- Experiment 1: `registry_path=None`, so `model_weights = {}` and π is uniform.
- Experiment k ≥ 2: `_load_model_weights(experiment{k-1}/model_registry.yaml)`
  returns the uniform weights over the carried set (§2). `_model_prior`
  (eig_selection.py:90) looks up each usable model's weight and normalises.
  If the total is 0 it falls back to uniform, and eig.py:274-279 logs a message.
  In practice π is uniform over the usable models in every experiment.

### 3.5 What EIG is

The objective is the mutual information between model identity M and the
joint vector of response counts for the selected set S:

```
I(M; K_S) = H(π) − E_{K_S}[ H(M | K_S) ]          (bits)
```

under the generative process (n = 40 = `n_responses`)

```
m ~ π,   d ~ Uniform{1..D_m},   K_j | m, d ~ Binomial(n, p[m][d, j])  independently for j ∈ S,
```

so the within-model predictive is the mixture over draws:

```
P(k_S | m) ∝ (1/D_m) Σ_d  Π_{j∈S} p[m][d,j]^{k_j} (1 − p[m][d,j])^{n−k_j}
P(m | k_S) ∝ π(m) · P(k_S | m)
```

The binomial coefficients are the same for every model and draw, so they
cancel from the posterior and are left out of the likelihoods
(eig_selection.py module docstring). It is **joint** over S: each draw's
likelihood is a product over stimuli before averaging over draws, so the
correlation between stimuli induced by shared parameters is kept, and a
near-duplicate of a selected stimulus adds little.

**Monte Carlo estimator** (`_ScenarioState`, eig_selection.py:120). With
T = 1,000 scenarios and `rng = np.random.default_rng(seed)`:

1. For each scenario t, sample once and fix `m_t ~ π` and
   `d_t ~ Uniform{0..D_{m_t}−1}` (eig_selection.py:141-143).
2. Keep `logL[m][t, d] = Σ_{i∈S} [ k_{t,i} log p[m][d,i] + (n − k_{t,i}) log(1 − p[m][d,i]) ]`
   for every model and draw. It starts at 0.
3. Scenario t's posterior over models: `w_t(m) ∝ π(m) · mean_d exp(logL[m][t,d] − c_t)`.
   `c_t` is a per-scenario max, which cancels on normalisation
   (`posterior_entropy`, eig_selection.py:172). `H_t(S)` is its entropy in bits.
4. In-sample joint EIG: `Î(S) = H(π) − (1/T) Σ_t H_t(S)`.

The scenario's own draw `d_t` is one of the D_m draws in the likelihood
average, so the trajectory is an in-sample estimate. `estimate_joint_eig` with
a fresh seed would give an out-of-sample estimate, but the pipeline does not
call it. The ceiling on `Î(S)` is log2 K for K usable models (1.585 bits for
the three seeds of experiment 1).

### 3.6 Greedy joint selection with a noise-floor stop (`select_n_joint_eig`, eig_selection.py:302)

First call (eig.py:310-318): `n_select = 64`, `n_responses = 40`,
`stop_below_noise=True`, exact greedy (`lazy=False`, `chunk_size=4096`):

```
S ← ∅;  H_t ← H(π) for all t
repeat up to 64 times:
  for every pool index j (in chunks of 4096):
      q_{t,j} = p[m_t][d_t, j]                                  # the scenario's true P(left)
      for k = 0..40:
          w_t^{(k)}(m) ∝ π(m) · (1/D_m) Σ_d exp(logL[m][t,d] − c_t) · p[m][d,j]^k (1 − p[m][d,j])^{40−k}
          H_t^{(k)}(j) = entropy(w_t^{(k)})
      E_t(j) = Σ_k C(40,k) q_{t,j}^k (1 − q_{t,j})^{40−k} H_t^{(k)}(j)
      gain(j) = mean_t H_t − mean_t E_t(j)
  gain[S] ← −∞
  j* ← argmax_j gain(j)                                          # np.argmax: first maximum
  g_t = H_t − E_t(j*)                                            # per-scenario gain of j*
  if mean_t g_t ≤ max(2 · sd(g)/√T, 1e-6):  stop                # the noise floor
  for every scenario t: draw k_{t,j*} = #{40 uniforms < q_{t,j*}} with the same rng; logL += …
  S ← S ∪ {j*};  record joint_eig_bits = H(π) − mean_t H_t(S)
```

`next_entropy` (eig_selection.py:203) computes `E_t(j)`: for each outcome k,
one matrix product per model, `lhat[m] @ (p^k (1−p)^{n−k}) / D_m`, gives each
candidate's marginal likelihood without a (T, D, N) tensor. Each candidate's
outcome is marginalised analytically using the scenario's true probability,
while earlier outcomes are the sampled counts held in `logL`. `observe`
(eig_selection.py:240) samples the count as 40 uniform draws rather than
`rng.binomial`, so that with one response the random stream is the
single-response one.

**Noise-floor stop** (eig_selection.py:391-396). Before adding the best
candidate j*, the code recomputes its per-scenario gains `g_t` (T = 1000
values; their mean is `gain(j*)`) and stops if the mean is at most twice its
Monte Carlo standard error (`sd` with ddof=1, divided by √T), or at most
`NEGLIGIBLE_GAIN_BITS = 1e-6` (eig_selection.py:56). The check runs at every
step, including the first, so the 40-response selection can return anywhere
from 0 to 64 picks. With 40 responses per stimulus, a few picks can drive the
posterior entropy of most scenarios near 0, after which every remaining gain
is noise.

**Single-response fill** (eig.py:320-339). If the first call returned fewer
than 64 picks, a second `select_n_joint_eig` fills the remaining
`64 − n_picks` slots with `n_responses = 1`, `seed = design seed + 1`,
`preselected = the first call's picks`, and no stop rule:

- It builds **new** scenarios (new `m_t`, `d_t` from `seed + 1`).
- It first `observe`s every preselected stimulus in every scenario as **one**
  Bernoulli response (not 40), then runs exact greedy with single-response
  gains, `gain[S ∪ preselected] ← −∞`, for exactly `64 − n_picks` steps.
- So the fill picks are conditioned on the preselected picks, but in
  single-response units: its objective is the information of one more
  response per stimulus given one response to each earlier pick.
- The fill's `joint_eig_bits` are `H(π) − mean_t H_t` after the preselected
  single responses plus the fill picks so far, i.e. in single-response units
  and including the preselected stimuli's contribution. They are not
  comparable with the first call's 40-response values.
- A log line reports how many picks each objective made.

**Tie-breaking.** `np.argmax` returns the lowest pool index among equal gains.
Pool order is length ascending, then enumeration order, so an all-zero-gain
fill step picks the shortest, earliest-enumerated pairs. There is no other tie
rule. Given the draws, selection is deterministic in the design seed.

**Per-stimulus fields written to `stimuli.json`** (eig.py:341-351):

- `eig`: the marginal **single-response** EIG from draw-averaged means,
  `eig_from_prior_means` (pymc_inference.py:262). With `p̄_m = mean_d p[m][d,j]`,
  `p̄ = Σ_m π(m) p̄_m` and
  `EIG_j = H(π) − [p̄ H(M|R=1) + (1−p̄) H(M|R=0)]`. It is a report field,
  **not** used for selection, and it ignores within-model parameter uncertainty.
- `selection_rank`: 1..64, the 40-response picks in greedy order, then the
  fill picks in greedy order.
- `joint_eig_bits`: the selecting call's in-sample `Î` after this pick (units
  as above).
- `source`: `"eig"` for a pick of the 40-response selection,
  `"eig_single_response_fill"` for a fill pick, `"random"` for the random
  part (§3.7).

### 3.7 The random part (not used by the faithful config)

eig.py:363-380. With `n_random > 0`, `remaining` is every pool index not
chosen by EIG, and `random.Random(<design seed>).sample(remaining, n_random)`,
sorted by pool index, is appended with `eig: null`, `joint_eig_bits: null`,
`source: "random"` and `selection_rank` continuing after the EIG picks. The
faithful config sets `n_random: 0`, so every stimulus is EIG-chosen. Ablations
(config comment): `{n_eig: 32, n_random: 32}` (the earlier split) and
`{n_eig: 0, n_random: 64}` (pure random; screening and scoring are skipped).

No part of the design excludes stimuli used in earlier experiments. A pair can
recur across experiments and then gets fresh responses.

### 3.8 Order in `stimuli.json`

The 40-response EIG picks in greedy order, then the fill picks in greedy
order (then any random picks in pool order). This order becomes
`trial_index` (§4), so every participant sees the stimuli in the same order.

### 3.9 Caching

- Prior-predictive draws are recomputed every time; they are not cached.
- Design-time posterior fits (k ≥ 2) are cached on disk in
  `experiment{k}/design/_fit_cache/<name>.<fingerprint>.nc` and in process
  (`_FIT_CACHE`). The key is (model file sha256, responses file sha256,
  resolved sampler settings), pymc_inference.py:665-702; an escalated refit has
  its own fingerprint. This cache is separate from the cell's shared
  `mcmc_cache`.
- `load_pymc_model_cached` caches model loading for the screen.

---

## 4. Data collection (simulated from the ground truth)

`run_holdout_experiments` (holdout_recovery.py:236-254) →
`generate_responses` (holdout_data.py:175), with
`seed = derive_seed(cell seed, GT, exp_num, "responses")` and
`participant_id_offset = (exp_num − 1) · 40`:

1. `p = p_left_fixed_params(gt, gt_models_src, stimuli, DEFAULT_PARAMS, seed)`
   gives one deterministic `p_left` per stimulus in its designed orientation,
   and `p_swapped` the same for every pair with `sequence_a`/`sequence_b`
   swapped (holdout_data.py:200-206).
2. `rng = np.random.default_rng(seed)`.
3. For each participant 0..39: `swap = rng.random(64) < 0.5` (a fair coin per
   trial), `p_shown = where(swap, p_swapped, p)`, then
   `chose_left = rng.random(64) < p_shown`. Each row records the pair **as
   displayed** (swapped or not) and whether the left one was chosen.
   - **No individual differences.** Every participant has the same parameters.
   - **No lapse or noise term** beyond the model's own `p_left`.
   - Participants and trials are independent.
   - **Trial order** is the `stimuli.json` order for everyone.
4. The `generating_model` column is removed (`strip_generating_model`,
   holdout_data.py:229) before writing `experiment{k}/data/responses.csv` with
   `write_responses_csv`. `_require_no_generating_model_column` rechecks this
   on every path.

Columns written: `sequence_a, sequence_b, participant_id, trial_index,
chose_left` (`RAW_RESPONSE_COLUMNS`, columns.py:14). Each experiment writes
40 × 64 = 2,560 rows. `participant_id` runs 0..39 in experiment 1, 40..79 in
experiment 2 and 80..119 in experiment 3, so ids are unique after pooling.
`trial_index` runs 0..63 in every experiment.

---

## 5. Inner loop (one experiment)

Entry: `run_inner_model_loop_programmatic` (model_loop_runner.py:239) →
`run_pymc_inner_loop` (pymc_orchestrator.py:162). Artifacts go under
`experiment{k}/model_loop/`.

### 5.1 Data and task description

`_pooled_response_rows` (model_loop_runner.py:39) concatenates
`experiment1..k/data/responses.csv` into `model_loop/responses.csv`. That is
2,560, 5,120 and 7,680 rows for k = 1, 2, 3. Every fit, score, critique and
novelty check in experiment k uses this pooled file.

`write_task_description` (model_loop_runner.py:51) copies the project's
`task_description.md` into `model_loop/`; it raises if the project has none.
Its text (what a trial is, the instructions participants read, that left/right
is randomised, and what each column means, including that `participant_id` is
unique across experiments) opens every candidate's `CONTEXT.md` and the
critique's `CRITIQUE_CONTEXT.md` (`read_task_description`). The agents do not
receive `problem_definition.md`, which is also excluded from the agent tree.

### 5.2 Start of experiment

1. `_seed_model_set` (model_zoo.py:235) copies `cognitive_models/` (the carried
   or seeded set) into the zoo `model_loop/models/`.
2. `_resolve_protected_names` (scoring.py:32) sets the protected set to the
   project seeds present, which is the three non-GT seeds.
3. `HypothesisLedger.create` (hypothesis_ledger.py:102) copies
   `cognitive_models/attempted_hypotheses.jsonl` if present, and otherwise
   starts empty.
4. `_drop_unfittable_models` (model_zoo.py:284) runs `model_logp_is_finite`
   (pymc_inference.py:83). This checks that responses bind, and that the
   initial-point logp and its gradient are finite. Failures are dropped and
   recorded in the ledger as `dropped`. The step raises only if nothing survives.
5. `_drop_nonfinite_elpd_models` (model_zoo.py:325) is the experiment's first
   MCMC pass. `fit_models_to_cache` fits the whole set concurrently. Models
   whose fit fails or whose ELPD-LOO is non-finite are dropped (`dropped`),
   except a protected seed: that raises. Only the model's own failure drops
   it; an infrastructure failure (a broken fit pool, an unreadable `.nc`,
   `OSError`, `MemoryError`) raises and fails the cell, to be resumed.
   Models that fail the convergence gate are **not** dropped here.
6. If the threshold is > 0, `novelty_pool_rows()` generates the novelty pool
   and writes `model_loop/novelty_pool.json` (§5.9).
7. Seed scoring step: `_score`, then `_compare`, then `_record_history_step`
   with `iteration=None` (pymc_orchestrator.py:314-321). This is step 0 of
   `history.json`. `_record_history_step` calls `_best_exportable_model`, which
   raises if no model is both PSIS-reliable and converged (§5.5); the cell then
   fails at this step.

### 5.3 Fitting

`fit_model` (pymc_inference.py:709). Settings resolve in this order: explicit
caller value, then the model file's `SAMPLER_SETTINGS`, then `_FIT_DEFAULTS`
(`resolve_fit_settings`, pymc_inference.py:421), except that a model's
declared `target_accept` is a **floor** on the caller's (pymc_inference.py:449-456).

| Setting | Value in the loop | Source |
| --- | --- | --- |
| draws / tune | 2000 / 1000 | config `fit` (defaults would be 4000/3000, mcmc_defaults.py:13-14) |
| chains | 4 | config, and sbatch `--chains ${CHAINS:-4}` |
| target_accept | 0.8; 0.9 for `motif_stack` (it declares 0.9, which is a floor over the config's 0.8); 0.95 on an escalated refit | config; `SAMPLER_SETTINGS`; `ESCALATED_TARGET_ACCEPT` |
| max_treedepth | 10 | `_FIT_DEFAULTS` |
| cores | 4 | `PRODUCTION_CORES` |
| random_seed | 42 | `_FIT_DEFAULTS`: the same seed for every fit in every cell |
| log-likelihood | stored (`idata_kwargs={"log_likelihood": True}`) | needed for LOO |

That gives 8,000 posterior draws per fit.

**Convergence gate** (`convergence_problems`, pymc_inference.py:853, over the
model's free RVs). A fit has not converged if any of these holds:

- the trace records no `diverging` statistic;
- divergent transitions > `MAX_DIVERGENCE_FRACTION = 0.001` of all draws
  (more than 8 of 8,000);
- max R-hat > `MAX_R_HAT = 1.05`, or R-hat is undefined (e.g. one chain);
- min bulk ESS < `MIN_BULK_ESS = 100`.

**Escalation** (pymc_inference.py:756-768). If the fit has ≥ 2 chains, its
`target_accept` is below 0.95, and it fails the gate, `fit_model` refits once
at `target_accept = 0.95` and returns that fit whether or not it passes. Both
fits are written to the disk cache under their own fingerprints; a later
`fit_model` call with the original settings loads the first, finds it failing
and loads the second, so every caller gets the escalated fit. A one-chain fit
(the candidate self-check) is never refit.

The gate is used by admission (§5.9), best-model selection and export (§5.5),
pruning and the live-set cap (§5.11), and the fitted-seed baseline (§7.3).
Separately, divergences (any) and R-hat > 1.01 are printed as warnings on every
fit and cache hit; those warnings are advisory.

**Concurrency** (`_fit_outcomes`, pymc_inference.py:1117). Models that need
sampling (at least 2 of them) run in a spawned `ProcessPoolExecutor` with
`allocated_cpus() // min(cores, chains)` workers. That is 16 // 4 = 4
concurrent fits. Each worker pins BLAS to 1 thread, calls `fit_model` (so it
can escalate), writes the `.nc` (to a temporary name, then `os.replace`, so a
killed write leaves no truncated file), and returns the fit's fingerprint; the parent
checks it against the fingerprints it expects (at the loop's settings, or at
the 0.95 refit's) and loads the file. Candidate admission fits run one at a
time.

**Caching.** The in-process key is `(name, sha256(model.py), sha256(csv), sampler signature)`
of the requested settings. The on-disk file is `<cache_dir>/<name>.<fp>.nc`,
where `fp` is the first 16 hex characters of sha256(model sha ‖ csv sha ‖
signature of the resolved settings). The cell's `cache_dir` is
`$AGENT_DIR/mcmc_cache` → `RUN_DIR/mcmc_cache`. A new pooled CSV in each
experiment means every carried model is refit on it.

### 5.4 Scoring

- **ELPD-LOO:** `FittedModel.loo_diagnostics` → `loo_diagnostics`
  (loo_reliability.py:92) → `az.loo(idata, pointwise=True)` on the per-trial
  Bernoulli log-likelihood. It is computed once per fit.
- **Softmax "posterior"** (`model_posterior`, posterior.py:231):
  `score_m = elpd_m + c · lines_m`, with `c = DEFAULT_COMPLEXITY_PRIOR_CONST = −0.05`
  (scoring.py:29) and `lines_m` the number of non-blank, non-comment lines in
  the model file. Then `posterior_m = softmax(score)`, rounded to 6 decimals.
  It raises on a non-finite ELPD. It is used only as a report field and as the
  BMA weights in evaluation (§7). It does **not** select the best model.
- **Comparison table** (`compare_table`, posterior.py:120): `az.compare` on the
  precomputed `ELPDData` (ic="loo", default stacking weights). Per model it
  records `rank`, `elpd_loo`, `elpd_diff` and `dse` (both relative to rank 0),
  `dse_clustered`, `weight`, `loo_unreliable`, `n_bad_k`, `frac_bad_k`,
  `n_exact_loo_points`, `max_pareto_k`, `convergence_problems` (list of
  reasons) and `not_converged`.
- **Stimulus-clustered dse** (`src/models/clustered_se.py`). Each pooled
  response row is assigned to its stimulus, the **unordered** pair
  (`sorted((sequence_a, sequence_b))`, so both displayed orientations are one
  stimulus). For a model m ≠ rank 0, the pointwise differences
  `loo_i(best) − loo_i(m)` are summed within each stimulus, and
  `dse_clustered = sqrt(G · var(sums))` (numpy `var`, ddof=0) over the G
  distinct stimuli in the pooled data. It is 0 for the rank-0 model. `az.compare`'s
  `dse` is the same formula over trials.

**PSIS reliability** (loo_reliability.py):

- A trial is *exact* if its log-likelihood spread across all draws is ≤ 1e-8
  (`EXACT_TRIAL_LOGLIK_SPREAD`). A clipped, saturated `p_left` produces this.
  Exact trials are exempt.
- A trial is *bad* if it is not exact and not `k ≤ good_k`, so an infinite or
  NaN k counts as bad. `good_k` is arviz's value from `az.loo`. With 8,000
  draws this should be 0.7, but I read that from arviz's formula rather than
  checking the installed version.
- The model is *unreliable* if `n_bad / n_points > 0.01`
  (`DEFAULT_BAD_K_TOLERANCE`). The denominator counts all trials, exact ones
  included.

### 5.5 Best model per step

`_best_exportable_model` (scoring.py:74): among models that are neither
`loo_unreliable` nor `not_converged`, take the lowest `az.compare` rank, which
is the highest raw ELPD-LOO, with no complexity prior. It raises if no model
qualifies, or if the table is inconsistent. This one rule defines `best_model`
in `history.json`, the critique's incumbent, the incumbent-refinement target
and the exported winner.

### 5.6 A round (5 per experiment)

For `iteration` in 0..4 (pymc_orchestrator.py:324-585):

1. **Critique** of the current incumbent (§5.7). It runs before the
   candidates, sequentially.
2. **Slots.** `incumbent = history[-1]["best_model"]`. `slot_roles(6)`
   (model_zoo.py:101) gives `[explore, explore, explore, refine incumbent, refine incumbent, refine chosen]`.
3. **Lenses** (exploratory slots only). `DEFAULT_CANDIDATE_HINTS` has 12
   lenses (candidate_agent.py:51). The lens index is
   `(lens_offset + iteration·3 + e) % 12` for exploratory slot e ∈ {0,1,2}
   (`_lens_index`, model_zoo.py:141), with
   `lens_offset = (exp − 1) · 5 · 3` (`_lens_offset`, model_zoo.py:129).
   Experiment 1 therefore walks lenses 0-2, 3-5, 6-8, 9-11, 0-2. Experiment 2
   starts at lens 3 (15 mod 12), and experiment 3 starts at lens 6. No lens
   repeats within a round. Round 4 of each experiment repeats round 0's lenses.
4. **Context** for each slot (`_write_candidate_context`,
   candidate_agent.py:374). It is written to files and also **inlined into the
   prompt** (`_build_candidate_prompt`, candidate_agent.py:651). The prompt is
   `prompts/pymc_theory.md`, then the output instructions (absolute paths,
   bash heredocs only), then these sections:

   | Section | Explore slot | Refine-incumbent slot | Refine-chosen slot |
   | --- | --- | --- | --- |
   | `ATTEMPT_NOTE.md` (retry/repair only) | yes | yes | yes |
   | `CONTEXT.md`: the task description (§5.1), responses path and columns, the note that no feature columns exist so `compute_features`/`prepare_observed` is required, the import allowlist and the ban on file reads and interpreter escapes, the 3-step instruction, the `check_candidate` command with a note that admission also requires convergence (almost no divergent transitions, R-hat ≤ 1.05, bulk ESS ≥ 100), that a failing fit is already refit once at target_accept 0.95 so smaller steps are not a fix, and which reparameterisations are (non-centred, priors that constrain every parameter, no parameters that trade off, no hard thresholds), a description of the other docs | yes | yes | yes |
   | `CANDIDATE_BRIEF.md` | the lens text + the one-hypothesis rule (+ critique note) | names the incumbent, its standing, hypothesis and source; lifts the anti-grafting/anti-composition rules; asks for one stated change (+ critique note) | "refine a model of your choosing" from the menu; same lifted rules (+ critique note) |
   | `existing_hypotheses.md`: every zoo model's manifest rationale, ranked by `az.compare` with "rank r, Δ ± dse nats behind (x× dse: tied/lost), ELPD" (trial-level `dse`) and a PSIS-reliability note | yes | yes | yes |
   | `attempted_hypotheses.md` ("Tried before"): the ledger's retired entries with a hypothesis (§5.10) | yes | no | no |
   | `refinement_menu.md`: live non-incumbent models ranked by standing, then ledger-pruned models, narrowest margin first, each with hypothesis and source path (a pruned model's file is in the `model_loop/models/pruned/` of the experiment that pruned it, found through the ledger context; `candidate_agent._pruned_source`) | no | yes | yes |
   | `critiques.md` (only if the round has a critique) | yes | yes | yes |

5. **Spawn.** All pending slots run concurrently (`ThreadPoolExecutor`, 6
   workers, `candidate_parallelism=None` → `candidate_count`). Each is
   `run_coding_agent` with cwd = the agent tree, a 1800 s timeout, sandboxed.
   Admission then happens **sequentially in slot order** (`settle`,
   pymc_orchestrator.py:429), so a later slot's novelty gate compares against
   earlier slots admitted in the same round.
6. **Retry and repair per slot** (`settle`):
   - If the agent process failed (non-zero exit or **timeout**) but a
     `candidate.py` exists in its directory, that candidate goes through
     admission as usual (pymc_orchestrator.py:435-445); a half-written file
     fails the gates. If it is rejected, the slot gets a repair (below).
   - If the agent process failed and wrote no `candidate.py`, the attempt is
     recorded in the ledger as `rejected` / "agent process failed — nothing
     admitted" under the fallback name, and the slot gets one **retry**.
   - If the process succeeded but wrote no `candidate.py`, the slot gets one
     **retry** in `candidate_<i>_retry_1/` with `_retry_note`.
   - If a file was written but admission rejected it, the slot gets one
     **repair** in `candidate_<i>_repair_1/`. The rejection reason is quoted
     verbatim (`_repair_note`), and the rejected `candidate.py`,
     `hypothesis.md` and `model_name.txt` are copied in. A repair is final.
   - The maximum is 3 attempts per slot (original, retry, repair). Retries and
     repairs spawn together as the next wave.
7. **Empty-round guard.** If no slot was admitted and every slot's result is
   "no candidate.py written" or a failed spawn with no file, the whole round is
   rerun once in `iter_<i>_retry_1/` (`MAX_EMPTY_ROUND_RETRIES = 1`). If that
   fails too, the round is abandoned: a ledger line `__round__ / round_abandoned`
   is written, **no history step** is recorded, and the loop moves to the next
   round. If every round is abandoned, the loop raises `AllCandidatesNoFileError`.
8. **Rescore:** `_score`, then `_compare`, then `_record_history_step` with the
   round's `critique` status. **Nothing is pruned during the rounds**; the
   zoo only grows within an experiment.

The loop runs all `max_iterations` rounds. There is no early stopping.

**End of experiment** (pymc_orchestrator.py:596-626): `_prune_losers`, then
`_cap_live_set` (§5.11), both with ledger context `"experiment<k> end of experiment"`.
If anything was retired, `_score` and `_compare` are rerun and the names are
added to the **last** history step as `retired_at_experiment_end` (no new
step). Then `_export` (§5.13).

### 5.7 CriticAL critique (critique_round.py, critique/ppc.py)

- **Incumbent:** `_best_exportable_model` of the latest scoring
  (critique_round.py:570).
- The incumbent's fit is written to the cache dir under its fingerprint
  (`_seed_critique_fit_cache`). `CRITIQUE_CONTEXT.md` is written and inlined
  into the prompt (`prompts/critique.md` + the context). The context opens
  with the task description (§5.1), then the incumbent's name, code path and
  hypothesis, the responses path, its columns (the 5 raw columns), the zoo
  path, and the request for **8** files `test_stats/<name>.py`, each defining
  `test_statistic(df) -> float` with `# name:` and `# description:` headers.
  The number 8 is only requested; any number of usable files at or above 1 is
  accepted.
- The critique agent's writable directory is `iter_<i>/critique/`. The zoo and
  data are read-only.
- **Usable statistics** (`_usable_test_statistics`): each `test_stats/*.py`
  file must pass the code gate (§5.9; offending files are deleted) and then
  run once on the observed data (`check_test_statistic`: no raise, within
  5 s, a finite value, and fast enough that `1 + n_replicates` calls at that
  speed fit the 300 s budget per statistic). Files that fail the run are moved to
  `critique/broken_statistics/`. If none is usable, the agent is re-spawned
  once (`MAX_CRITIQUE_RETRIES = 1`) with a note listing each set-aside
  statistic and its error. If there are still none, the round status is
  `no_critique` and the candidates run without a critique. Any exception other
  than `AgentPermissionDenied` is caught and also gives `no_critique` with the
  reason.
- **Evaluation** (the pipeline runs it in-process; the agent does not):
  `run_ppc_for_model` → `evaluate_test_stat_dir` (ppc.py:428).
  - Observed frame: the pooled `responses.csv` as a DataFrame.
  - Replicates: `sample_synthetic_responses` (pymc_inference.py:616) draws
    posterior-predictive `response` over all chain×draw samples (seed 42) of
    the incumbent's fit (the escalated fit if there was one). It then keeps
    **1000** evenly strided rows (`CRITIQUE_PPC_REPLICATES`), and each
    replicate frame is the observed frame with `chose_left` replaced by one row.
  - For each statistic: `t_obs`, `t_null[1..1000]`. Each call has its own
    5 s SIGALRM limit (`_TEST_STAT_CALL_TIMEOUT_SEC`) and all of one
    statistic's calls share a 300 s budget (`_TEST_STAT_BUDGET_SEC`); the
    context tells the agent both. (One 30 s limit used to cover all 1001
    calls, ~30 ms per call, which ordinary statistics missed at 7,680 rows.)
    `n_ge = #{t_null ≥ t_obs}`, `n_le = #{t_null ≤ t_obs}`, and
    `p = min(1, 2 · min((n_ge+1)/(n+1), (n_le+1)/(n+1)))` (two-sided with the
    +1 correction). Also `z = (t_obs − mean)/sd`. If the code raises or returns
    a non-finite value, or runs out of time, `error` is set (naming the call,
    e.g. "on replicate 17 of 1000") and p is NaN.
  - **Significant = raw p ≤ 0.05** (`CRITIQUE_SIGNIFICANCE_ALPHA`), with no
    correction. A Benjamini–Hochberg q (`_benjamini_hochberg`, ppc.py:382,
    over the finite p's) and `significant_fdr` are reported alongside.
- **Outputs:** `critique/ppc_results.json` (all statistics, significant ones
  first, then by |z|) and `critique/critiques.md`. `critiques.md` states how
  many of the evaluated statistics were significant (and how many could not be
  evaluated; if none could, it says there is no critique), then lists **only**
  the raw-significant statistics with observed value, null mean, z, p, q and a
  "[survives FDR]" mark, then every statistic that could not be evaluated
  with its error. It is inlined into every candidate prompt of that
  round, and the context says to prefer discrepancies that survive FDR.
- **History record:** `{"status": "critiqued", incumbent, attempts, n_statistics, n_evaluated, n_significant, n_significant_fdr}`
  or `{"status": "no_critique", incumbent, [attempts], reason}`. A round in
  which the check ran but **no** statistic produced a p-value is
  `no_critique`, its reason listing each statistic's error, and the
  candidates run without a critique.

### 5.8 The candidate self-test (check_candidate.py)

`CONTEXT.md` shows the command
`<harness sys.executable> -m src.pipelines.inner_loop.check_candidate --candidate-dir <dir> --responses <pooled csv>`.
The interpreter is the venv through the opaque `$AGENT_DIR/venv` link, and the
command runs from the agent tree's code. It runs, in order: the code gate,
load, finite logp and gradient, a smoke fit (100 draws, 100 tune, 1 chain,
1 core, no cache dir), and finite ELPD-LOO. It prints `OK` or the rejection
reason in admission's own wording. It does **not** check `hypothesis.md`,
convergence (one chain has no R-hat, and the smoke fit is never refit) or
novelty. The pipeline does not require the agent to run it.

### 5.9 Admission gates, in order

`_resolve_candidate_name` (model_zoo.py:189) runs first, then
`_admit_candidate_with_reason` (model_zoo.py:858). The first failure rejects
the candidate (`reject` records it in the ledger with the reason):

| # | Gate | Detail |
| --- | --- | --- |
| – | name | `model_name.txt` must match `[a-z][a-z0-9_]{2,40}`, must not look like `iterN_candidateM`, and must not be `inner_loop_model`/`best_model`. Otherwise the fallback `iter{i}_candidate{j}` is used, which is **not a rejection**. A name already in the zoo gets `_2`, `_3`, … |
| 1 | `candidate.py` exists | "no candidate.py written" |
| 2 | `hypothesis.md` exists and is non-empty | |
| 3 | code gate | AST walk, `import_gate.py`: imports only from numpy, pymc, pytensor, arviz, scipy, math, itertools, functools, collections, re, typing, dataclasses, statistics, operator; relative imports and unparseable source are forbidden; no use of the names `open`, `__import__`, `exec`, `eval`, `compile`, `globals`, `vars`, `locals`, `getattr`, `setattr`, `delattr`, `breakpoint`, `input`, `__builtins__`, `__loader__`, `__spec__`; no attribute (nor `from … import` name, nor dotted import component) in `FORBIDDEN_ATTRIBUTES`: module names that allowed modules re-export (`.sys`, `.os`, `.builtins`, `.io`, `.npyio`, …), file readers and writers (`.open`, `.read`, `.load`, `.DataSource`, `.read_*`, `.to_csv`, `.save`, …), `attrgetter`/`methodcaller`, and introspection routes (`__dict__`, `__traceback__`, frame attributes, …); no `str.format` whose fields look up attributes (`"{0.sys}".format(...)`). The same gate screens critique statistics, which run with `pd` injected. The harness also clears `sys.argv`/`sys.orig_argv` once parsed (`forget_command_line`), since they name the GT |
| 4 | loadable | `load_pymc_model`: a module-level `model: pm.Model`, with hooks attached |
| 5 | finite logp and gradient at the initial point on the pooled responses | `model_logp_is_finite` |
| 6 | real fit | full production `fit_model` (§5.3), cached, with the escalation refit if needed |
| 7 | convergence | the returned fit passes the gate (§5.3); the rejection reason says the fit already ran at target_accept ≥ 0.95 (for a multi-chain fit), so raising it will not help, and suggests changing the geometry: non-centred parameterisations, tighter priors on weakly constrained parameters, fewer weakly identified parameters, no hard thresholds (it used to suggest declaring `SAMPLER_SETTINGS = {"target_accept": 0.95}`, which the refit had already done) |
| 8 | finite ELPD-LOO | from that fit |
| 9 | novelty | see below; skipped if threshold = 0 |

On admission, `hypothesis.md` is copied to `models/<name>.hypothesis.md`, the
manifest gets `{name, rationale: hypothesis}`, and the ledger gets `admitted`.
**PSIS reliability is not an admission gate.** An unreliable model is admitted
and competes, but it cannot be selected as best and is shielded from pruning.
Carried and seed models never pass through admission, so a non-converged seed
or carried model stays in the zoo, likewise unselectable and unprunable.

**Novelty gate** (`_min_prediction_rmse`, model_zoo.py:475):

- Pool: `novelty_pool_rows()` = `generate_candidate_pool(512, lengths=(4,5,6,7,8), seed=20260919)`
  (model_zoo.py:582-603, stimulus_design.py:45). That is 512 distinct
  same-length unordered pairs, sampled round-robin over lengths (103, 103,
  102, 102, 102). The sample is identical in every cell. It is deliberately
  not the eval pool, and it does not include length 2–3 pairs, which the
  design pool does include.
- For the candidate and for **every model currently in the zoo manifest**
  (seeds, carried models and every model admitted earlier in this experiment,
  including this round's; nothing is pruned before the end of the experiment):
  posterior-mean `p_left` on the pool, from the production fit on the pooled
  data, averaged over all 8,000 draws (no thinning). A model that binds
  `participant_id` is averaged over the training participant ids. A candidate
  that needs other non-stimulus columns (e.g. `trial_index`) is rejected.
  A candidate whose `p_left` is undefined (NaN or outside [0, 1]) on any pool
  stimulus is rejected, with the count and example pairs in the reason. A zoo
  model undefined on some pool stimuli is compared on the rest (a
  `[novelty]` line says so) and is left out when undefined on all of them.
- `RMSE(c, m) = sqrt(mean_j (p̄_c,j − p̄_m,j)²)`. The candidate is rejected if
  `min_m RMSE(c, m) < 0.002`, and the reason names the nearest model.

### 5.10 The ledger (`attempted_hypotheses.jsonl`)

This is `model_loop/attempted_hypotheses.jsonl` (hypothesis_ledger.py). One
JSON line is appended per event, with exactly the keys `name, outcome, detail, hypothesis, context`.
Outcomes are `admitted`, `rejected`, `pruned`, `dropped` and `round_abandoned`.
The hypothesis is stored in full (whitespace collapsed). The context has the
form `"experiment2 round 3 candidate 1 lens 10"`,
`"… candidate 3 refine incumbent <name>"`, `"… candidate 5 refine chosen"` or
`"experiment2 end of experiment"`, with `" retry 1"` / `" repair 1"` appended
for later attempts. A prune's `detail` is `"<Δ> nats behind <best> (<Δ/dse_clustered>× dse)"`,
which is parsed back to rank the refinement menu. A model retired by the cap
is also recorded with outcome `pruned`, with the detail
`"<Δ> nats behind <best>; retired to keep the live set at 8 models: <why>"`,
so the menu ranks it the same way.

- `retired(live_names)` takes the latest entry per name that is not in the zoo
  **and has a non-empty hypothesis**, so the `__round__` pseudo-entry and
  failed-agent-process lines are left out. It is rendered as
  `attempted_hypotheses.md` ("Tried before") for exploratory slots, and a
  slot's own previous attempt is left out. The framing: a *pruned* entry lost
  by the stated margin on the data of its time and may come back only with a
  substantive change; a *rejected* near-duplicate must not be re-proposed; a
  candidate rejected for its code or fit was never tested and may be tried
  again correctly.
- `pruned(live_names)` is the subset whose latest outcome is `pruned`. These
  are the refinement menu's pruned targets, sorted by `parse_prune_margin`.
- The ledger is copied to `cognitive_models/` at export and inherited by the
  next experiment.

### 5.11 Pruning and the live-set cap (end of experiment only)

**`_prune_losers`** (model_zoo.py:625), once, after the last round:

1. `compare_table` on the zoo. The baseline is rank 0, the raw-ELPD best,
   whether or not it can be trusted.
2. If the baseline is untrusted (`loo_unreliable` or `not_converged`),
   **nothing is pruned**.
3. Otherwise a model m is pruned if it is not protected, is in the table, is
   trusted, has `dse_clustered > 0`, and has
   `elpd_diff_m > 2.0 · dse_clustered_m`.
4. Pruned files (`.py`, `.hypothesis.md`) move to `models/pruned/`. The
   in-process fit cache entry is evicted, a ledger `pruned` line with the
   margin is written, and the manifest is rewritten (`_retire`, model_zoo.py:776).

**`_cap_live_set`** (model_zoo.py:720), right after: if the zoo manifest
(seeds included) holds more than 8 models, it retires the excess among
non-protected models, untrusted ones first, then the worst `az.compare` rank
first. Retired models go to `models/pruned/` with a ledger `pruned` line (see
§5.10 for its detail). If there are too few non-protected models to reach 8,
it retires what it can and warns.

There is no stacking-weight criterion. Protected seeds stay however far behind
they are. Untrusted models are never pruned but are the first retired by the
cap. Neither step can remove the best exportable model.

### 5.12 `history.json`

This is rewritten after every scoring step (`_record_history_step`,
scoring.py:141). It has one entry for the seed step plus one per
non-abandoned round, so up to 6 per experiment and 18 per cell:

| Field | Meaning |
| --- | --- |
| `step` | 0.. within the experiment |
| `iteration` | `null` for the seed step, else the round index |
| `best_model` | `_best_exportable_model` (§5.5) |
| `argmax_model` | softmax-posterior argmax (with complexity prior), for audit |
| `excluded_unreliable` | names that are `loo_unreliable` **or** `not_converged` |
| `posteriors` | softmax posterior (6 dp) |
| `elpd_loo` | ELPD-LOO (4 dp) |
| `critique` | round steps only (§5.7) |
| `retired_at_experiment_end` | last step only, if the end-of-experiment prune or cap retired anything |

The `pruned` field `_record_history_step` supports is never written now,
because the rounds pass `pruned=[]`.

### 5.13 End of the inner loop: `_export` (scoring.py:216)

This writes `model_posterior.json` (posterior + `comparison` +
`excluded_unreliable`, then `best_model`), `best_model.py` and `report.md`,
from the scoring after any end-of-experiment retirement.

---

## 6. Export and carry-forward

`_export_inner_loop_models` (model_loop_runner.py:102) rewrites
`experiment{k}/cognitive_models/` as the **live set**:

- It keeps every previous entry that is protected or still in the zoo, in its
  original order, and deletes the files of carried non-protected models the
  loop pruned, retired or dropped.
- It appends every zoo survivor that is not already present, in zoo order,
  under its own name. A survivor with an auto name (`iterN_candidateM`) is
  exported as `inner_loop_model`, `inner_loop_model_2`, …, with the best model
  named first.
- Each entry's rationale is the model's hypothesis. The step raises on an
  empty rationale or a missing file.
- It checks that every exported model can bind a raw row.
- It copies `attempted_hypotheses.jsonl` beside the manifest.

The carried set is the three protected seeds plus every non-protected
survivor of the prune and the cap, so at most 8 models (the cap counts the
zoo). Untrusted survivors are included. Then
`update_registry_from_interpretation` writes the uniform registry (§2), and
the `5_model_loop` validator checks `model_posterior.json` and `report.md`.
Experiment k+1 copies this set (§2), re-seeds its zoo from it, and refits
everything on the larger pooled data. `models/pruned/` is not carried; pruned
models survive as ledger lines, and their files stay in the pruning
experiment's `model_loop/models/pruned/`, where the refinement menu points.

---

## 7. Evaluation of recovery

This runs in the harness after the three experiments
(holdout_recovery.py:615-709). No agents are involved.

### 7.1 Held-out eval pool

`build_eval_stimuli` (holdout_eval.py:94) with `exhaustive=True` and lengths
1..8:

- The pool is `enumerate_all_pairs([1..8], same_length_only=True)`, which is
  43,435 pairs (the design's 43,434 plus the single length-1 pair H/T).
- Every unordered pair that appears in any `experiment{1..3}/data/responses.csv`
  (in either displayed order) is removed (`collect_trained_pairs`), at most
  3 × 64 = 192 pairs.
- The step raises if fewer than 100 remain (`min_remaining`). The result is
  written to `cell_1/eval_stimuli.json`, and `n_eval_dropped` is recorded.
- `n_pairs: 500` and `seed: 11` (defaults) are unused when the pool is
  exhaustive.

### 7.2 Per-step trajectory (`evaluate_trajectory`, holdout_eval.py:320)

- GT: `q = p_left_fixed_params(gt, gt_models_src, eval, DEFAULT_PARAMS)`.
- For every `history.json` entry of every experiment (global_step 0..17):
  - `best_model` and every model with `posteriors > 0`: refit with the loop's
    `fit_kwargs` on that experiment's `model_loop/responses.csv`. This is a
    cache hit (including an escalated fit). A model pruned or retired later is
    loaded from `models/pruned/`.
  - Posterior-mean `p_left` on the eval pool uses `predict_max_draws = 500`,
    i.e. 125 evenly spaced draws per chain (`_eval_prediction`).
    Participant-effect models are averaged over the training participants.
  - **Undefined predictions** (`mask_invalid=True`). If any posterior draw of a
    model gives a `p_left` that is NaN or outside [0, 1] for a pair, that
    model's prediction for the pair is NaN (`InvalidPredictions`,
    pymc_inference.py:485). A pair is excluded from the step's metrics if the
    best model's or the BMA's prediction is NaN there (so a NaN from any
    positive-weight model excludes it). The row records `n_eval_excluded` and
    `eval_excluded_models`; each affected step appends one JSON line (with
    the pairs) to `RUN_DIR/eval_exclusions.jsonl`. The step raises if every
    pair is excluded. Any other prediction error raises.
  - **BMA:** each model with `posteriors > 0` at that step, weighted by its
    softmax posterior renormalised over those models (`_bma_prediction`). If
    no model has positive weight, the best model's prediction is used. The
    posterior is rounded to 6 dp, so models more than about 14 nats behind get
    weight 0.
- Metrics for both best and BMA over the non-excluded pairs
  (`recovery_metrics.py`, `recover.pearson_r`):

  | Field | Formula |
  | --- | --- |
  | `pearson_r` | Pearson r(q, p). `None` if either side has a single distinct value |
  | `rmse` | sqrt(mean (p − q)²) |
  | `kl_regret` | mean_j [ q log(q/p) + (1−q) log((1−q)/(1−p)) ], both clipped to [1e-9, 1−1e-9] (nats) |
  | `bias` | mean(p − q) |
  | `calib_slope`, `calib_intercept` | OLS of p on q |

### 7.3 Baselines

- **`baseline` (fixed-parameter seeds, no learning):**
  `seed_baseline_correlation` (holdout_eval.py:467). Each registry model other
  than the GT, at its own `DEFAULT_PARAMS`, gives a fixed `p_left` on the eval
  pool and a Pearson r with q. Output: `per_model` r and `mean_r`. No RMSE is
  computed.
- **`fitted_baseline` (fitted seeds, no agents):**
  `fitted_seed_baseline_correlation` (holdout_eval.py:546). The three non-GT
  registry models (from `pymc_model_families/`) are fit with the loop's
  `fit_kwargs` on the final experiment's `model_loop/responses.csv`, which
  holds every experiment's responses once (`_all_responses_so_far`,
  holdout_eval.py:515, checks its row count against the experiments'
  `data/responses.csv`: 7,680). The files and data match the loop's
  experiment-3 fits of the protected seeds, so these are normally cache hits.
  Each predicts the eval pool (≤500 draws, no masking of undefined values).
  Output: `per_model` {`pearson_r`, `rmse`, `elpd_loo`, `trusted`} (trusted =
  PSIS-reliable and converged); `elpd_best_model`, `elpd_best_r`,
  `elpd_best_rmse` — the trusted seed with the highest ELPD-LOO, chosen on the
  training data as the loop chooses its winner — or `None` with
  `elpd_best_reason` if no seed is trusted; `mean_r` and `mean_rmse` over the
  seeds as reference fields; and `n_responses`.

### 7.4 Incumbent record (incumbent.py)

- `starting_models_of_run`: the keys of `posteriors` in experiment 1's seed
  step. The harness checks these against the pool minus the GT
  (`_require_seeded_from_pool`).
- For each trajectory row, `incumbent_changed` is true if `best_model` differs
  from the previous global step. It is always false at step 0, and experiment
  boundaries count as steps. `incumbent_is_discovered` is true if `best_model`
  is not a starting model.
- `incumbent` block: `starting_models`, `n_steps`, `n_incumbent_changes`,
  `n_steps_discovered_incumbent`, `final_incumbent`, and `changes[]`
  (global_step, experiment, step, from, to).

### 7.5 Leakage check (`leakage_check`, leakage_audit.py)

This is a heuristic audit. It flags but does not enforce. Over every
`experiment*/cognitive_models/*.py` and `experiment*/model_loop/models/*.py`
(not `models/pruned/`), it records:

- `identical`: sha256 equals the GT's PyMC file.
- `mentions_gt_params`: the source contains any of the GT family's parameter
  names other than `beta`/`side_bias`, as a **substring**. The set is empty for
  `falk_konold_dp` and `finite_experience_occurrence`. For `motif_stack` it
  includes the generic words `delta` and `alpha`, which gives false positives.
- `mentions_gt_values`: the source contains the `str()` of a distinctive
  default value other than 0, 0.5 or 1.
- `gt_named`: the filename is `<gt>.py`.
- `data_columns`: the `pm.Data` names the file binds.

It also flags CSVs under the run tree whose header has `generating_model`, and
seed manifests in the agent tree (outside `_runs`) that still list the GT.
`any_manifest_gt_named` is `None` if no checkout was scanned. The GT-name
scans and the activity report of §1.2 are separate, sbatch-level checks.

### 7.6 Outputs

| File | Written by | Contents |
| --- | --- | --- |
| `$WORK_ROOT/run<r>/<gt>/trajectory.json` | `_run_holdout_recovery_resolved` (holdout_recovery.py:709) | one `gt_run`: `gt_model`, `params` (the true params, hence outside the agent tree), `run_root`, `n_eval_stimuli`, `n_eval_dropped`, `trajectory[]` (rows include `n_eval_excluded`, `eval_excluded_models`), `incumbent`, `baseline`, `fitted_baseline`, `leakage`, `experiments[{experiment, manifest_models}]`. Its presence makes `--resume` skip the cell. |
| `$WORK_ROOT/run<r>/<gt>/eval_exclusions.jsonl` | `evaluate_trajectory` | one line per step with excluded held-out pairs (only if any) |
| `$WORK_ROOT/run<r>/<gt>/holdout.json` | script `main` | `project_id`, `seed_models_dir`, `n_experiments`, `n_participants`, `inner_loop{max_iterations, candidate_count, novelty_rmse_threshold, n_critique_proposals}`, `fit_kwargs`, `seed`, `eval_pool`, `metrics_version: 2`, `gt_runs[ … ]` |
| `holdout.csv` | `trajectory_tidy_rows` | one row per step: `TRAJECTORY_COLUMNS` + incumbent flags (not the exclusion fields) |
| `holdout.png` | `plot_holdout_trajectories` | trajectory figure |
| `_runs/token_usage.jsonl` + report | `start_usage_log` / `write_usage_report` | agent token spend |
| `run<r>/<gt>/gt_name_mentions.txt`, `agent_activity.md` | sbatch, after the run | §1.2 |
| `run<r>/<gt>/agent_runs.tar.gz` | sbatch, on success | the whole `_runs/` tree. The agent tree is then deleted, unless `KEEP_REPO_COPY=1` |

---

## 8. Seeds: what varies between repeats

`derive_seed(*parts)` (holdout_recovery.py:294) is the first 4 bytes of
sha256 of the `|`-joined parts, mod 2^31. The cell seed is
`BASE_SEED + REPEAT` (sbatch:83).

| RNG | Seed | Varies by |
| --- | --- | --- |
| Synthetic responses and left/right coin flips | `derive_seed(cell seed, GT, exp, "responses")` | repeat, GT and experiment |
| Prior-predictive draws, design posterior-predictive draws, 40-response EIG scenarios, random part | `derive_seed(cell seed, GT, exp, "design")` | repeat, GT and experiment |
| Single-response fill scenarios | design seed + 1 | repeat, GT and experiment |
| All MCMC fits (loop, design, evaluation) | 42 | nothing |
| Novelty pool | 20260919 | nothing |
| PPC replicates | 42 | nothing |

Apart from the LLM agents, repeats of the same GT differ in the Bernoulli
responses and in the design (experiment 1's prior-predictive draws and EIG
scenarios are seeded per cell, and later designs also depend on the fitted
posteriors).

---

## Doc/code discrepancies and other observations

Discrepancies, where the code wins:

1. **README.md:14** says the seed pool is "the best models discovered by three
   earlier human replicate runs". The live pool is the four literature-faithful
   models. The hero-run pool is archived in `seed_models/archive_hero_run_2026_07/`.
2. **CLAUDE.md, Ledger bullet** says the ledger is rendered as "already tried —
   do not re-propose". The heading is now "Tried before", and pruned mechanisms
   may return with a substantive change (as CLAUDE.md's own Pruning bullet
   says). Its Export bullet says a rival "within 2·dse" is carried; the margin
   is 2 · `dse_clustered`, subject to the cap of 8.
3. **model_zoo.py:573-576** says the novelty pool is "over the design's pair
   universe (same-length H/T pairs at lengths 4–8)". The design universe is
   lengths 2–8.
4. **model_zoo.py:553-558, `_prune_losers` docstring, and pymc_orchestrator.py:249**
   say pruning runs "after each scoring pass" and compares against `dse`.
   Pruning runs once, at the end of the experiment, against `dse_clustered`.
5. **posterior.py:149-151, scoring.py:228-229, loo_reliability.py:3-4** say an
   unreliable model is excluded from, or zeroed in, the next design's prior.
   The registry is uniform over every carried model, including untrusted ones.
6. **pymc_orchestrator.py:6-8 (module docstring)** says the softmax posterior
   "selects the incumbent". Selection is by `az.compare` rank among trusted
   models.
7. **holdout_recovery_array.sbatch:25-29 and model_loop_runner.py:75-76** say
   agents run with "no read sandbox". Every loop agent runs in bubblewrap
   (`sandbox=True`).
8. **Faithful config, `agent.backend` comment** says "null -> CODING_AGENT env
   var, then 'claude'". The code default is `opencode` (coding_agent.py:57).
   Under the array the config key is overridden anyway by
   `--backend ${AGENT_BACKEND:-opencode}`.
9. **Faithful config `seed: 7`** is never used under the array, because the
    sbatch always passes `--seed BASE_SEED+REPEAT`.
10. **scripts/subjective_randomness/holdout_recovery.py:4-6** mentions "real
    theory, design, and candidate-conjecturing agents". Only critique and
    candidate agents exist.
11. **src/subjective_randomness/holdout_recovery.py:11-16 (module docstring)**
    shows the run tree as `<gt_model>/…/trajectory.json`. The run directory is
    `cell_<i>/`, and `trajectory.json` is written to the summary root, outside
    the agent tree.
12. **holdout_recovery_array.sbatch:163-189** stubs `model_families/<gt>.py`,
    but `agent_tree.exclude` already removes `src/subjective_randomness/`, so
    the stub is never written. This is harmless.
13. **candidate_agent.py:47-48** says twelve lenses let a round walk "four
    rounds without repeating". With 5 rounds, each experiment's round 4 reuses
    round 0's lenses. The config comment's weaker claim (no repeat *within* a
    round) holds.
14. **`existing_hypotheses.md` and the refinement menu** (`_describe_standing`,
    candidate_agent.py:102) label a model "tied" or "has lost" at 2 × the
    trial-level `dse`, while pruning uses 2 × `dse_clustered` (larger). An
    agent can be told a model has lost that pruning will keep. The standing
    text also does not mention a failed convergence gate, only PSIS
    reliability.

Bugs and behaviour worth a decision (read from the code, not observed in a run):

- **A cell with no trusted model fails.** If at any scoring step no model is
  both PSIS-reliable and converged (after escalation), `_best_exportable_model`
  raises in `_record_history_step`. Seeds and carried models are never gated,
  only excluded from selection.
- **The design scores a non-converged posterior.** A design-time fit that
  still fails the gate after its refit is used for the posterior-predictive
  draws without a warning beyond the sampler's own.
- **EIG ignores counterbalancing** (§3.3): it scores 40 responses to the
  enumeration orientation, while the data show each pair in a random
  orientation. The difference matters only for draws with a nonzero
  `side_bias` (or other orientation asymmetry).
- **Fill `joint_eig_bits` are in different units** from the 40-response
  picks' (§3.6), in the same `stimuli.json` column, distinguished only by
  `source`.
- **holdout.csv omits the evaluation exclusions.** `n_eval_excluded` and
  `eval_excluded_models` are in `trajectory.json`/`holdout.json` but not in
  `TRAJECTORY_COLUMNS`, so a tidy-CSV reader cannot see that a step's metrics
  used fewer pairs.
- **The fitted-seed baseline does not mask undefined predictions**, unlike
  the trajectory: a seed with an invalid `p_left` on some held-out pair raises.

Things I did not verify at runtime (read from code only): arviz's `good_k`
value at 8,000 draws (assumed 0.7), `az.compare`'s default weight method
(stacking), whether `pm.sample_prior_predictive(draws=1)` of `p_left` under
`pm.do` is exactly deterministic for every GT (it should be, since all free
RVs are fixed and `p_left` is a Deterministic of them and the data), how many
of the 64 picks the 40-response selection makes before its noise floor in
practice, and the two bugs above.
