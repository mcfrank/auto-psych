# The modeling loop, step by step: one holdout-recovery cell

This is a reference for reviewing what the code does in one holdout-recovery
cell, from the Slurm array task to the recovery metrics. It was written by
reading the code on branch `consolidate/2026-09`. Where a docstring, `README.md`
or `CLAUDE.md` says something different, this document follows the code, and
the difference is listed in the last section.

Concrete values come from
`scripts/subjective_randomness/configs/holdout_recovery_faithful.yaml` and the
defaults it does not override. `file:line` references are to this branch.

> **Leak warning.** This file names the ground-truth models and describes the
> harness. `holdout_recovery_array.sbatch` builds each agent tree with
> `rsync --exclude-from=agent_tree.exclude`, and `agent_tree.exclude` does not
> exclude `docs/`, so this file ends up in every agent tree. The array's
> GT-named-file scrub does not remove it either, because its filename names no
> model. For that reason the numeric `DEFAULT_PARAMS` values are left out here
> and cited by line instead. Before a sweep, add `docs/modeling_loop.md` (or
> `docs/`) to `agent_tree.exclude`. Several other files in `docs/` already name
> the ground truths.

---

## 0. Overview and call chain

```
holdout_setup.sbatch            (once per sweep: stage harness_repo, gt_models_src, gt_family_src)
holdout_recovery_array.sbatch   (one task = one (repeat, ground truth) cell)
 └─ scripts/subjective_randomness/holdout_recovery.py : main
     └─ src/subjective_randomness/holdout_recovery.py : run_holdout_recovery_from_config
         └─ _run_holdout_recovery_resolved            (one GT, because --gt-model is passed)
             ├─ run_holdout_experiments               (experiments 1..3)
             │   for each experiment:
             │   ├─ model set: seed_experiment_models_from_project | carry_forward_cognitive_models
             │   ├─ design:    orchestrator.run_design_programmatic → eig.design_exhaustive
             │   ├─ collect:   holdout_data.generate_responses (GT, fixed params)
             │   └─ inner loop: model_loop_runner.run_inner_model_loop_programmatic
             │                  → inner_loop.pymc_orchestrator.run_pymc_inner_loop
             │                  → _export_inner_loop_models, update_registry_from_interpretation
             ├─ build_eval_stimuli, evaluate_trajectory
             ├─ annotate_incumbents / summarise_incumbents
             ├─ leakage_check
             ├─ seed_baseline_correlation, fitted_seed_baseline_correlation
             └─ write trajectory.json (outside the agent tree)
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
| Stimuli per experiment | 64 = 32 EIG + 32 random | config `design` |
| Design pair lengths | 2..8, same-length pairs only | `run_design_programmatic` default `lengths` (orchestrator.py:430), `design_exhaustive` (eig.py:241) |
| Cell seed | `BASE_SEED + REPEAT` (default 0 + r) | sbatch:83, passed as `--seed`; **overrides config `seed: 7`** |
| Inner-loop rounds per experiment | 5 | config `inner_loop.max_iterations` |
| Candidate slots per round | 6 (3 explore, 2 refine incumbent, 1 refine chosen) | config `candidate_count`; `model_zoo.slot_roles` |
| Critique test statistics requested | 8 | config `n_critique_proposals` |
| Novelty RMSE threshold | 0.002 | `DEFAULT_NOVELTY_RMSE_THRESHOLD` (model_zoo.py:570); not set in config |
| Pruning multiplier | 2.0 · dse | `DEFAULT_PRUNE_DSE_MULTIPLIER` (model_zoo.py:559) |
| Agent backend / model | `opencode` / `google/gemini-3.1-pro-preview` | sbatch `--backend ${AGENT_BACKEND:-opencode}` (sbatch:354); config `agent.model` unless `AGENT_MODEL` is set |
| Agent timeout | 900 s per agent attempt | config `agent.timeout_sec` |
| Production MCMC | 2000 draws, 1000 tune, 4 chains, target_accept 0.8, max_treedepth 10, seed 42 | config `fit` + `_FIT_DEFAULTS` (pymc_inference.py:346) |
| Eval pool | exhaustive same-length pairs, lengths 1..8, minus trained pairs; ≤500 posterior draws per prediction | config `eval_pool` |

---

## 1. Setup and holdout

### 1.1 Array task → (repeat, ground truth)

`holdout_recovery_array.sbatch:80-83`: with `G = 4` ground truths, task `T`
maps to `REPEAT = (T-1)/G + 1`, `GT = GTS[(T-1) % G]`, `SEED = BASE_SEED + REPEAT`.
All four ground truths of one repeat get the same seed. Resources: 16 CPUs,
64 GB, 1 day, partition `normal` (sbatch:2-6).

Directories:

| Path | Contents | Seen by agents? |
| --- | --- | --- |
| `$WORK_ROOT/run<r>/<gt>/` (`RUN_DIR`) | `holdout.{json,csv,png}`, `trajectory.json`, `mcmc_cache/`, `agent_runs.tar.gz` | no |
| `$AGENT_TREES_ROOT/<sha256(RUN_DIR)[:16]>/` (`AGENT_DIR`) | `repo/` (agent tree), `mcmc_cache` → `RUN_DIR/mcmc_cache`, `venv` → the shared venv, `.xdg/` | only `repo/` (see §1.5) |
| `AGENT_DIR/repo/_runs/cell_1/` | the run tree: `experiment1..3/`, `agent_notes/`, `eval_stimuli.json`, `pooled_responses.csv` | yes (read-only, except own dirs) |
| `$WORK_ROOT/harness_repo` | full repo copy; the harness process runs from here | no |
| `$WORK_ROOT/gt_models_src`, `gt_family_src` | pristine copies of `pymc_model_families/` and `model_families/` | no |

The run directory is named `cell_<i>`, not after the ground truth
(`holdout_recovery.py:547`), because every path in the run tree appears in agent prompts.

### 1.2 Building the agent tree

In order (sbatch:126-300):

1. `rsync -a --exclude-from=agent_tree.exclude $REPO/ $RUN_REPO/`. The exclude
   list drops `.git`, `data`, caches, `CLAUDE.md`, `AGENTS.md`,
   `ground_truth_models.py`, `evaluate_recovery.py`, two literature test files,
   `features.py`, `sequence_stats.py`, `stimulus_design.py`, `model_recovery.py`,
   `pymc_recover.py`, the whole `model_families/` and `pymc_model_families/`
   directories, and `configs/holdout_recovery*.yaml`.
2. Delete `<gt>.py` from the registry path and from the live seed pool
   `src/pipelines/outer_loop/projects/subjective_randomness/seed_models/`.
3. Replace `model_families/<gt>.py` with a stub, but only if the file exists.
   It never does, because step 1 already excluded `model_families/`, so this
   step has no effect under the current exclude list.
4. Write `opencode.json` read/glob/grep deny rules (seed_models, both family
   dirs, holdout configs, `ground_truth_models.py`, `evaluate_recovery.py`,
   `gt.txt`). Only opencode reads this file.
5. Remove the GT's entry from both manifests (`remove_manifest_entry.py`), then
   from every other `models_manifest.yaml` outside `_runs/` that mentions it.
6. Delete every file whose name contains the GT's name, outside `_runs/`.
7. Verify: both family directories are absent, and no `*<gt>*.py` remains.
   Otherwise exit 1.
8. Remove from the agents' environment every variable whose value contains the
   GT name, `$WORK_ROOT` or `$REPO`, plus `GT_MODELS` and `CONFIG` (sbatch:336-346).

The harness then runs from `harness_repo` with `--agent-root $RUN_REPO`,
`--results-root $RUN_REPO/_runs`, `--summary-root $WORK_ROOT/run<r>`,
`--cache-dir $AGENT_DIR/mcmc_cache`, `--gt-models-dir $WORK_ROOT/gt_models_src`,
`--gt-family-dir $WORK_ROOT/gt_family_src`, and `--resume` (sbatch:348-366).

What the agent tree still contains: the three non-held-out seed models in
`seed_models/` (with their manifest), `seed_models/archive_hero_run_2026_07/`,
all of `src/pipelines/`, `src/models/`, `tests/` and `scripts/` except
GT-named files, and `docs/`.

### 1.3 The ground-truth generator and its parameters

- **Generator:** the GT's PyMC model file from the pristine snapshot
  `gt_models_src/<gt>.py`, a copy of `src/subjective_randomness/pymc_model_families/<gt>.py`
  (`run_holdout_experiments` uses `gt_models_dir`, holdout_recovery.py:171).
- **Parameters:** config `gt_models: {name: null}` means the pure-Python
  family's `DEFAULT_PARAMS`. `resolve_generating_params`
  (holdout_data.py:86) calls `_family_default_params` (holdout_data.py:72),
  which reads `gt_family_src/<gt>.py` by **parsing** the `DEFAULT_PARAMS`
  literal with `ast` (`_default_params_from_file`, holdout_data.py:54), without
  importing it. If no pristine directory is given, it imports
  `src.subjective_randomness.model_families.<gt>`.
- **Where the values live:**

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
`pymc_model_families/models_manifest.yaml`, and a test asserts the model files
are byte-identical.

The same three models are the **protected** set: they are never pruned and are
always carried forward (`_protected_seed_names`, model_loop_runner.py:62). The
protected set is the pool manifest intersected with the experiment's
`cognitive_models/`.

### 1.5 What agents can see: the sandbox

Every loop agent (critique and candidates) is launched with `sandbox=True`
(critique_round.py:312, candidate_agent.py:725). `run_coding_agent`
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
`ps` arguments are not visible. The network is shared. Logins: claude needs
`CLAUDE_CODE_OAUTH_TOKEN`, codex gets a private `CODEX_HOME` holding only
`auth.json`, and opencode reads its provider key from the environment
(agent_sandbox.py:215). For opencode, `external_directory` is set to `allow`
because the sandbox itself does the confining (agent_sandbox.py:121-129).

The MCMC cache (`$AGENT_DIR/mcmc_cache`) is outside the agent tree, so agents
cannot see it. Its path does appear in the critique context text.

`stock=True` makes a Claude agent run without the user's configuration.
`memory_dir` is the run's `agent_notes/`, shared by every agent of this cell
across rounds and experiments.

---

## 2. Model set per experiment

`run_holdout_experiments` (holdout_recovery.py:116):

- **Experiment 1:** `ensure_experiment_dirs` creates `cognitive_models/`,
  `design/`, `experiment/`, `data/`, `model_loop/`. `init_registry` writes an
  empty `model_registry.yaml` (model_loop_runner.py:334).
  `seed_experiment_models_from_project` (orchestrator.py:82) copies the kept
  seed files and writes a manifest without the excluded entry. It is a no-op
  if a manifest already exists, which is what makes `--resume` safe.
- **Experiments ≥ 2:** `carry_forward_cognitive_models` (orchestrator.py:147)
  copies the previous experiment's `cognitive_models/*.py`, its manifest and
  `attempted_hypotheses.jsonl`. It raises on a missing file.
- After either path, the `"models"` validator loads every model.

**Registry (the design prior).** After each inner loop,
`update_registry_from_interpretation` (model_loop_runner.py:344) writes
`model_registry.yaml` as `{theories: {name: 1/n for every model in cognitive_models/}, reserved_for_new: 0.0}`.
Experiment k+1's design reads experiment k's registry. The carried set is
copied verbatim into experiment k+1's `cognitive_models/`, so the design prior
is always uniform over exactly the models it scores. In experiment 1 no
registry is passed, and the prior is also uniform. The `az.compare` stacking
weights appear only in `model_posterior.json`.

---

## 3. Stimulus design

Entry: `run_design_programmatic` (orchestrator.py:422) →
`eig.design_exhaustive` (eig.py:181) with `n_select = 32` (the call's `k`),
`n_random = 32`, `lengths = (2,3,4,5,6,7,8)`, `n_samples = 200`,
`n_scenarios = 1000`, `seed = 42` and `random_seed = exp_num`. Output:
`design/stimuli.json` and `design/screened_out.json`.

### 3.1 Candidate pair space

`enumerate_all_pairs(lengths, same_length_only=True)`
(stimulus_design.py:114, called at eig.py:241):

- All `2^L` H/T strings for each L in 2..8, pooled in order (L ascending,
  then `itertools.product("HT", repeat=L)` order).
- Every **unordered** pair of two **distinct** strings
  (`itertools.combinations`), kept only if both have the **same length**.
- Each pair appears once, with `sequence_a` being the string that comes first
  in enumeration order. For equal lengths this is the lexicographically
  smaller string with H < T. **Left/right is never randomised or
  counterbalanced**, here or in data collection. All four GTs have
  `side_bias = 0`, so this has no effect on data generation, but it is a fixed
  property of every stimulus.
- Size: Σ_{L=2..8} C(2^L, 2) = 6 + 28 + 120 + 496 + 2,016 + 8,128 + 32,640 =
  **43,434 pairs**. Length-8 pairs are 75% of the pool.

### 3.2 Screening (`_screen_usable_models`, eig.py:61)

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
- Any other exception: the model is **dropped** and recorded with the error.
  This branch is broader than CLAUDE.md describes.
- No usable model: raise.

`design/screened_out.json` is always written as a list of
`{model, missing, reason}`; it is empty when nothing was dropped. The model
prior is renormalised over the surviving models (§3.4).

### 3.3 Predictive draws: which distribution, how many

For every usable model m, the design builds an array `p[m]` of shape
`(D_m, N)` with `N = 43,434`: `p[m][d, j]` is `p_left` for pair j under
parameter draw d.

- **Experiment 1: prior predictive.** `prior_predict_p_left_draws`
  (pymc_inference.py:181) binds all 43,434 rows at once and runs
  `pm.sample_prior_predictive(draws=200, var_names=["p_left"], random_seed=42)`.
  D_m = 200 draws from each model's **prior** over its parameters.
- **Experiments k ≥ 2: posterior predictive.** `_posterior_p_left_draws`
  (eig.py:136) calls `fit_model` for each model on
  **`experiment{k-1}/data/responses.csv` only**. That is the previous
  experiment's 2,560 responses, *not* the pooled data the inner loop used.
  Settings: `DESIGN_TWIN_DRAWS = 500`, `DESIGN_TWIN_TUNE = 500`,
  `DESIGN_TWIN_CHAINS = 2` (mcmc_defaults.py:26-28). `target_accept` is not
  passed, so it resolves to the model's `SAMPLER_SETTINGS` (0.9 for
  `motif_stack`) or `PRODUCTION_TARGET_ACCEPT = 0.99` otherwise. The config's
  `fit.target_accept: 0.8` does not reach the design. Other settings: `cores` 4
  (2 processes), `random_seed` 42, max_treedepth 10. The posterior is then
  thinned by `_thin_posterior(max_draws=200)` (pymc_inference.py:453) to 100
  evenly spaced draws per chain, and `pm.sample_posterior_predictive` of
  `p_left` gives D_m = 200 (`predict_p_left_draws`, pymc_inference.py:486,
  seed 42).
- `p_left` is deterministic given the parameters, so each "draw" is one
  parameter vector's choice probabilities. Parameter uncertainty enters **only**
  through the D_m draws, as a within-model mixture.
- **No participant structure.** The EIG treats every selected stimulus as
  receiving one response from one virtual respondent. The experiment then shows
  each stimulus to 40 simulated participants who share identical parameters,
  and the objective does not model that replication.
- Probabilities are clipped to [1e-12, 1 − 1e-12] (`_P_CLIP`, eig_selection.py:45).

### 3.4 Model prior π

- Experiment 1: `registry_path=None`, so `model_weights = {}` and π is uniform.
- Experiment k ≥ 2: `_load_model_weights(experiment{k-1}/model_registry.yaml)`
  returns the uniform weights over the carried set (§2). `_model_prior`
  (eig_selection.py:79) looks up each usable model's weight and normalises.
  If the total is 0 it falls back to uniform, and eig.py:262 logs a message.
  In practice π is uniform over the usable models in every experiment.

### 3.5 What EIG is

The objective is the mutual information between model identity M and the
joint binary response vector for the selected set S:

```
I(M; R_S) = H(π) − E_{R_S}[ H(M | R_S) ]          (bits)
```

under the generative process

```
m ~ π,   d ~ Uniform{1..D_m},   R_j | m, d ~ Bernoulli(p[m][d, j])  independently for j ∈ S,
```

so the within-model predictive is the mixture over draws:

```
P(r_S | m) = (1/D_m) Σ_d  Π_{j∈S} p[m][d,j]^{r_j} (1 − p[m][d,j])^{1−r_j}
P(m | r_S) ∝ π(m) · P(r_S | m)
```

It is **joint** over S. Because each draw's likelihood is taken as a product
over the stimuli before averaging over draws, correlation between stimuli
induced by shared parameters is kept. A near-duplicate of a selected stimulus
therefore adds little (eig_selection.py module docstring).

**Monte Carlo estimator** (`_ScenarioState`, eig_selection.py:104). With
T = 1,000 scenarios and `rng = np.random.default_rng(42)`:

1. For each scenario t, sample once and fix `m_t ~ π` and
   `d_t ~ Uniform{0..D_{m_t}−1}` (eig_selection.py:123-125).
2. Keep `logL[m][t, d] = Σ_{i∈S} log Bern(r_{t,i}; p[m][d,i])` for every model
   and draw. It starts at 0.
3. Scenario t's posterior over models: `w_t(m) ∝ π(m) · mean_d exp(logL[m][t,d] − c_t)`.
   `c_t` is a per-scenario max, which cancels on normalisation
   (`posterior_entropy`, eig_selection.py:154). `H_t(S)` is its entropy in bits.
4. In-sample joint EIG: `Î(S) = H(π) − (1/T) Σ_t H_t(S)`.

The scenario's own draw `d_t` is one of the D_m draws in the likelihood
average. The trajectory is therefore an in-sample estimate, as the module
docstring says. `estimate_joint_eig` with a fresh seed would give an
out-of-sample estimate, but the pipeline does not call it.

### 3.6 Greedy joint selection (`select_n_joint_eig`, eig_selection.py:251)

Run with `lazy=False` (exact greedy) and `chunk_size=4096`:

```
S ← ∅;  H_t ← H(π) for all t
repeat 32 times:
  for every pool index j (in chunks of 4096):
      for r ∈ {1 (left), 0 (right)}:
          w_t^{(r)}(m) ∝ π(m) · (1/D_m) Σ_d exp(logL[m][t,d] − c_t) · Bern(r; p[m][d,j])
          H_t^{(r)}(j) = entropy(w_t^{(r)})
      q_{t,j} = p[m_t][d_t, j]                                 # the scenario's true P(left)
      gain(j) = mean_t H_t − mean_t [ q_{t,j} H_t^{(1)}(j) + (1 − q_{t,j}) H_t^{(0)}(j) ]
  gain[S] ← −∞
  j* ← argmax_j gain(j)                                        # np.argmax: first maximum
  for every scenario t: sample r_{t,j*} ~ Bernoulli(q_{t,j*}) with the same rng; logL += log Bern(r_{t,j*}; p[m][·, j*])
  S ← S ∪ {j*};  record joint_eig_bits = H(π) − mean_t H_t(S)
```

`marginal_gains` (eig_selection.py:172) computes the inner sum over draws as
one matrix product per model and outcome: `lhat[m] @ p[m][:, cols] / D_m`.
Each candidate's gain is the expected reduction in posterior entropy from
adding it, averaged over scenarios. Its response is marginalised analytically
using the scenario's true probability, while earlier responses are the sampled
ones held in `logL`.

**Tie-breaking.** `np.argmax` returns the lowest pool index among equal gains.
Pool order is length ascending, then enumeration order, so an all-zero-gain
step (all models agree everywhere unselected) picks the shortest,
earliest-enumerated pairs. There is no other tie rule. Selection is
deterministic given the draws, because `seed = 42` is fixed for every cell,
repeat and experiment.

**Per-stimulus fields written to `stimuli.json`** (eig.py:299-312):

- `eig`: the marginal single-stimulus EIG from draw-averaged means,
  `eig_from_prior_means` (pymc_inference.py:258). With `p̄_m = mean_d p[m][d,j]`,
  `p̄ = Σ_m π(m) p̄_m` and
  `EIG_j = H(π) − [p̄ H(M|R=1) + (1−p̄) H(M|R=0)]`. It is a report field,
  **not** used for selection, and it ignores within-model parameter uncertainty.
- `selection_rank` (1..32, greedy order), `joint_eig_bits` (the in-sample
  `Î(S)` after this pick), `source: "eig"`.
- The ceiling on `joint_eig_bits` is log2 K. With 3 models in experiment 1
  that is 1.585 bits.

### 3.7 The random half

eig.py:324-341. `remaining` is every pool index not chosen by EIG, 43,402 of
them. `random.Random(exp_num).sample(remaining, 32)`, sorted by pool index, is
appended with `eig: null`, `joint_eig_bits: null`, `source: "random"` and
`selection_rank` 33..64.

- **Seed:** the Python `random` module seeded with the **experiment number**,
  not the cell seed. The random half is therefore (nearly) the same across all
  cells and repeats for a given experiment number. It can differ only where
  `sample`'s positions land differently because of which 32 indices EIG removed.
- **Why:** per the config comment, it provides uniform coverage of the "flat
  middle" of the space that EIG avoids, so that the selected model has to fit
  the whole space. It is one fixed sample shown to every participant.
- Neither half excludes stimuli used in earlier experiments. A pair can recur
  across experiments and then gets fresh Bernoulli draws.
- Ablations: `n_eig: 64, n_random: 0` (pure EIG) and `n_eig: 0, n_random: 64`
  (pure random; screening and scoring are skipped entirely).

### 3.8 Order in `stimuli.json`

The order is the 32 EIG picks in greedy order, then the 32 random picks in
pool order. This order becomes `trial_index` (§4), so every participant sees
the EIG stimuli first.

### 3.9 Caching

- Prior-predictive draws are recomputed every time; they are not cached.
- Design-time posterior fits (k ≥ 2) are cached on disk in
  `experiment{k}/design/_fit_cache/<name>.<fingerprint>.nc` and in process
  (`_FIT_CACHE`). The key is (model file sha256, responses file sha256,
  resolved sampler settings), pymc_inference.py:631-667. This cache is
  separate from the cell's shared `mcmc_cache`.
- `load_pymc_model_cached` caches model loading for the screen.

---

## 4. Data collection (simulated from the ground truth)

`run_holdout_experiments` (holdout_recovery.py:233-248) →
`generate_responses` (holdout_data.py:175):

1. `p = p_left_fixed_params(gt, gt_models_src, stimuli, DEFAULT_PARAMS, seed)`
   gives one deterministic `p_left` per stimulus (§1.3).
2. `rng = np.random.default_rng(seed)` with `seed = cell_seed + exp_num`. With
   the array's `cell_seed = BASE_SEED + REPEAT`, repeat 1 uses seeds 2, 3, 4
   for experiments 1–3.
3. For each participant 0..39: `draws = rng.random(64) < p`, one row per
   stimulus.
   - **No individual differences.** Every participant has the same `p`.
   - **No lapse or noise term** beyond the model's own `p_left`.
   - Participants and trials are i.i.d. Bernoulli.
   - **Trial order** is the `stimuli.json` order for everyone.
   - **Left/right** is `sequence_a`/`sequence_b` as designed, never swapped.
4. The `generating_model` column is removed (`strip_generating_model`,
   holdout_data.py:214) before writing `experiment{k}/data/responses.csv` with
   `write_responses_csv`. `_require_no_generating_model_column` rechecks this
   on every path.

Columns written: `sequence_a, sequence_b, participant_id, trial_index,
chose_left` (`RAW_RESPONSE_COLUMNS`, columns.py:14). Each experiment writes
40 × 64 = 2,560 rows. `participant_id` runs 0..39 and `trial_index` runs 0..63
**in every experiment**, so after pooling (§5.1) participant 0 of experiment 1
and participant 0 of experiment 2 share an id. This only matters for a model
with a participant random effect.

---

## 5. Inner loop (one experiment)

Entry: `run_inner_model_loop_programmatic` (model_loop_runner.py:220) →
`run_pymc_inner_loop` (pymc_orchestrator.py:160). Artifacts go under
`experiment{k}/model_loop/`.

### 5.1 Data

`_pooled_response_rows` (model_loop_runner.py:38) concatenates
`experiment1..k/data/responses.csv` into `model_loop/responses.csv`. That is
2,560, 5,120 and 7,680 rows for k = 1, 2, 3. Every fit, score, critique and
novelty check in experiment k uses this pooled file.

### 5.2 Start of experiment

1. `_seed_model_set` (model_zoo.py:234) copies `cognitive_models/` (the carried
   or seeded set) into the zoo `model_loop/models/`.
2. `_resolve_protected_names` (scoring.py:32) sets the protected set to the
   project seeds present, which is the three non-GT seeds.
3. `HypothesisLedger.create` (hypothesis_ledger.py:102) copies
   `cognitive_models/attempted_hypotheses.jsonl` if present, and otherwise
   starts empty.
4. `_drop_unfittable_models` (model_zoo.py:283) runs `model_logp_is_finite`
   (pymc_inference.py:79). This checks that responses bind, and that the
   initial-point logp and its gradient are finite. Failures are dropped and
   recorded in the ledger as `dropped`. The step raises only if nothing survives.
5. `_drop_nonfinite_elpd_models` (model_zoo.py:324) is the experiment's first
   MCMC pass. `fit_models_to_cache` fits the whole set concurrently. Models
   whose fit fails or whose ELPD-LOO is non-finite are dropped (`dropped`).
6. If the threshold is > 0, `novelty_pool_rows()` generates the novelty pool
   and writes `model_loop/novelty_pool.json` (§5.9).
7. Seed scoring step: `_score`, then `_compare`, then `_record_history_step`
   with `iteration=None`. This is step 0 of `history.json`. No pruning happens
   at this step.

### 5.3 Fitting

`fit_model` (pymc_inference.py:675). Settings resolve in this order: explicit
caller value, then the model file's `SAMPLER_SETTINGS`, then `_FIT_DEFAULTS`
(`resolve_fit_settings`, pymc_inference.py:417).

| Setting | Value in the loop | Source |
| --- | --- | --- |
| draws / tune | 2000 / 1000 | config `fit` (defaults would be 4000/3000, mcmc_defaults.py:13-14) |
| chains | 4 | config, and sbatch `--chains ${CHAINS:-4}` |
| target_accept | 0.8 | config; overrides `motif_stack`'s declared 0.9 (and the 0.99 default) |
| max_treedepth | 10 | `_FIT_DEFAULTS` |
| cores | 4 | `PRODUCTION_CORES` |
| random_seed | 42 | `_FIT_DEFAULTS`: the same seed for every fit in every cell |
| log-likelihood | stored (`idata_kwargs={"log_likelihood": True}`) | needed for LOO |

That gives 8,000 posterior draws per fit.

**Concurrency** (`_fit_outcomes`, pymc_inference.py:1012). Models that need
sampling (at least 2 of them) run in a spawned `ProcessPoolExecutor` with
`allocated_cpus() // min(cores, chains)` workers. That is 16 // 4 = 4
concurrent fits. Each worker pins BLAS to 1 thread and writes the `.nc`, and
the parent loads it. Candidate admission fits run one at a time.

**Caching.** The in-process key is `(name, sha256(model.py), sha256(csv), sampler signature)`.
The on-disk file is `<cache_dir>/<name>.<fp>.nc`, where `fp` is the first 16
hex characters of sha256(model sha ‖ csv sha ‖ signature). The cell's
`cache_dir` is `$AGENT_DIR/mcmc_cache` → `RUN_DIR/mcmc_cache`. A new pooled CSV
in each experiment means every carried model is refit on it. Divergences and
R-hat > 1.01 are printed as warnings, also on cache hits. They are never gates.

### 5.4 Scoring

- **ELPD-LOO:** `FittedModel.loo_diagnostics` → `loo_diagnostics`
  (loo_reliability.py:92) → `az.loo(idata, pointwise=True)` on the per-trial
  Bernoulli log-likelihood. It is computed once per fit.
- **Softmax "posterior"** (`model_posterior`, posterior.py:201):
  `score_m = elpd_m + c · lines_m`, with `c = DEFAULT_COMPLEXITY_PRIOR_CONST = −0.05`
  (scoring.py:29) and `lines_m` the number of non-blank, non-comment lines in
  the model file. Then `posterior_m = softmax(score)`, rounded to 6 decimals.
  It raises on a non-finite ELPD. It is used only as a report field and as the
  BMA weights in evaluation (§7). It does **not** select the best model.
- **Comparison table** (`compare_table`, posterior.py:118): `az.compare` on the
  precomputed `ELPDData` (ic="loo", default stacking weights). Per model it
  records `rank`, `elpd_loo`, `elpd_diff` and `dse` (both relative to rank 0),
  `weight`, `loo_unreliable`, `n_bad_k`, `frac_bad_k`, `n_exact_loo_points`
  and `max_pareto_k`.

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

`_best_exportable_model` (scoring.py:68): among models whose
`loo_unreliable` is False, take the lowest `az.compare` rank, which is the
highest raw ELPD-LOO, with no complexity prior. It raises if no model is
reliable, or if the table is inconsistent. This one rule defines `best_model`
in `history.json`, the critique's incumbent, the incumbent-refinement target
and the exported winner.

### 5.6 A round (5 per experiment)

For `iteration` in 0..4 (pymc_orchestrator.py:322-587):

1. **Critique** of the current incumbent (§5.7). It runs before the
   candidates, sequentially.
2. **Slots.** `incumbent = history[-1]["best_model"]`. `slot_roles(6)`
   (model_zoo.py:100) gives `[explore, explore, explore, refine incumbent, refine incumbent, refine chosen]`.
3. **Lenses** (exploratory slots only). `DEFAULT_CANDIDATE_HINTS` has 12
   lenses (candidate_agent.py:48). The lens index is
   `(lens_offset + iteration·3 + e) % 12` for exploratory slot e ∈ {0,1,2}
   (`_lens_index`, model_zoo.py:140), with
   `lens_offset = (exp − 1) · 5 · 3` (`_lens_offset`, model_zoo.py:128).
   Experiment 1 therefore walks lenses 0-2, 3-5, 6-8, 9-11, 0-2. Experiment 2
   starts at lens 3 (15 mod 12), and experiment 3 starts at lens 6. No lens
   repeats within a round. Round 4 of each experiment repeats round 0's lenses.
4. **Context** for each slot (`_write_candidate_context`,
   candidate_agent.py:371). It is written to files and also **inlined into the
   prompt** (`_build_candidate_prompt`, candidate_agent.py:635). The prompt is
   `prompts/pymc_theory.md`, then the output instructions (absolute paths,
   bash heredocs only), then these sections:

   | Section | Explore slot | Refine-incumbent slot | Refine-chosen slot |
   | --- | --- | --- | --- |
   | `ATTEMPT_NOTE.md` (retry/repair only) | yes | yes | yes |
   | `CONTEXT.md`: responses path and columns, the note that no feature columns exist so `compute_features`/`prepare_observed` is required, the import allowlist, the 3-step instruction, the `check_candidate` command, a description of the other docs | yes | yes | yes |
   | `CANDIDATE_BRIEF.md` | the lens text + the one-hypothesis rule (+ critique note) | names the incumbent, its standing, hypothesis and source; lifts the anti-grafting/anti-composition rules; asks for one stated change (+ critique note) | "refine a model of your choosing" from the menu; same lifted rules (+ critique note) |
   | `existing_hypotheses.md`: every zoo model's manifest rationale, ranked by `az.compare` with "rank r, Δ ± dse nats behind (x× dse: tied/lost), ELPD" and a reliability note | yes | yes | yes |
   | `attempted_hypotheses.md`: the ledger's retired entries (§5.10) as "do not re-propose" | yes | no | no |
   | `refinement_menu.md`: live non-incumbent models ranked by standing, then ledger-pruned models, narrowest margin first, each with hypothesis and source path | no | yes | yes |
   | `critiques.md` (only if the round has a critique) | yes | yes | yes |

   The prompts never state the experimental task, i.e. what `chose_left`
   means. `problem_definition.md` is not given to inner-loop agents. The task
   has to be inferred from the existing hypotheses and the column names.
5. **Spawn.** All pending slots run concurrently (`ThreadPoolExecutor`, 6
   workers, `candidate_parallelism=None` → `candidate_count`). Each is
   `run_coding_agent` with cwd = the agent tree, a 900 s timeout, sandboxed.
   Admission then happens **sequentially in slot order** (`settle`), so a later
   slot's novelty gate compares against earlier slots admitted in the same
   round.
6. **Retry and repair per slot** (the `settle` closure, pymc_orchestrator.py:427):
   - If the agent process failed (non-zero exit or **timeout**), the attempt is
     recorded as `rejected` / "agent process failed". Nothing it wrote is
     admitted, even a valid `candidate.py`.
   - If there is no `candidate.py`, or the process failed, the slot gets one
     **retry** in `candidate_<i>_retry_1/` with `_retry_note`.
   - If a file was written but admission rejected it, the slot gets one
     **repair** in `candidate_<i>_repair_1/`. The rejection reason is quoted
     verbatim (`_repair_note`), and the rejected `candidate.py`,
     `hypothesis.md` and `model_name.txt` are copied in. A repair is final.
   - The maximum is 3 attempts per slot (original, retry, repair). Retries and
     repairs spawn together as the next wave.
7. **Empty-round guard.** If no slot was admitted and every slot's result is
   "no candidate.py written" or a failed spawn, the whole round is rerun once
   in `iter_<i>_retry_1/` (`MAX_EMPTY_ROUND_RETRIES = 1`). If that fails too,
   the round is abandoned: a ledger line `__round__ / round_abandoned` is
   written, **no history step** is recorded, and the loop moves to the next
   round. If every round is abandoned, the loop raises `AllCandidatesNoFileError`.
8. **Rescore:** `_score`, then `_prune_losers` (§5.11), then `_score` again if
   anything was pruned, then `_compare`, then `_record_history_step` with the
   round's `pruned` list and `critique` status.

The loop stops after `max_iterations` rounds. There is no early stopping or
convergence criterion.

### 5.7 CriticAL critique (critique_round.py, critique/ppc.py)

- **Incumbent:** `_best_exportable_model` of the latest scoring
  (critique_round.py:509).
- The incumbent's fit is written to the cache dir under its fingerprint
  (`_seed_critique_fit_cache`). `CRITIQUE_CONTEXT.md` is written and inlined
  into the prompt (`prompts/critique.md` + the context). The context contains
  the incumbent's name, code path and hypothesis, the responses path, its
  columns (the 5 raw columns), the zoo path, and the request for **8** files
  `test_stats/<name>.py`, each defining `test_statistic(df) -> float` with
  `# name:` and `# description:` headers. The number 8 is only requested; any
  number of usable files at or above 1 is accepted.
- The critique agent's writable directory is `iter_<i>/critique/`. The zoo and
  data are read-only.
- **Usable statistics:** `test_stats/*.py` files that pass the import
  allowlist. Offending files are deleted. If there are none, the agent is
  re-spawned once with a note (`MAX_CRITIQUE_RETRIES = 1`). If there are still
  none, the round status is `no_critique` and the candidates run without a
  critique. Any exception other than `AgentPermissionDenied` is caught and
  also gives `no_critique` with the reason.
- **Evaluation** (the pipeline runs it in-process; the agent does not):
  `run_ppc_for_model` → `evaluate_test_stat_dir` (ppc.py:408).
  - Observed frame: the pooled `responses.csv` as a DataFrame.
  - Replicates: `sample_synthetic_responses` (pymc_inference.py:582) draws
    posterior-predictive `response` over all chain×draw samples (seed 42).
    It then keeps **200** evenly strided rows (`CRITIQUE_PPC_REPLICATES`),
    and each replicate frame is the observed frame with `chose_left` replaced
    by one row.
  - For each statistic: `t_obs`, `t_null[1..200]` (30 s limit via SIGALRM),
    `n_ge = #{t_null ≥ t_obs}`, `n_le = #{t_null ≤ t_obs}`, and
    `p = min(1, 2 · min((n_ge+1)/(n+1), (n_le+1)/(n+1)))` (two-sided with the
    +1 correction). Also `z = (t_obs − mean)/sd`. If the code raises or returns
    a non-finite value, `error` is set and p is NaN.
  - **Significant = raw p ≤ 0.05** (`CRITIQUE_SIGNIFICANCE_ALPHA`), with no
    correction. A Benjamini–Hochberg q (`_benjamini_hochberg`, ppc.py:362,
    over the finite p's) and `significant_fdr` are reported alongside.
- **Outputs:** `critique/ppc_results.json` (all statistics, significant ones
  first, then by |z|) and `critique/critiques.md`, which lists **only** the
  raw-significant statistics with observed value, null mean, z, p, q and a
  "[survives FDR]" mark. `critiques.md` is inlined into every candidate prompt
  of that round, and the briefs say to prefer discrepancies that survive FDR.
- **History record:** `{"status": "critiqued", incumbent, attempts, n_statistics, n_significant, n_significant_fdr}`
  or `{"status": "no_critique", incumbent, [attempts], reason}`.

### 5.8 The candidate self-test (check_candidate.py)

`CONTEXT.md` shows the command
`<harness sys.executable> -m src.pipelines.inner_loop.check_candidate --candidate-dir <dir> --responses <pooled csv>`.
The interpreter is the venv through the opaque `$AGENT_DIR/venv` link, and the
command runs from the agent tree's code. It runs, in order: import allowlist,
load, finite logp and gradient, a smoke fit (100 draws, 100 tune, 1 chain,
1 core, no cache dir), and finite ELPD-LOO. It prints `OK` or the rejection
reason in admission's own wording. It does **not** check `hypothesis.md` or
novelty. The pipeline does not require the agent to run it.

### 5.9 Admission gates, in order

`_resolve_candidate_name` (model_zoo.py:188) runs first, then
`_admit_candidate_with_reason` (model_zoo.py:764). The first failure rejects
the candidate (`reject` records it in the ledger with the reason):

| # | Gate | Detail |
| --- | --- | --- |
| – | name | `model_name.txt` must match `[a-z][a-z0-9_]{2,40}`, must not look like `iterN_candidateM`, and must not be `inner_loop_model`/`best_model`. Otherwise the fallback `iter{i}_candidate{j}` is used, which is **not a rejection**. A name already in the zoo gets `_2`, `_3`, … |
| 1 | `candidate.py` exists | "no candidate.py written" |
| 2 | `hypothesis.md` exists and is non-empty | |
| 3 | import allowlist | AST walk, `import_gate.py`: numpy, pymc, pytensor, arviz, scipy, math, itertools, functools, collections, re, typing, dataclasses, statistics, operator. Relative imports and unparseable source are forbidden. |
| 4 | loadable | `load_pymc_model`: a module-level `model: pm.Model`, with hooks attached |
| 5 | finite logp and gradient at the initial point on the pooled responses | `model_logp_is_finite` |
| 6 | real fit | full production `fit_model` (§5.3), cached |
| 7 | finite ELPD-LOO | from that fit |
| 8 | novelty | see below; skipped if threshold = 0 |

On admission, `hypothesis.md` is copied to `models/<name>.hypothesis.md`, the
manifest gets `{name, rationale: hypothesis}`, and the ledger gets `admitted`.
**PSIS reliability is not an admission gate.** An unreliable model is admitted
and competes, but it cannot be selected as best and is shielded from pruning.

**Novelty gate** (`_min_prediction_rmse`, model_zoo.py:474):

- Pool: `novelty_pool_rows()` = `generate_candidate_pool(512, lengths=(4,5,6,7,8), seed=20260919)`
  (model_zoo.py:581-602, stimulus_design.py:45). That is 512 distinct
  same-length unordered pairs, sampled round-robin over lengths (103, 103,
  102, 102, 102). The sample is identical in every cell. It is deliberately
  not the eval pool, and it does not include length 2–3 pairs, which the
  design pool does include.
- For the candidate and for **every model currently in the zoo manifest**
  (seeds, carried models and earlier admissions including this round's; not
  pruned ones): posterior-mean `p_left` on the pool, from the production fit
  on the pooled data, averaged over all 8,000 draws (no thinning). A model
  that binds `participant_id` is averaged over the training participant ids.
  A candidate that needs other non-stimulus columns (e.g. `trial_index`) is
  rejected.
- `RMSE(c, m) = sqrt(mean_j (p̄_c,j − p̄_m,j)²)`. The candidate is rejected if
  `min_m RMSE(c, m) < 0.002`, and the reason names the nearest model.

### 5.10 The ledger (`attempted_hypotheses.jsonl`)

This is `model_loop/attempted_hypotheses.jsonl` (hypothesis_ledger.py). One
JSON line is appended per event, with exactly the keys `name, outcome, detail, hypothesis, context`.
Outcomes are `admitted`, `rejected`, `pruned`, `dropped` and `round_abandoned`.
The hypothesis is stored in full (whitespace collapsed). The context has the
form `"experiment2 round 3 candidate 1 lens 10"`,
`"… candidate 3 refine incumbent <name>"` or `"… candidate 5 refine chosen"`,
with `" retry 1"` / `" repair 1"` appended for later attempts. A prune's
`detail` is `"<Δ> nats behind <best> (<Δ/dse>× dse)"`, which is parsed back to
rank the refinement menu.

- `retired(live_names)` takes the latest entry per name that is not in the zoo.
  It is rendered as `attempted_hypotheses.md` for exploratory slots, and a
  slot's own previous attempt is left out. The `__round__` pseudo-entry of an
  abandoned round also counts as "retired" and would be rendered.
- `pruned(live_names)` is the subset whose latest outcome is `pruned`. These
  are the refinement menu's pruned targets.
- The ledger is copied to `cognitive_models/` at export and inherited by the
  next experiment.

### 5.11 Pruning (`_prune_losers`, model_zoo.py:605)

This runs after each round's scoring, never at the seed step:

1. `compare_table` on the zoo. The baseline is rank 0, the raw-ELPD best,
   whether or not it is reliable.
2. If the baseline is unreliable, **nothing is pruned** that round.
3. Otherwise a model m is pruned if it is not protected, is in the table, has
   `loo_unreliable == False`, has `dse > 0`, and has
   `elpd_diff_m > 2.0 · dse_m`.
4. Pruned files (`.py`, `.hypothesis.md`) move to `models/pruned/`. The
   in-process fit cache entry is evicted, a ledger `pruned` line with the
   margin is written, and the manifest is rewritten.

There is no stacking-weight criterion. Protected seeds stay however far behind
they are. Unreliable models are never pruned.

### 5.12 `history.json`

This is rewritten after every scoring step (`_record_history_step`,
scoring.py:122). It has one entry for the seed step plus one per
non-abandoned round, so up to 6 per experiment and 18 per cell:

| Field | Meaning |
| --- | --- |
| `step` | 0.. within the experiment |
| `iteration` | `null` for the seed step, else the round index |
| `best_model` | `_best_exportable_model` (§5.5) |
| `argmax_model` | softmax-posterior argmax (with complexity prior), for audit |
| `excluded_unreliable` | names with `loo_unreliable` |
| `posteriors` | softmax posterior (6 dp) |
| `elpd_loo` | ELPD-LOO (4 dp) |
| `pruned` | present only if something was pruned this step |
| `critique` | round steps only (§5.7) |

### 5.13 End of the inner loop: `_export` (scoring.py:197)

This writes `model_posterior.json` (posterior + `comparison` +
`excluded_unreliable`, then `best_model`), `best_model.py` and `report.md`.

---

## 6. Export and carry-forward

`_export_inner_loop_models` (model_loop_runner.py:83) rewrites
`experiment{k}/cognitive_models/` as the **live set**:

- It keeps every previous entry that is protected or still in the zoo, in its
  original order, and deletes the files of carried non-protected models the
  loop pruned or dropped.
- It appends every zoo survivor that is not already present, in zoo order,
  under its own name. A survivor with an auto name (`iterN_candidateM`) is
  exported as `inner_loop_model`, `inner_loop_model_2`, …, with the best model
  named first.
- Each entry's rationale is the model's hypothesis. The step raises on an
  empty rationale or a missing file.
- It checks that every exported model can bind a raw row.
- It copies `attempted_hypotheses.jsonl` beside the manifest.

The carried set is the three protected seeds plus every non-protected
survivor. Unreliable survivors are included: survivors are never excluded for
reliability. Then `update_registry_from_interpretation` writes the uniform
registry (§2), and the `5_model_loop` validator checks `model_posterior.json`
and `report.md`. Experiment k+1 copies this set (§2), re-seeds its zoo from it,
and refits everything on the larger pooled data. `models/pruned/` is not
carried; pruned models survive only as ledger lines.

---

## 7. Evaluation of recovery

This runs in the harness after the three experiments
(holdout_recovery.py:597-688). No agents are involved.

### 7.1 Held-out eval pool

`build_eval_stimuli` (holdout_eval.py:94) with `exhaustive=True` and lengths
1..8:

- The pool is `enumerate_all_pairs([1..8], same_length_only=True)`, which is
  43,435 pairs (the design's 43,434 plus the single length-1 pair H/T).
- Every unordered pair that appears in any `experiment{1..3}/data/responses.csv`
  is removed (`collect_trained_pairs`), at most 3 × 64 = 192 pairs.
- The step raises if fewer than 100 remain (`min_remaining`). The result is
  written to `cell_1/eval_stimuli.json`, and `n_eval_dropped` is recorded.
- `n_pairs: 500` and `seed: 11` (defaults) are unused when the pool is
  exhaustive.

### 7.2 Per-step trajectory (`evaluate_trajectory`, holdout_eval.py:296)

- GT: `q = p_left_fixed_params(gt, gt_models_src, eval, DEFAULT_PARAMS)`.
- For every `history.json` entry of every experiment (global_step 0..17):
  - `best_model`: refit with the loop's `fit_kwargs` on that experiment's
    `model_loop/responses.csv`. This is a cache hit. If the model was pruned
    later, it is loaded from `models/pruned/`.
  - Posterior-mean `p_left` on the eval pool uses `predict_max_draws = 500`,
    i.e. 125 evenly spaced draws per chain (`_eval_prediction`).
    Participant-effect models are averaged over the training participants.
  - **BMA:** each model with `posteriors > 0` at that step, weighted by its
    softmax posterior renormalised over those models (`_bma_prediction`). If
    no model has positive weight, the best model's prediction is used. The
    posterior is rounded to 6 dp, so models more than about 14 nats behind get
    weight 0.
- Metrics for both best and BMA (`recovery_metrics.py`, `recover.pearson_r`):

  | Field | Formula |
  | --- | --- |
  | `pearson_r` | Pearson r(q, p). `None` if either side has a single distinct value |
  | `rmse` | sqrt(mean (p − q)²) |
  | `kl_regret` | mean_j [ q log(q/p) + (1−q) log((1−q)/(1−p)) ], both clipped to [1e-9, 1−1e-9] (nats) |
  | `bias` | mean(p − q) |
  | `calib_slope`, `calib_intercept` | OLS of p on q |

### 7.3 Baselines

- **`baseline` (fixed-parameter seeds, no learning):**
  `seed_baseline_correlation` (holdout_eval.py:404). Each registry model other
  than the GT, at its own `DEFAULT_PARAMS`, gives a fixed `p_left` on the eval
  pool and a Pearson r with q. Output: `per_model` r and `mean_r`. No RMSE is
  computed.
- **`fitted_baseline` (fitted seeds, no agents):**
  `fitted_seed_baseline_correlation` (holdout_eval.py:488). The three non-GT
  registry models (from `pymc_model_families/`) are fit with the loop's
  `fit_kwargs` on `cell_1/pooled_responses.csv` and predict the eval pool
  (≤500 draws). Output: `per_model` r/RMSE, `mean_r`, `mean_rmse`,
  `n_responses`.
  **Bug:** `_pool_experiment_responses` (holdout_eval.py:452) concatenates
  `experiment{1..3}/model_loop/responses.csv`. Each of those files is
  **already cumulative** (§5.1), so the pooled file holds experiment 1's data
  three times and experiment 2's twice: 6 × 2,560 = 15,360 rows instead of
  7,680. The fitted baseline therefore sees duplicated trials, and
  `n_responses` is doubled. `recovery_ceiling.training_responses` correctly
  takes the last experiment's `model_loop/responses.csv` as the full training
  set.

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
`any_manifest_gt_named` is `None` if no checkout was scanned.

### 7.6 Outputs

| File | Written by | Contents |
| --- | --- | --- |
| `$WORK_ROOT/run<r>/<gt>/trajectory.json` | `_run_holdout_recovery_resolved` (holdout_recovery.py:686) | one `gt_run`: `gt_model`, `params` (the true params, hence outside the agent tree), `run_root`, `n_eval_stimuli`, `n_eval_dropped`, `trajectory[]`, `incumbent`, `baseline`, `fitted_baseline`, `leakage`, `experiments[{experiment, manifest_models}]`. Its presence makes `--resume` skip the cell. |
| `$WORK_ROOT/run<r>/<gt>/holdout.json` | script `main` | `project_id`, `seed_models_dir`, `n_experiments`, `n_participants`, `inner_loop{max_iterations, candidate_count, novelty_rmse_threshold, n_critique_proposals}`, `fit_kwargs`, `seed`, `eval_pool`, `metrics_version: 2`, `gt_runs[ … ]` |
| `holdout.csv` | `trajectory_tidy_rows` | one row per step: `TRAJECTORY_COLUMNS` + incumbent flags |
| `holdout.png` | `plot_holdout_trajectories` | trajectory figure |
| `_runs/token_usage.jsonl` + report | `start_usage_log` / `write_usage_report` | agent token spend |
| `run<r>/<gt>/agent_runs.tar.gz` | sbatch, on success | the whole `_runs/` tree. The agent tree is then deleted, unless `KEEP_REPO_COPY=1` |

---

## 8. Seeds: what varies between repeats

| RNG | Seed | Varies by |
| --- | --- | --- |
| Synthetic responses | `BASE_SEED + REPEAT + exp_num` | repeat and experiment (not GT) |
| Prior-predictive draws, EIG scenarios, posterior-predictive draws for design | 42 | nothing |
| Random design half | `exp_num` | experiment only |
| All MCMC fits | 42 | nothing |
| Novelty pool | 20260919 | nothing |
| PPC replicates | 42 | nothing |

Apart from the LLM agents, repeats of the same GT differ only in the Bernoulli
responses. Designs differ in experiments ≥ 2 because they depend on the fitted
posteriors.

---

## Doc/code discrepancies and other observations

Discrepancies, where the code wins:

1. **README.md:14** says the seed pool is "the best models discovered by three
   earlier human replicate runs". The live pool is the four literature-faithful
   models. The hero-run pool is archived in `seed_models/archive_hero_run_2026_07/`.
2. **README.md:41-44 and 218-220** say agent models "with negligible stacking
   weight are pruned", "the winner is recorded in `cognitive_models/`" and
   "az.compare's stacking weights become the model prior". In the code, pruning
   is `elpd_diff > 2·dse` with no weight floor, the whole live set is carried,
   and the registry is uniform over the carried set.
3. **CLAUDE.md, Novelty gate bullet** says a candidate is admitted only with
   `model_name.txt`. It is optional: an invalid or missing name falls back to
   `iter{i}_candidate{j}` (exported as `inner_loop_model*`). The same bullet
   omits the hypothesis-must-exist and import-allowlist gates.
4. **CLAUDE.md, Supporting modules** says `_screen_usable_models` may omit a
   model *only* for missing `participant_id`/`trial_index`. Any exception other
   than `BROKEN_MODEL_CODE_ERRORS` and stimulus-column `MissingStimulusColumns`
   also drops the model (eig.py:114-118).
5. **eig.py:201 (docstring)** says "cross-length pairs included". The code uses
   `same_length_only=True` (eig.py:241).
6. **model_zoo.py:574** says the novelty pool is "over the design's pair
   universe (same-length H/T pairs at lengths 4–8)". The design universe is
   lengths 2–8.
7. **posterior.py:142-144, scoring.py:210-212, loo_reliability.py:3-4** say an
   unreliable model is excluded from, or zeroed in, the next design's prior.
   The registry is uniform over every carried model, including unreliable ones.
8. **pymc_orchestrator.py:6-8 (module docstring)** says the softmax posterior
   "selects the incumbent". Selection is by `az.compare` rank among reliable
   models.
9. **holdout_recovery_array.sbatch:25-27 and model_loop_runner.py:55-57** say
   agents run with "no read sandbox". Every loop agent now runs in bubblewrap
   (`sandbox=True`).
10. **Faithful config, `agent.backend` comment** says "null -> CODING_AGENT env
    var, then 'claude'". The code default is `opencode` (coding_agent.py:57).
    Under the array the config key is overridden anyway by
    `--backend ${AGENT_BACKEND:-opencode}`.
11. **Faithful config `seed: 7`** is never used under the array, because the
    sbatch always passes `--seed BASE_SEED+REPEAT`.
12. **scripts/subjective_randomness/holdout_recovery.py:4-6** mentions "real
    theory, design, and candidate-conjecturing agents". Only critique and
    candidate agents exist.
13. **holdout_recovery_array.sbatch:150-176** stubs `model_families/<gt>.py`,
    but `agent_tree.exclude` already removes `model_families/`, so the stub is
    never written. This is harmless.
14. **candidate_agent.py:44-45** says twelve lenses let a round walk "four
    rounds without repeating". With 5 rounds, each experiment's round 4 reuses
    round 0's lenses. The config comment's weaker claim (no repeat *within* a
    round) holds.

Bugs and behaviour worth a decision:

- **Fitted-seed baseline double-counts data** (§7.3). It pools already-cumulative
  files, giving 15,360 rows instead of 7,680. The fix is to use the last
  experiment's `model_loop/responses.csv`, or `experiment*/data/responses.csv`.
- **Design uses only the previous experiment's data** (§3.3): each model is fit
  on `experiment{k-1}/data/responses.csv`, not the pooled data the inner loop
  scored on. The docstring states this, so it may be intended. It is
  asymmetric with the inner loop.
- **Design-time target_accept** is 0.99 (or 0.9 for motif_stack), not the
  sweep's 0.8, because design fits pass no `target_accept`.
- **EIG assumes one response per stimulus**, while the data have 40
  same-parameter participants per stimulus.
- **The random half is nearly identical across cells and repeats** (seeded by
  `exp_num`), and no stimulus is excluded across experiments.
- **Left/right is fixed** by enumeration order, with the lexicographically
  smaller string always on the left.
- **Timed-out agents are discarded** even if they wrote a valid candidate
  (§5.6.6).
- **participant_id restarts at 0 in each experiment**, so pooled
  participant-effect models conflate participants across experiments.
- **Agents are not told the task.** `problem_definition.md` never reaches
  inner-loop prompts. I searched `src/pipelines/inner_loop/` and found no
  statement of what `chose_left` means.
- **The `__round__` ledger pseudo-entry** is rendered as a "retired hypothesis"
  in `attempted_hypotheses.md` after an abandoned round.

Things I did not verify at runtime (read from code only): arviz's `good_k`
value at 8,000 draws (assumed 0.7), `az.compare`'s default weight method
(stacking), and whether `pm.sample_prior_predictive(draws=1)` of `p_left`
under `pm.do` is exactly deterministic for every GT. It should be, since all
free RVs are fixed and `p_left` is a Deterministic of them and the data.
