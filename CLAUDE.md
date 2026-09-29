# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

An automated cognitive-science discovery pipeline. Coding agents conjecture PyMC
models of human judgment (currently *subjective randomness*), design maximally
informative experiments, run them on real (Prolific + Firebase) or simulated
participants, and let Bayesian model comparison decide which hypothesis survives.
`README.md` is the authoritative user-facing guide; this file captures the
architecture and workflow facts that require reading multiple files.

## Environment & commands

`uv` drives everything. The default sync installs the `dev` **and** `pymc`
groups (see `pyproject.toml` `default-groups`) — the test suite and inner loop
need the PyMC stack, so a plain `uv sync` is enough; `open-models` (torch/
transformers) is heavy and opt-in.

```bash
uv sync                      # dev + pymc (everything the tests need)
uv run pytest -q             # full suite incl. real-MCMC tests (~2 min)
uv run pytest -q -m "not slow"          # fast suite (~30 s), skips NUTS sampling
uv run pytest tests/test_eig_selection.py::test_name   # single test
./scripts/test               # wrapper: full suite (run before committing)
./scripts/test_fast          # wrapper: fast suite (run after each change)
```

- The `slow` marker means "runs PyMC NUTS". Deselect it for fast iteration.
- CI (`.github/workflows/tests.yml`) runs `uv sync --locked`, then
  `tests/test_python_sources_compile.py` (byte-compiles every `.py`), then the
  fast suite. There is **no committed linter/formatter config** (the `.ruff_cache`
  is from ad-hoc runs); match surrounding style.
- **Do not move off Python 3.11** (`.python-version`). 3.12+ resolves arviz 1.x /
  pymc 6.x, whose API rename breaks the model-comparison code. The numpy/scipy/
  pandas version caps in `pyproject.toml` exist so wheels resolve on Sherlock's
  glibc 2.17 nodes — don't drop them unless you only target glibc ≥ 2.28.

## Conventions (from the user's global instructions — they apply here)

- Filepaths via `pyprojroot.here()`; CLIs are `tyro` dataclasses (`--help` lists
  every knob). Every entry point is `python -m src.<...>` run under `uv run`.
- **Fail loudly.** This codebase deliberately raises on missing models, corrupt
  registries/artifacts, absent credentials, etc. rather than silently falling
  back. Preserve that — never add a silent default or fallback.
- Follow BDD dual-loop TDD: a failing integration test first, then inner
  red-green-refactor. Run the relevant test after each green step, the full
  suite before a commit-worthy checkpoint.

## Architecture — the mental model

Two nested loops. **The single most important fact: stages are decoupled
processes that pass state through on-disk artifacts (CSV / JSON / YAML / `.py`
files), not in-memory objects.** Every cognitive model is a self-contained `.py`
file discovered via a `models_manifest.yaml`; MCMC fits are content-addressed and
cached; the only state crossing an experiment boundary is (a) the carried model
files (the live set), (b) the ledger of attempted hypotheses beside them and
(c) the registry (a uniform prior over the carried set).

### Outer loop — `src/pipelines/outer_loop/` (`run.py` → `orchestrator.py`)

Per experiment, stages run in order. `AGENT_KEYS = ["2_design", "3_implement",
"4_collect", "5_model_loop"]` — **there is no stage 1** (the theorist and design
agents were removed; the numbering is historical). Before the stages, a
pseudo-stage establishes the model set: experiment 1 seeds from
`projects/<id>/seed_models/`; experiments ≥2 `carry_forward_cognitive_models`
from the previous experiment. Both seed/carry steps are idempotent, which is what
makes `--resume` (run into an existing `experimentN/` dir) and `--agent <stage>`
(run one stage) safe.

- `2_design` — **programmatic, no agent.** `eig.design_exhaustive` enumerates the
  H/T pair space (every same-length pair, lengths 2–8) and greedily selects the
  max-joint-EIG stimulus set (64 by default) → `design/stimuli.json`. The EIG
  counts every participant's response: a stimulus yields k ~ Binomial(n, p_left)
  with `n_responses` = the experiment's participant count (required at every
  entry point). This joint EIG saturates after a few picks: selection stops
  once the best gain is within two Monte Carlo standard errors of zero (or
  below `NEGLIGIBLE_GAIN_BITS`), and the remaining slots are filled by
  single-response EIG conditioned on the picks so far (`source`
  `eig_single_response_fill` in `stimuli.json`). Experiment 1 uses prior-predictive draws; experiments ≥2 fit
  the models on all data so far (the previous experiment's cumulative
  `model_loop/responses.csv`) at the model's declared `target_accept`, else
  `DESIGN_TWIN_TARGET_ACCEPT` (0.9). The holdout harness derives every seed
  from (cell seed, ground truth, experiment, purpose) (`derive_seed`). The
  greedy search is **lazy batched greedy in float32 on every allocated CPU**
  (`DESIGN_LAZY_SEARCH`, `DESIGN_SCORING_DTYPE` in `eig.py`): a full pass over
  the pool every 16 picks, in between only the best-ranked candidates
  re-scored in batches of 512. It is an approximation (joint EIG is not
  submodular), validated against exact float64 greedy on two experiment-2
  designs (`scripts/subjective_randomness/validate_lazy_eig.py`: within exact
  greedy's own scenario-to-scenario spread) — ~3 min instead of 11-13 h.
  Each Monte Carlo scenario's likelihood average **leaves out the draw that
  generated it** (`DESIGN_LEAVE_ONE_OUT`, first audit C5): including it
  rewarded the true model for memorising its draw and inflated the joint EIG,
  the noise-floor stop and `joint_eig_bits` (by ~0.02-0.1 bits on the two
  validation designs). Same cost; the estimate is consistent but not
  unbiased (finite draw average); a model needs ≥ 2 draws.
- `3_implement` — the one true coding-agent stage: writes a jsPsych experiment;
  skipped in `simulated_participants_nobrowser` mode. Optional Firebase/Prolific
  deploy phase follows when `--deploy-target != none`.
- `4_collect` — programmatic: writes `data/responses.csv` (simulated / LLM-as-
  participant / live / ground-truth).
- `5_model_loop` — drives the inner loop, then writes `model_registry.yaml`.

Coding stages that fail their validator are re-spawned with the error injected as
repair feedback (`--max-validation-repairs`); programmatic stages fail terminally.
Validators (`_validate_*` in `orchestrator_validators.py`) enforce the artifact
contracts (e.g. jsPsych button-only + `chose_left` data column).

### Inner loop — `src/pipelines/inner_loop/`

`pymc_orchestrator.py` orchestrates; `model_zoo.py` manages seeding, admission,
pruning and the novelty gate; `scoring.py` handles ELPD-LOO scoring, best-model
selection and export; `candidate_agent.py` writes candidate briefs and spawns
agents. The model zoo lives at `model_loop/models/`; the project's seeds are
`protected_names` (never pruned — the outer loop passes them explicitly, so a
model carried from an earlier experiment *can* lose and leave). Each round:
optional CriticAL critique → spawn candidate agents in parallel (exploratory
slots steered by a rotating exploration "lens", refinement slots by a named
target — see **Slot roles**) → admit sequentially.

- **Admission and the novelty gate** (`_admit_candidate` in `model_zoo.py`): a
  candidate is admitted only with a loadable `candidate.py` (module-level
  `model: pm.Model`) that passes the code gate (`import_gate.py`: an import
  allowlist, and no `open`/`np.load`/`eval`/`__import__`/`getattr`/dunder
  escapes, no forbidden module reached as an attribute (`typing.sys`,
  `dataclasses.builtins`), no file reader (`np.DataSource`, `pd.read_*`) and
  no attribute lookup inside `str.format` — the code runs in the harness
  process, which clears its own argv once parsed) + `hypothesis.md` (`model_name.txt` is
  optional; a slot name is the fallback), passing logp/real-fit/finite-ELPD
  gates and the **convergence gate** (≤0.1% divergent transitions, R-hat ≤
  1.05, bulk ESS ≥ 100; `fit_model` refits a failing fit once at
  `target_accept` 0.95, with a random seed of its own derived from the first
  fit's seed and fingerprint (`refit_settings`), only when it is a near miss
  — ≤2% divergent, R-hat ≤ 1.2, bulk ESS ≥ 20, `NEAR_MISS_*` in
  `mcmc_defaults.py` — and uses that fit everywhere, while a fit far from converging is returned as is; a
  model's declared `target_accept` is a floor on the loop's — so the rejection
  and the brief advise reparameterising, never smaller steps), within the
  **admission time limit** (each sampling run of a candidate's fit is killed,
  chains and all, after `CANDIDATE_FIT_TIME_LIMIT_SEC` = 15 min, and the
  candidate is rejected as too slow to fit; seeds and carried models are never
  limited), AND with posterior-
  mean `p_left` ≥ `novelty_rmse_threshold` (0.002) RMSE from every admitted
  model **on the loop's novelty pool** — 512 same-length H/T pairs at lengths
  4–8 that the loop generates from its own seed (`novelty_pool_rows`) and
  records as `model_loop/novelty_pool.json` — not on the training stimuli.
  Measured on the 64 training stimuli at 0.02, the September 2026 sweep's 23
  rejection margins were bimodal: ~5 re-skins at ~0 and ~18 distinct
  mechanisms spread from 0.006 to the threshold that merely agreed on the
  training points. The pool is deliberately not the recovery harness's eval
  pool: the loop must not select models on the stimuli it is scored against.
  A candidate that binds `participant_id` is marginalised over the training
  participants; one that binds `trial_index` is rejected with that reason
  (it cannot be evaluated on any stimulus pool), and so is one whose `p_left`
  is undefined (NaN or outside [0, 1]) on any pool stimulus
  (`NoveltyPoolUndefined`, the reason names example pairs). An admitted model
  undefined on some pool stimuli is compared on the rest (and left out when
  undefined on all). The threshold is a knob of
  the holdout config (`inner_loop.novelty_rmse_threshold`, CLI
  `--novelty-rmse-threshold`, sbatch `NOVELTY_RMSE_THRESHOLD`).
- **Slot retry and repair** (`_Slot` in `pymc_orchestrator.py`): a round is
  spawn → prefit (concurrent candidate fits) → settle waves. A slot whose
  agent wrote no `candidate.py` (or whose agent process failed) is
  re-spawned once in `candidate_<i>_retry_1/`; a
  candidate that `_admit_candidate_with_reason` rejects is re-spawned once in
  `candidate_<i>_repair_1/` with the rejection reason verbatim in its prompt
  (`_repair_note`) and the rejected files copied in — a repair is always
  final and never counts as an unfilled slot. Every attempt is a ledger line
  (`… retry 1` / `… repair 1` in its context). The all-slots-empty round
  retry (`MAX_EMPTY_ROUND_RETRIES`) stays as the outer guard. `CONTEXT.md`
  documents a self-check command (`check_candidate.py`; `CANDIDATE_CHECK_*`
  in `mcmc_defaults.py`) that runs the admission gates with a smoke fit.
- **Slot roles — breadth and depth** (`slot_roles` in `model_zoo.py`): a
  round's `candidate_count` slots have roles. With four or more, `C - 3`
  exploratory slots walk the twelve-lens battery (`DEFAULT_CANDIDATE_HINTS`;
  only exploratory slots advance the walk, so `_lens_offset` counts them),
  two slots **refine the incumbent** (the latest history step's
  `best_model`, named in the brief with its hypothesis, standing and source
  path) and one **refines a non-incumbent model of the agent's choosing**
  from `refinement_menu.md` — the live non-incumbent models ranked by
  standing and the ledger's pruned models ranked by margin
  (`parse_prune_margin` reads the margin `_prune_losers` wrote), each with
  its full hypothesis and source (`models/<name>.py`, or for a pruned model
  the `model_loop/models/pruned/<name>.py` of the experiment that pruned it,
  found through its ledger context), framed as a menu. Below four slots the
  refinement slots go one at a time: at 3 one of each, at 2 exploratory +
  incumbent, at 1 exploratory only. Refinement briefs lift the anti-grafting
  and no-composition clauses (exploratory briefs keep them and the "do not
  re-propose" list). The role is the slot's *assignment*, recorded in the
  ledger context (`… candidate 1 refine incumbent <name>`, `… candidate 2
  refine chosen`); which model the agent refined is stated in its
  `hypothesis.md` in prose and **never parsed** — no target file, no regex,
  no ledger field, no novelty-gate exemption. Motivation: 0 of 27 incumbent
  changes in the September 2026 sweep, with three breadth mechanisms and no
  depth mechanism. The holdout configs run **five rounds of six** per
  experiment (`max_iterations: 5`, `candidate_count: 6`; the sweep that
  motivated the plan ran two rounds of three). A round's agents run
  concurrently, one worker per slot (`candidate_parallelism`, default
  `candidate_count`); retry and repair attempts spawn concurrently too.
- **Pruning** (`_prune_losers` in `model_zoo.py`) runs **once, at the end of each
  experiment**: non-protected models with a trusted fit (reliable PSIS-LOO and
  converged) that are statistically distinguishable from the best
  (`elpd_diff > dse_multiplier·dse_clustered`, the stimulus-clustered SE of
  `src/models/clustered_se.py`: the trial-level `dse` treats the correlated
  responses to one pair as independent and is ~2x too small) move to
  `models/pruned/`. Then
  `_cap_live_set` keeps at most `MAX_LIVE_MODELS` (8) live models: untrusted
  fits retire first, then the lowest by ELPD-LOO; seeds never. There is no
  stacking-weight floor (stacking weights are ensemble coefficients, not
  plausibility). A pruned mechanism may come back with a substantive change:
  the ledger forbids only unchanged copies and near-duplicates.
- **Ledger** (`src/pipelines/inner_loop/hypothesis_ledger.py`):
  `model_loop/attempted_hypotheses.jsonl` records every candidate slot
  (admitted / rejected, with the reason) and every prune (with the margin),
  continues the ledger the previous experiment carried, and is rendered into
  every candidate brief as `attempted_hypotheses.md` ("already tried — do not
  re-propose": the retired hypotheses, i.e. those no longer in the set).
  Without it, pruned hypotheses vanished from `existing_hypotheses.md` and were
  re-proposed (in the weakest recovery cell 11 of 13 re-proposals had already
  been pruned there). Hypotheses are stored in full — whitespace collapsed,
  never truncated — and rendered one heading per retired model rather than as
  a table, so a multi-sentence hypothesis reaches the next round's brief
  intact. `LedgerEntry` must not gain fields: `from_json` requires an exact
  key-set match, so a new field makes every inherited ledger unreadable.
- **Export**: the exported winner (`_best_exportable_model` in `scoring.py`) is the best model
  by **ELPD-LOO rank** (`az.compare`'s `rank`) among those whose PSIS-LOO is
  *reliable* — never the softmax posterior argmax: the posterior is rounded to
  six decimals, so every model more than ~14 nats behind reads 0.0 and a
  `max` over it picked by manifest order (62 of 230 baseline experiments
  exported a far-behind seed that way). The same rule selects the per-step
  `best_model` in `history.json` (which also records `argmax_model` and
  `excluded_unreliable`) and the critique incumbent, so what the recovery
  harness scores, what the critic critiques and what is carried agree. The
  outer loop (`_export_inner_loop_models` in `model_loop_runner.py`) then makes `cognitive_models/` the
  **live set**: the protected seeds plus every zoo survivor (not only the
  winner — a rival within 2·dse is carried and left to the next design), with
  a carried model the loop pruned removed, and the ledger copied beside the
  manifest. Before this, only the winner crossed the boundary: 19 unresolved
  rivals were dropped at 40 boundaries in the iteration-2 recovery sweep.
  "Reliable" is `src/models/loo_reliability.py`'s verdict (a tolerated
  proportion of high-Pareto-k trials, with constant-log-likelihood trials
  exempt as exact), **not** arviz's blanket any-k>0.7 flag — that flag fired on
  clipped `p_left` trials and was silently discarding genuine winners.

### How weights flow between experiments (the registry)

`src/registry/io.py` schema: `{theories: {name: prob}, reserved_for_new}`. After
experiment N, `update_registry_from_interpretation` writes a **uniform prior
over the carried set** (the `cognitive_models/` manifest the export just
wrote) to `model_registry.yaml`; experiment N+1's design reads it as the model
prior for EIG selection. It used to copy `az.compare`'s stacking weights, which
are ensemble coefficients rather than plausibility and were written over the
whole zoo: in 15 of 40 next-experiment designs of the iteration-2 recovery
sweep every model actually present had weight ~0 (or one had 1.0), so all 32
EIG-selected stimuli had zero EIG. The stacking weights remain a report field
in `model_posterior.json`. Model *files* flow separately via carry-forward.

### Supporting modules

- **Agent isolation** (`src/runtime/agent_sandbox.py`, holdout agent trees):
  loop agents run stock and in bubblewrap (every job's `_env.sh`, live and
  simulation, runs `ml load system bubblewrap` and stops without `bwrap`) —
  their tree read-only, only their own candidate/critique dir, the run's notes, a scratch dir at /tmp and a
  private home writable, and an allowlisted environment
  (`agent_environment`: system basics, the compiler toolchain, network
  settings and the backend's own login — no other `.secrets` key, no
  `SLURM_*`, no sweep variables). Agent trees contain only
  `src/` and the run tree (`agent_tree.exclude`: no docs, tests, scripts,
  research library, project literature or `.secrets`). Before agents start,
  `scan_gt_name.sh --before-agents` stops a cell whose tree names its held-out
  ground truth; after a run it only warns (`gt_name_mentions.txt`), and
  `agent_activity_report.py` lists the URLs agents looked at and any paths
  outside their directory. Inner-loop agents are told the task from the
  project's `task_description.md`.

- **Screening a model out of a design is recorded.**
  `eig._screen_usable_models` omits a model whose only unbindable columns are
  response-row bookkeeping (`NON_STIMULUS_COLUMNS`: `participant_id`,
  `trial_index`) — a participant-level random effect, say — and also one whose
  probe raises any other non-code error (recorded with the error); broken model
  code raises. Missing stimulus columns raise too: that means the design rows lack columns the model
  needs, and dropping the model would renormalize EIG over whichever ones
  happen to bind. `make_stim_data` signals this with `MissingStimulusColumns`,
  which carries `.missing` as data so callers classify structurally rather than
  by re-parsing a message. A model whose predictive `p_left` is undefined (NaN
  or outside [0, 1]) on some design-pool pairs (`InvalidPredictions`, prior or
  posterior) is also left out of that design — it used to crash the cell, on
  every retry — recorded with `invalid_pairs` and a reason naming example
  pairs; `verify_holdout_run.sh` warns on those rather than failing. Every
  drop is written to `design/screened_out.json`
  (empty list = the screen ran and dropped nothing), so a silent shrink of the
  hypothesis set is visible in the run tree.
- **Raw-only pipeline.** There is no featurizer: `responses.csv` carries only
  the five `RAW_RESPONSE_COLUMNS` (`sequence_a`, `sequence_b`, `participant_id`,
  `trial_index`, `chose_left`, defined in `src/pipelines/outer_loop/columns.py`).
  Every model computes its own features via a `compute_features(sequence_a,
  sequence_b)` or `prepare_observed(rows)` hook. Collection keeps only these
  in `data/responses.csv` (`raw_response_rows`) and the full collected rows
  (`/results` adds `participant_id_str`, the Prolific ID) in
  `<project>/raw_collected/`, beside the experiment dirs, which no agent is
  given (`raw_collected_responses_path`); pooling keeps only them too.
  `pymc_model_families/` (and
  the live pool `seed_models/`) all carry `compute_features` hooks. The verifier
  (`scripts/subjective_randomness/slurm/verify_holdout_run.sh`) checks that
  every agent-facing CSV in a finished run carries only raw columns. The
  agents' prompts say so too: `prompts/pymc_theory.md` and
  `prompts/critique.md` describe only the raw columns (its skeleton computes
  its features with `compute_features`), the candidate brief raises on a CSV
  with any other column, and `tests/test_agent_prompts_raw_only.py` fails if
  an agent-facing text presents a feature column as data (first audit, D9).
- `src/models/mcmc_defaults.py` — the **single source of MCMC sampler defaults**
  (`PRODUCTION_*`, `DESIGN_TWIN_*`). Every entry point imports from here; change
  defaults only here.
- `src/models/model_loading.py` — loads agent-written `.py` files; requires a
  module-level `model: pm.Model`. Attaches optional data-binding hooks
  (`compute_features`, `prepare_observed`, mutually exclusive).
  `src/models/data_binding.py` maps CSV rows / row dicts to `pm.set_data` dicts.
  `src/models/pymc_inference.py` adds fitting, prediction, caching and
  diagnostics. Fits are cached on `(model sha, csv sha, sampler sig)` in-process
  and on disk (`<name>.<fingerprint>.nc`). `fit_models_cached` samples the
  models that need MCMC concurrently: a spawned `ProcessPoolExecutor` whose
  workers (`fit_workers`; by default as many as `workers × cores-per-fit`
  fits in the CPUs Slurm allocated, `allocated_cpus`) each fit one model
  with BLAS pinned to one thread and persist its `.nc`, which the parent
  loads. `fit_models_to_cache` is the tolerant sibling (a model's own
  failures reported by name) that the experiment-start ELPD screen
  (`model_zoo._drop_nonfinite_elpd_models`) uses to sample the whole set in
  one batch. A wave's candidates are fitted concurrently before their
  sequential admission (`model_zoo.prefit_candidates` →
  `fit_time_limited_concurrently`: every candidate past the cheap gates, under
  its predicted admission name, each run in its own time-limited child, as
  many at once as the CPUs hold); admission then loads those fits (or their
  remembered failure/timeout), so its verdicts are sequential admission's. **An infrastructure failure is never a
  model's**: a broken pool (one worker killed breaks every pending fit), an
  unreadable `.nc`, `OSError`/`MemoryError` (`INFRASTRUCTURE_ERRORS`,
  `FitInfrastructureFailure`) raise everywhere — never a drop, a rejection or
  a ledger line. The screen never drops a protected seed (it raises), and
  `.nc` files are written to a temporary name and `os.replace`d into place
  (`write_fit_file`). Every fit process (pool worker or time-limited child)
  gets an `XDG_CACHE_HOME` of its own under a temporary root the parent
  removes (`_fit_process_caches`): arviz writes a once-a-day marker there on
  import through a fixed-name temporary file, and fit processes started
  together after midnight collided on it (11 of 24 cells, 2026-09-28).
- `src/model_comparison/{posterior,likelihood}.py` — ELPD-LOO softmax posterior
  (`model_posterior`, documented as overconfident) + `az.compare` PSIS-LOO table.
- `src/critique/ppc.py` — CriticAL posterior-predictive check: agent-written
  `test_statistic(df) -> float` scored against posterior-predictive replicates,
  two-sided empirical p + BH-FDR q; results steer the next candidate round.
  The critique agent's context is inlined into its prompt
  (`critique_round._build_critique_prompt`); an agent that writes no usable
  statistic is retried once, then the round runs with **no** critique and the
  round's `history.json` entry records `"no_critique"` — as does a round in
  which no statistic produced a p-value (each errored or ran out of time;
  the reason lists why). Each call of a statistic has a 5 s limit and all of
  its 1001 calls a 300 s budget (`_TEST_STAT_CALL_TIMEOUT_SEC`,
  `_TEST_STAT_BUDGET_SEC`); one 30 s limit over all calls used to time out
  ordinary statistics at 7,680 rows. There is no
  pipeline-written fallback battery — one existed and, under the raw-only
  schema, reduced to the marginal choice rate, which hid a critique subsystem
  that had never produced a statistic through an entire sweep.
  `verify_holdout_run.sh` warns when no round of an experiment was critiqued.
  The critique agent runs **before** the round's candidate agents, not
  alongside them: its `critiques.md` is inlined into every candidate brief,
  so running it concurrently would hand the candidates a stale critique (of
  the previous incumbent) or none — measured cost 1–4 min of agent time per
  round. `inner_loop.n_critique_proposals` (default 8) is a holdout-config
  knob (CLI `--n-critique-proposals`, sbatch `N_CRITIQUE_PROPOSALS`).
- `src/runtime/coding_agent.py` — backend-agnostic agent launcher.
  `run_coding_agent(...)` is the single call site; `select_backend` resolves
  explicit arg → `CODING_AGENT` env → `opencode` default. `claude` uses
  `--dangerously-skip-permissions --add-dir`; `opencode` uses `opencode run`
  (no `--add-dir`). Token usage is always recorded.
- `src/subjective_randomness/incumbent.py` — the **incumbent record**, the
  loop-improvement plan's primary metric: per scoring step of a holdout cell,
  did the exported `best_model` change from the previous step, and is it a
  *discovered* model (not scored at experiment 1's seed step, i.e. not one of
  the project seeds the cell started with). The harness writes the two flags
  onto every `trajectory.json` row / `holdout.csv` column and a per-cell
  `incumbent` summary block; `scripts/subjective_randomness/incumbent_report.py`
  reports it over a finished sweep (archives or kept repo copies), and
  `verify_holdout_run.sh` warns (never fails) on a cell with zero changes.
  Baseline: 0 changes over 27 steps in the three complete `motif_stack` cells
  of the September 2026 sweep.

### Projects vs. the research library — two different things

- `src/pipelines/outer_loop/projects/<id>/` = **assets the generic pipeline
  consumes** (`problem_definition.md`, `ground_truth_models.py`, `seed_models/`,
  `evaluate_recovery.py`, `prolific_config.yaml`). Adding a project = adding an
  asset directory. This lives under `src/`, not the run-output `projects/` tree.
- `src/subjective_randomness/` = a **standalone research library** for the
  subjective-randomness domain (model families, `stimulus_design.py`,
  `sequence_stats.py`, recovery harnesses). Coupling to the pipeline is
  deliberately thin (two cross-imports). `pymc_model_families/` is the frozen
  recovery registry; the project's `seed_models/` manifest mirrors it (a test
  asserts they agree, byte for byte: the `motif_stack` seed is the Viterbi
  model that is also the ground truth; a softmax seed was tried and reverted
  on 2026-09-27 because it failed the convergence gate on two of the three
  ground truths' data). The seed baselines load the seeds' code from the run
  tree (`holdout_eval.seeded_models_dir`), not from the registry, so a run
  is re-scored with the seeds it ran with.

### Inspecting results

- `src/viewer/` — Flask + static SPA explorer over finished runs on disk
  (`python -m src.viewer.server`); `freeze.py` snapshots curated runs to a static
  site. The tree is re-walked per request (no build step).
- `src/monitor/` — live dashboard for an in-progress human study (Firestore +
  Prolific), discovered from `deployment_manifest.json` files. Its first job is
  catching degenerate data (participants answering one side every trial).

### Legacy / not part of the live loops

`src/experiments/` and `src/validation/` are from the old pipeline and are not
used by the active loops (per `README.md`).

## Cluster & live runs

- Holdout sweeps resume their own failed cells: after each array a retry job
  (`holdout_retry.sbatch`, via `cell_status.py`) resubmits timeouts, crashes
  and out-of-memory tasks (the latter with `--mem=128G` on their array only)
  with `--resume`, capped at `%MAX_PARALLEL`, up to `MAX_RETRY_ROUNDS` (2);
  the summary job lists any expected cell still without a result in
  `MISSING_CELLS.txt`. A sweep runs on **one code**: the setup job stages
  `harness_repo` and `agent_src` once and records `$WORK_ROOT/code_commit`;
  retries skip setup and run the staged scripts; every cell records the code
  it started on and refuses to resume on other code. `verify_holdout_run.sh`
  judges cells by `holdout.json` / `MISSING_CELLS.txt`, not by task logs.
- **Live runs recruit real participants and spend real money.** They are double-
  gated: the config needs `confirm_live_recruitment: true` **and** `run.py`
  enforces `--confirm-live-recruitment`; the launchers print a cost summary and
  require typing `yes`. See `scripts/outer_loop_live/README.md`. `scancel` kills
  the pipeline job but **not** an already-published Prolific study — stop that in
  the Prolific dashboard.
- **One live study per experiment.** The live job always passes `--resume`, so
  a relaunch used to redesign, redeploy and publish a second study. Now an
  experiment whose `deployment/deployment_manifest.json` records a live
  `prolific_study_id` (published or not confirmed) refuses `2_design`,
  `3_implement` and the deploy (`refuse_second_live_study` in
  `deployment/manifest.py`, checked in `run.py` and `run_deployment`), raising
  `LiveStudyAlreadyRecorded` with the recovery: `RESUME_AGENTS=4_collect:5_model_loop`,
  or the deliberate `--publish-another-prolific-study`
  (`PUBLISH_ANOTHER_PROLIFIC_STUDY=1`), which archives the old manifest.
- Secrets live in repo-root `.secrets` (see `.secrets.example`): `PROLIFIC_API_TOKEN`,
  `FIREBASE_TOKEN` (`firebase login:ci`), `AUTO_PSYCH_RESULTS_TOKEN` (guards the
  `/submit` & `/results` Cloud Functions — deploy/collect fail loudly without it),
  `GOOGLE_API_KEY` (simulated/Gemini paths only).
- On Sherlock: never run heavy work on the login node (submit via Slurm), keep
  job I/O on `$SCRATCH`, and note Playwright browser simulation does **not** run
  on the compute nodes (glibc 2.17) — use `--mode simulated_participants_nobrowser`.
  Recovery-harness runbooks: `scripts/subjective_randomness/README.md` and its
  `slurm/README.md`.
