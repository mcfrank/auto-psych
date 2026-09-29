# How the loop works

One experiment, step by step, with where the code is. Why each piece exists
is in the [brief](BRIEF.md) § 2. The stage names (`2_design`, …) are the ones
the code, the logs and the `--agent` flag use; there is no stage 1.

## The data

A trial is a pair of H/T sequences of the **same length** (2–8). Every
response is one row with five columns (`src/pipelines/outer_loop/columns.py`):

| column | meaning |
|---|---|
| `sequence_a`, `sequence_b` | the pair **as shown**; `sequence_a` was on the left (the page randomises the side per trial) |
| `participant_id` | an integer, unique across the run's experiments |
| `trial_index` | trial number within the participant |
| `chose_left` | 1 if the left sequence was chosen |

A model is one `.py` file with a module-level `model` (a `pm.Model`), a
`p_left` per trial, and a Bernoulli likelihood on `chose_left`. It computes
its own features with `compute_features(sequence_a, sequence_b)` or
`prepare_observed(rows)`. Task assets, including the wording participants see,
are in `src/pipelines/outer_loop/projects/subjective_randomness/`.

## Step 0: the model set

- **Experiment 1** copies the four starting models from `seed_models/`
  (`seed_experiment_models_from_project` in `orchestrator.py`). The run records
  them in `starting_models.json`, beside the experiment folders, with
  `starting_models_prunable: true`. No proposal may take their names, but
  they can be removed like any model (step 4).
- **Later experiments** copy the previous experiment's `cognitive_models/`
  (`carry_forward_cognitive_models`), only if that experiment's model stage
  finished (its `model_loop/export_complete.json` validates).

## Step 1: choose stimuli (`2_design`, no agent)

`run_design_programmatic` (`orchestrator.py`) → `design_exhaustive`
(`src/pipelines/outer_loop/eig.py`) → the estimator in
`src/models/eig_selection.py`.

- **Candidates:** every distinct same-length pair, lengths 2–8 (43,434).
- **Score:** joint EIG, the expected drop in uncertainty (bits) about which
  model is right. The pipeline pretends each model in turn is true, draws
  parameters (200 per model), simulates N participants' answers to each pair
  (a count of "left" choices, Binomial(N, `p_left`)), and averages over 1,000
  such scenarios. The model prior is uniform over the models carried in.
- **Selection:** 64 pairs, one at a time, each maximising the whole set's
  EIG. When the best gain is within two Monte Carlo standard errors of zero,
  the remaining places go to the pairs most informative for one answer
  (marked `eig_single_response_fill` in `design/stimuli.json`).
- **Priors or posteriors:** experiment 1 uses the models' priors. Later
  experiments first fit each model to all data so far (500 draws, 2 chains)
  and use the posteriors.
- **Speed:** the search re-scores only the most promising pairs between full
  passes, in single precision (minutes instead of hours). A scenario's
  likelihood leaves out the draw that generated it, so the true model is not
  flattered.
- **Models left out:** a model whose `p_left` is undefined or whose own code
  fails on some pairs is left out of that design only, and listed in
  `design/screened_out.json` (written even when empty).

## Step 2: build and deploy the page (`3_implement`)

A coding agent follows the fixed template in
`src/pipelines/outer_loop/prompts/3_implement.md` and writes
`experiment/index.html` (jsPsych, two buttons, side randomised). A validator
checks it; with `--validate` a failure goes back to the agent up to
`--max-validation-repairs` (default 2) more times.

With `--deploy-target firebase` (`run_deployment` in
`src/pipelines/outer_loop/deployment/local.py`), in this order: the results
token is checked, and in `live` mode Prolific's eligibility settings; the
deployment record (`deployment/deployment_manifest.json`) is built with the
commit the code came from (from git, or from the `code_provenance.json` the
launcher wrote into its run copy; it stops without either); the IRB consent page from
`templates/consent.txt` is put in front of the experiment; the site and
functions (`/submit` and the token-protected `/results`) are deployed; the
page is fetched to check it is live, and `/results` is checked to refuse a
read without the token and accept one with it. Only then is a **draft** Prolific study
created (US, English-fluent, approval ≥ 98%, desktop, automatic payment on
completion) and its id recorded, and, in `live` mode only, published. A
failed deploy therefore leaves no study. `test` mode stops after the draft,
`none` mode after the deploy.

## Step 3: collect (`4_collect`, no agent)

`run_collect_programmatic` (`orchestrator.py`), `_collect_live` (`collect.py`).
In live mode it checks Prolific every 30 s until N people have finished or 3
hours have passed (`_PROLIFIC_MAX_WAIT_SEC`); in the second case it pauses
the study. It downloads from
`/results`, numbers participants uniquely across the run, and writes the five
raw columns to `data/responses.csv`. The full download, Prolific IDs
included, goes to `<project>/raw_collected/experiment<N>_responses.csv`,
beside the experiment folders. It aborts if every response is on one side.

Simulated modes: `simulated_participants` draws answers from the models'
priors (or `--ground-truth-model`); `simulated_participants_nobrowser` asks a
language model to answer.

## Step 4: the model stage (`5_model_loop`, the inner loop)

Code: `src/pipelines/inner_loop/` (`pymc_orchestrator.py` runs it,
`model_zoo.py` admission and removal, `scoring.py` comparison,
`candidate_agent.py` and `critique_round.py` the agents).

1. **Start.** `begin_model_loop_stage` records (or, on a restart, restores)
   the model set and agent notes the stage started from, and empties
   `model_loop/`. All experiments' responses are pooled into
   `model_loop/responses.csv`.
2. **Fit and score.** Every model is fitted by MCMC (defaults in
   `src/models/mcmc_defaults.py`: 4 chains, 4,000 draws, 3,000 tuning steps,
   `target_accept` 0.99; the live configs use 2,000–3,000 draws). Models are
   ranked by ELPD-LOO; the best is the top-ranked one with a reliable
   estimate (`_best_exportable_model`; reliability in
   `src/models/loo_reliability.py`). A model that cannot be fitted is
   dropped and logged, starting models included; a starting model that
   breaks the data contract, or a machine failure, stops the run. Each fit
   process compiles in its own directory.
3. **Rounds** (`--inner-loop-iterations`, default 2). Each round:
   - **Critique.** An agent writes up to 8 test statistics; each is computed
     on the data and on 1,000 datasets simulated from the best model
     (`src/critique/ppc.py`). Those with p ≤ 0.05 go into `critiques.md` for
     the theorists. A round without a usable statistic is marked
     `no_critique` in `history.json`.
   - **Proposals.** `--inner-loop-candidates` agents (default 3) run in
     parallel, each writing `candidate.py` and `hypothesis.md`. Roles
     (`slot_roles` in `model_zoo.py`): with C ≥ 4, C−3 explore (angles from
     `DEFAULT_CANDIDATE_HINTS` in `candidate_agent.py`), 2 improve the best,
     1 improves a model of its choice from `refinement_menu.md`; with 3: one
     of each; with 2: explore and improve-the-best; with 1: explore.
   - **Admission** (`_admit_candidate_with_reason`), in order: code check
     (`import_gate.py`); loadable with finite log-probability; data contract
     (`src/models/model_contract.py`); each sampling run of the fit within 30
     minutes (`CANDIDATE_FIT_TIME_LIMIT_SEC`); converged; finite
     ELPD-LOO; RMSE ≥ 0.002 from every admitted model on
     `model_loop/novelty_pool.json`. Candidates are fitted concurrently, then
     admitted one by one. A candidate's own broken code is a rejection; a
     machine or pipeline failure stops the run.
   - An empty slot is rerun once; a rejected proposal gets one repair
     attempt. Every attempt goes into `model_loop/attempted_hypotheses.jsonl`.
4. **End of experiment.** `_prune_losers`: models with a trustworthy score,
   starting models included, that trail the best trustworthy model by more
   than 2 × the clustered standard error (`src/models/clustered_se.py`) move
   to `model_loop/models/pruned/`. `_cap_live_set` keeps at most 8
   (untrustworthy scores go first, then the lowest); neither removes the
   best trustworthy model. The survivors and the attempts log are exported
   to `cognitive_models/` (`_export_inner_loop_models` in
   `model_loop_runner.py`). Then
   `finish_model_loop_stage` writes `model_registry.yaml` (equal weights) and
   `model_loop/export_complete.json`.

Agents are opencode with `google/gemini-3.1-pro-preview` (default) or Claude
Code (`--coding-agent claude`, `claude-sonnet-4-6`, billed as stated by
`--claude-auth subscription|api`), each with a 15-minute limit, run in a
bubblewrap sandbox (`src/runtime/agent_sandbox.py`). A call that hits the
account's usage limit is rerun after the reset, waiting up to 12 hours
(`src/runtime/usage_limits.py`).

## What a finished experiment contains

Under `<output dir>/subjective_randomness/experiment<N>/`:

| path | contents |
|---|---|
| `cognitive_models/` | the surviving models, `models_manifest.yaml` with each hypothesis, the attempts log |
| `design/stimuli.json`, `screened_out.json` | the 64 pairs and their EIG; models left out |
| `data/responses.csv` | the responses, five raw columns |
| `model_loop/report.md` | readable summary of the model stage |
| `model_loop/model_posterior.json` | comparison table (ELPD-LOO, differences, standard errors, reliability, convergence) |
| `model_loop/history.json` | best model and scores after every round |
| `model_loop/iter_<i>/` | every proposal, critique and agent transcript |
| `token_usage_summary.json` | language-model use and cost |

The answer to "what did the loop find?" is the best model of the last
experiment, read with the other survivors and the ELPD gaps. Browse runs with
`python -m src.viewer.server --data-root <output dir>` (port 8000; on Sherlock
run it inside a job and reach it by SSH port forwarding).
