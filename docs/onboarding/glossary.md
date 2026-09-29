# Glossary

Plain definitions of the terms used in these pages and the code names you will
meet in files, flags and logs. "Code term" marks words that come from the code
or from earlier work sessions rather than from standard usage.

## Statistics and modelling

**ELPD-LOO.** Expected log predictive density under leave-one-out
cross-validation. For each response, imagine fitting the model without it and
ask how much probability the model gives to the choice actually made; add up
the log probabilities. Higher is better, and it penalises overfitting. The
loop's main model-comparison score.

**ELPD difference (`elpd_diff`) and its standard error (`dse`).** How far a
model is behind the best one, and the uncertainty of that gap.
`dse_clustered` computes the uncertainty with all responses to the same
stimulus pair treated as one group (a *clustered standard error*). This is
needed because responses to one pair are correlated, and the ordinary `dse` is
then about half as large as it should be. Pruning uses the clustered one.

**PSIS, Pareto k, "reliable".** ELPD-LOO is estimated without refitting, by
reweighting the MCMC samples (Pareto-smoothed importance sampling, PSIS). For
each response, a diagnostic value *k* says whether that reweighting can be
trusted. In this code a model's score is **reliable** when at most 1% of
responses have a bad *k*. Responses that every posterior draw predicts
identically are exempt. Unreliable models cannot be named best and are removed
first when the set is full (`src/models/loo_reliability.py`).

**MCMC convergence.** The fitted posterior is trusted only when the independent
chains agree (**R-hat** ≤ 1.05), there are enough effectively independent draws
(**bulk ESS** ≥ 100), and ≤ 0.1% of sampler transitions were **divergent**
(a sign the sampler could not explore the posterior). `target_accept` is a
sampler setting; higher means smaller, more careful steps.

**Near miss** (code term). A proposal whose fit narrowly fails convergence
(≤ 2% divergent, R-hat ≤ 1.2, ESS ≥ 20). It is refitted once at
`target_accept` 0.95. That only happens if the model asked for a value below
0.95, since the default is 0.99.

**Expected information gain (EIG).** For a candidate set of stimuli: the
average reduction, in bits, of uncertainty about which model generated the
data, averaged over data simulated from the models themselves. **Joint EIG**
scores the whole set at once, with every participant's response counted.

**Noise floor, fill** (code terms). Adding pairs to the design stops helping
once the best additional gain is within two Monte Carlo standard errors of
zero (the *noise floor*). The remaining slots are then *filled* by the best
pairs for a single response (`eig_single_response_fill` in `stimuli.json`).

**Lazy greedy** (code term). The fast approximate search used to pick stimuli:
greedy one-at-a-time selection that re-scores only the most promising
candidates between full passes, in single precision.

**Leave-one-out (in the design).** When scoring a simulated scenario, the
parameter draw that generated it is left out of the average, so the true model
is not favoured artificially. Not related to ELPD-LOO.

**Prior predictive / posterior predictive.** Simulated data from a model using
parameter values drawn from its prior (before any data) or from its posterior
(after fitting). Experiment 1's design uses the prior; later designs use the
posterior.

**Posterior predictive check.** Compare a statistic of the real data with the
same statistic on data simulated from the fitted model. A value the model
rarely reproduces (small p) is a discrepancy.

**CriticAL.** The published method the critique step follows: an agent
proposes test statistics, and each gets a posterior predictive check. p-values
are adjusted for multiple comparisons with **BH-FDR** (the Benjamini–Hochberg
false-discovery rate); only the unadjusted p ≤ 0.05 flags a discrepancy.

**Model posterior** (`model_posterior.json`). A softmax of ELPD-LOO, with a
small penalty for long code, reported as "probability of each model".
Overconfident; used for reporting only, never for choosing.

**Stacking weights.** Weights for combining models' predictions, from
`arviz.compare`. Reported only. They are not model probabilities and are not
used for design.

**BMA** (Bayesian model averaging). In the recovery simulations, the metrics
with suffix `_bma` score the posterior-weighted average of all models' predictions
rather than the best model alone.

## The loop and its parts

**Outer loop / inner loop.** The outer loop is the sequence of experiments
(`src/pipelines/outer_loop/`). The inner loop is the model-discovery stage
inside each experiment (`5_model_loop`, `src/pipelines/inner_loop/`).

**Stage names.** `2_design` (choose stimuli), `3_implement` (agent builds the
web page; the deploy follows), `4_collect` (get responses), `5_model_loop`
(fit, critique, propose, compare, prune). There is no stage 1. `--agent <stage>`
runs one stage; `RESUME_AGENTS` does the same in the live job script.

**Seed models / starting models** (code: `seed_models/`). The four literature
models every run starts from. They are **protected** (never pruned).

**Candidate** (code term). A model proposed by an agent during the model stage,
before it is admitted.

**Admission.** The checks a candidate must pass to join the set: code safety,
loadable, finite, fits within 15 minutes, converged, finite ELPD-LOO, not a
near-duplicate.

**Novelty gate** (code term). The near-duplicate check. A candidate is rejected
if its predictions differ from some admitted model's by less than 0.002
root-mean-square in `p_left`, on the 512 random pairs of the **novelty pool**
(`model_loop/novelty_pool.json`).

**Zoo** (code term). The set of models in the current model stage,
`model_loop/models/`. Pruned models go to `model_loop/models/pruned/`.

**Live set** (code term). The models that survive an experiment and are
carried into the next one, in `experimentN/cognitive_models/`. **Carry-forward**
is copying them into the next experiment.

**Cap** (code term). The limit of 8 models kept after pruning.

**Pruning.** Removing non-starting models that are trustworthy and clearly
worse than the best: ELPD deficit > 2 × clustered standard error. Once per
experiment, at the end.

**Incumbent** (code term). The current best model: the top-ranked by ELPD-LOO
among reliable, converged models. The **incumbent record** in the simulations
tracks whether it changed and whether it is agent-discovered.

**Slot, role** (code terms). One proposing agent's place in a round, and what
it is asked to do: *explore* (a new mechanism), *refine incumbent* (improve
the best model), *refine chosen* (improve another model from a menu).

**Lens** (code term). The angle given to an explore agent, from a list of twelve
(`DEFAULT_CANDIDATE_HINTS`), e.g. "a process-level account". Overridable with
`hints_file`.

**Wave, retry, repair** (code terms). A *wave* is one batch of agents spawned
together within a round. A *retry* re-runs an agent that wrote nothing. A
*repair* re-runs an agent whose proposal was rejected, with the rejection
reason. Each happens at most once per agent.

**Ledger** (code term). `model_loop/attempted_hypotheses.jsonl`: one line per
proposal (admitted or rejected, with reason) and per removal (with margin). It
is carried across experiments and turned into the "already tried, do not
re-propose" list agents see.

**Registry** (`model_registry.yaml`). The model prior used by the next design:
equal weight for every carried model.

**Screened out** (`design/screened_out.json`). Models left out of one
experiment's design because they cannot predict all pairs validly. They stay
in the model set.

**Raw columns.** The five response columns models may use: `sequence_a`,
`sequence_b`, `participant_id`, `trial_index`, `chose_left`.

**Coding agent, backend.** A command-line AI coding assistant run as a
subprocess: opencode (default, with Google Gemini) or Claude Code. Chosen with
`--coding-agent` / `coding_agent:`.

**Sandbox** (bubblewrap, `bwrap`). A lightweight Linux isolation tool. Each
agent sees the code read-only, can write only its own directory, and gets only
its own API login among the credentials.

**Agent tree** (code term, simulations). The trimmed copy of the repository
the agents see in holdout simulations: no docs, tests, results or hidden-model
files.

## Simulations

**Recovery.** Running the loop on simulated participants from a known model and
measuring how close the loop's best model gets to it.

**GT (ground truth), held out** (code terms). The model that generated the
simulated data, removed from the starting set so the loop must rediscover it.

**Holdout harness** (code term). The code that runs and scores those
simulations (`src/subjective_randomness/holdout_recovery.py`, `holdout_eval.py`).

**Cell** (code term). One simulated run: one hidden model × one repeat.

**Impossible controls.** Simulations whose hidden rule is psychologically
implausible (e.g. "more heads looks more random"); the loop should *fail* to
recover them.

**Fitted-seed baseline** (code: `fitted_baseline`). The remaining starting
models fitted to the same data. The loop must beat it to show that discovery
added something.

**Leak / leakage** (code terms). Any route by which the hidden model's
identity could reach the agents. Checked before the agents start (name scan) and
afterwards (`leakage_audit.py`).

**Faithful config** (code term). `holdout_recovery_faithful.yaml`: the
simulation settings meant to match the live runs (40 participants, 3
experiments).

**Test–retest.** Repeating a simulation with different random seeds to measure
how reproducible its outcome is.

## Live deployment

**Run label** (`run_label`, `--run-label`). Your name for a run. It sets the
page URL (`/e<N>-<label>/`), the output directory and the session ids. Use a
new one for every launch.

**Run copy / worktree** (`$WORK_ROOT/runs/<label>/repo`). The per-run snapshot
of the code that a live job runs from.

**`WORK_ROOT`.** Where live runs write: default
`$SCRATCH/auto-psych/outer_loop_live`.

**Collection session** (`collection_session_id`). The id under which one
deployment's participant data are stored in Firestore. `/submit` accepts only
registered sessions.

**Deployment manifest** (`deployment/deployment_manifest.json`). The record of
one deploy: URLs, Prolific study id, payload, commit. The monitor finds studies
through these files.

**Prolific modes.** `test`: an unpublished draft study, then stop. `live`:
published, paid, full pipeline. `none`: no study.

**Deploy targets.** `dry-run`: stage everything locally, no network.
`firebase`: real deploy. `none`: no deploy.

**Prolific ID / PID.** A participant's 24-character Prolific identifier. In
the collected file (`raw_collected/experiment<N>_responses.csv`) it is the
`participant_id_str` column; `data/responses.csv` does not carry it.
Identifying: see the privacy rule in the runbook.

**Hero run, full run, pilot** (code terms). Names of the three preset live
configs: small single run (`pilot.yaml`); 3 parallel runs × 3 experiments
× 40 people (`full_run.yaml`); the same with a deeper model search
(`hero_run.yaml`).
