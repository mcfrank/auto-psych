# How the loop works

This page walks through one experiment and then explains what carries over to
the next one. Each step says what happens, why it was designed that way, and
where the code is. Terms in *italics* are defined in the [glossary](glossary.md).

## The task and the data

The project directory is
`src/pipelines/outer_loop/projects/subjective_randomness/`. It holds the task
definition (`problem_definition.md`, including the exact wording participants
see), the starting models and the Prolific study settings.

On each trial a participant sees two H/T sequences **of the same length**
(2 to 8 flips) and clicks the one that looks more random. Every response becomes
one row with five columns (`src/pipelines/outer_loop/columns.py`):

| column | meaning |
|---|---|
| `sequence_a`, `sequence_b` | the two sequences **as shown**: `sequence_a` is the one on the left (the web page randomly decides, per trial, which of the designed pair goes left) |
| `participant_id` | an integer, unique across the run's experiments (a new person gets the next number) |
| `trial_index` | trial number within the participant |
| `chose_left` | 1 if the participant clicked the left sequence (`sequence_a`), else 0 |

Every model is a self-contained PyMC file with a module-level `model`. It
computes its own features from the raw sequences through a hook
(`compute_features(sequence_a, sequence_b)` or `prepare_observed(rows)`). The
pipeline never hands models precomputed features. Every model is a
Bayesian model of `p_left`, the probability of choosing `sequence_a`, and
usually has free parameters such as decision noise or a side bias.

## Step 0: the starting models

Experiment 1 starts from the four models in
`projects/subjective_randomness/seed_models/` (the code calls them *seed
models*). Each is a quantitative version of a published account:

| model | idea |
|---|---|
| `falk_konold_dp` | Falk & Konold's (1997) "difficulty predictor": sequences that would be harder to memorise look more random |
| `motif_stack` | Griffiths et al. (2018): the likelihood ratio of "fair coin" versus a simple automaton that generates regular motifs |
| `finite_experience_occurrence` | Hahn & Warren (2009): how likely the string is to appear at least once in 20 fair flips |
| `local_representativeness` | Kahneman & Tversky (1972): sequences whose local windows look balanced and irregular look more random |

The starting models are **protected**: they are never removed, however badly
they fit. That keeps the literature baselines in every comparison.

> The main `README.md` says the starting models are "the best models discovered
> by three earlier human replicate runs", and `scripts/outer_loop_live/hero_run.yaml`
> says the same. That is out of date. The manifest
> (`seed_models/models_manifest.yaml`) says these four literature models replaced
> that earlier set.

Code: `seed_experiment_models_from_project` in
`src/pipelines/outer_loop/orchestrator.py`.

## Step 1: choosing the stimuli (`2_design`, no AI)

**What.** The pipeline lists every pair of distinct same-length sequences with
lengths 2–8, which is 43,434 pairs. It then picks 64 of them, one at a time:
each pick is the pair that most increases the *expected information gain* of the
whole set.

**Expected information gain (EIG)**, in plain terms: before running the
experiment, how many bits of uncertainty about *which model is right* would
the responses to these stimuli remove, on average over the outcomes the models
themselves predict? The calculation works like this:

- Pretend one of the models is true, draw plausible parameter values from it,
  and simulate what N participants would answer to each pair.
- Each pair yields a count of "left" choices, k ~ Binomial(N, p_left), where N is
  the planned number of participants (`--n-participants`).
- See how much that simulated data would shift the posterior over models.
- Average over 1,000 such simulated scenarios, with 200 parameter draws per
  model.

The models start with equal prior weight.

**Why this design.** Pairs where every model predicts the same thing are wasted
trials. EIG puts the trials where models disagree, weighted by how sharply N
participants could tell them apart.

**Details you may see in the output (`design/stimuli.json`):**

- *Stopping and filling.* The information gained by adding one more pair to the
  set shrinks quickly. Once the best available gain is within two Monte Carlo
  standard errors of zero, the remaining slots are chosen differently. Each is
  the pair with the highest information gain for a *single* response, given the
  pairs already picked. These are labelled `"source": "eig_single_response_fill"`;
  the main picks are labelled `"eig"`.
- *Experiment 1 versus later experiments.* Experiment 1 simulates from the
  models' priors. From experiment 2 on, every model is first fitted (quickly:
  500 draws, 500 tuning steps, 2 chains) to *all* data collected so far, and the
  simulation uses those posteriors.
- *Speed-ups.* The search is a fast approximation (re-scoring only the most
  promising candidates between full passes, in single precision). The
  developers checked it against the exact search on two designs. It takes
  minutes instead of hours. Each simulated scenario also leaves out the
  parameter draw that generated it when it averages the likelihood, so
  the true model is not rewarded for "remembering" its own draw.
- *Models left out of the design.* A model that cannot make a valid prediction
  for some pairs (a probability that is undefined, or outside 0–1) is left out
  of that experiment's design. So is a model whose predictions depend on the
  participant, and one whose own code fails on some pairs (for example a
  feature that looks at the fourth flip, on a pair of length 2). Every model left out is listed in `design/screened_out.json`,
  which is written even when it is empty.

Code: `run_design_programmatic` (`orchestrator.py`) →
`design_exhaustive` (`src/pipelines/outer_loop/eig.py`) → the
estimator in `src/models/eig_selection.py`.

> The standalone design command (`python -m src.pipelines.outer_loop.eig`) has
> different defaults (32 stimuli, lengths 4–8) from the pipeline (64 stimuli,
> lengths 2–8).

## Step 2: building and deploying the experiment (`3_implement`)

**What.** An AI coding agent reads a *fixed* template
(`src/pipelines/outer_loop/prompts/3_implement.md`), the task wording and
the chosen stimuli. It writes `experiment/index.html` (jsPsych 7, button
responses only, left/right order randomised on each trial) and
`experiment/config.json`. A validator checks the output. With `--validate`
(the live launchers always pass it), a failure is sent back to the agent up to
`--max-validation-repairs` (default 2) more times before the run stops.

**Note.** The template is fixed and only the stimuli change, so this step is
close to mechanical, but it is still done by an agent. The prompt insists the
page be identical across experiments apart from the stimuli. It is worth
opening the generated page yourself before any live study (see the rehearsal
in the runbook).

**Deployment** (`src/pipelines/outer_loop/deployment/local.py`, only when
`--deploy-target firebase`):

1. The IRB consent text in `templates/consent.txt` is injected as a
   full-screen "I agree" page in front of the experiment. The agent never
   writes consent text.
2. The site and two small server functions are deployed to Firebase:
   `/submit` stores a participant's data in Firestore, and `/results` returns
   it as CSV. `/results` and `/register_session` require a shared secret token.
3. The run's collection session is registered, so `/submit` accepts data only
   for this deployment.
4. The pipeline checks that the page actually loads, and refuses to go on if
   it does not.
5. A Prolific study is created: US residents, fluent in English, approval
   rate ≥ 98% by default, desktop only, with payment approved automatically on
   completion. The study is **published only in `live` mode**.

## Step 3: collecting responses (`4_collect`, no AI)

- **Live mode.** The pipeline checks Prolific every 30 seconds until the target
  number of participants have finished, **or 2 hours have passed**; in the
  second case it pauses the study so nobody else is recruited. It then
  downloads all submissions from `/results`. `data/responses.csv` gets the
  five raw columns only; the full download, Prolific IDs included, is kept in
  `raw_collected/experiment<N>_responses.csv` beside the experiment
  directories, which no agent is given.
- **Simulated modes.** Responses are simulated from the current models' priors
  (`--mode simulated_participants`, the default), or produced by asking a
  language model to act as each participant
  (`--mode simulated_participants_nobrowser`).

A basic quality check aborts the run if *every* collected response is on the
same side. The live dashboard (see the runbook) catches subtler problems while
the study is running.

Code: `run_collect_programmatic` (`orchestrator.py`), `_collect_live`
(`collect.py`).

## Step 4: finding better models (`5_model_loop`)

This is the only place where new hypotheses enter. The code calls it the
*inner loop*, as opposed to the *outer loop* over experiments. It works in the
directory `experimentN/model_loop/`.

### 4a. Fit everything

All responses from experiment 1 up to the current one are pooled into
`model_loop/responses.csv`, and every current model is fitted by MCMC.
Defaults: 4 chains, 4,000 draws and 3,000 tuning steps per chain,
`target_accept` 0.99 unless the model declares its own. These defaults are in
`src/models/mcmc_defaults.py`; the launcher configs set 2,000–3,000 draws and
2,000 tuning steps. A model that cannot be fitted or has no finite score is
dropped at this point; if that happens to a starting model, the run stops
instead.

**Comparison.** Models are compared by **ELPD-LOO**, the expected log predictive
density under leave-one-out cross-validation. For each response: fit the model
without it, and see how much probability the model gives to what the
participant actually chose. Sum over responses. Higher is better, and the
leave-one-out step penalises overfitting. It is estimated from the MCMC samples
by an importance-sampling shortcut (*PSIS*). That shortcut comes with a
diagnostic (*Pareto k*) saying when it cannot be trusted for a given response.
If more than 1% of responses fail that diagnostic, the model's score is marked
*unreliable* (`src/models/loo_reliability.py`).

**The best model** is the top-ranked model by ELPD-LOO among those whose score
is reliable and whose MCMC converged (`_best_exportable_model` in
`src/pipelines/inner_loop/scoring.py`). A "posterior probability" for each model
is also reported (`model_posterior.json`). It is a softmax of ELPD-LOO with a
small penalty for code length, it is known to be overconfident, and nothing
is selected on it.

### 4b. Rounds of critique and proposals

Each experiment runs a fixed number of rounds: `--inner-loop-iterations`
(default 2; the hero config uses 4). Each round has two parts.

1. **Critique** (the "CriticAL" method; `src/pipelines/inner_loop/critique_round.py`,
   `src/critique/ppc.py`). An agent proposes up to 8 test statistics that
   might reveal where the current best model fails, for example "how often
   people choose the sequence with more alternations". Each statistic is
   computed on the real data and on 1,000 datasets simulated from the fitted
   model (a posterior predictive check). Statistics with p ≤ 0.05 are reported
   as discrepancies, with a false-discovery-rate adjusted q alongside for
   information. They are written to `critiques.md` and passed to every
   proposing agent. If the agent writes no usable statistic twice, the round
   runs without a critique and `history.json` says so.

2. **Proposals.** Several agents, `--inner-loop-candidates` per round, run in
   parallel. Each writes one new model (`candidate.py`), a plain-language
   hypothesis (`hypothesis.md`) and optionally a name (`model_name.txt`). Each
   agent has a role:

   | number of agents per round | roles |
   |---|---|
   | 1 | 1 explore |
   | 2 | 1 explore, 1 improve-the-best |
   | 3 (default) | 1 explore, 1 improve-the-best, 1 improve-another |
   | C ≥ 4 | C−3 explore, 2 improve-the-best, 1 improve-another |

   - **Explore** agents must propose one genuinely new mechanism. Each gets a
     different "angle" from a fixed list of twelve (e.g. "a process-level
     account: memory, attention, encoding cost", "the decision rule: lapses,
     side bias, probability matching", "exemplar or prototype similarity";
     `DEFAULT_CANDIDATE_HINTS` in `candidate_agent.py`). They are told which
     hypotheses have already been tried and dropped, and not to re-propose them.
   - **Improve-the-best** agents get the current best model's code and
     hypothesis and must make one deliberate, stated change that would beat it.
   - **Improve-another** agents pick any other current or previously dropped
     model from a menu and improve it.

   Why the roles? An earlier batch of simulation runs showed the best model
   never changed. New proposals were all "breadth", and nobody was refining the
   leader.

**Admission.** A proposal joins the set only if it passes every check, in this
order (`_admit_candidate_with_reason` in `model_zoo.py`):

1. The files exist and the code passes a safety check: only an allowlist of
   imports (numpy, pymc, scipy, …), and no file access or `eval`.
2. It is a loadable PyMC model with a finite log-probability.
3. It can be fitted within 15 minutes.
4. The MCMC converged. That means the chains agree (R-hat ≤ 1.05), there are
   enough effectively independent draws (bulk ESS ≥ 100), and ≤ 0.1% of
   transitions were divergent. A fit that just misses is refitted once at a
   higher `target_accept`. Because the default is already 0.99, in practice
   this retry only happens for models that declare a lower `target_accept`.
5. It has a finite ELPD-LOO.
6. It is **not a near-duplicate**. Its predicted `p_left` must differ from
   every model already in the set by a root-mean-square difference of at least
   0.002. This is measured on a separate set of 512 random same-length pairs
   (lengths 4–8, written to `model_loop/novelty_pool.json`), not on the
   training stimuli. Two models that agree on the 64 training pairs but differ
   elsewhere therefore count as different.

A proposal whose own code breaks (a typo in its feature function, no
`p_left`, a `p_left` that is not one probability per pair) is rejected with the
error, like any other failed check. A failure of the machine or of the
pipeline's own code is not blamed on the proposal: it stops the run.

An agent that writes nothing gets one retry. A proposal that is rejected gets
one repair attempt, with the rejection reason in the agent's prompt. Every
attempt, admitted or not, is recorded in `model_loop/attempted_hypotheses.jsonl`,
together with every model removal. The code calls this file the *ledger*. It
carries over to later experiments, so agents are not asked to rediscover
hypotheses that were already tried and dropped.

**Where the agents run.** Agents are command-line coding assistants: opencode
with Google's Gemini (default model `google/gemini-3.1-pro-preview`), or Claude
Code (`claude-sonnet-4-6`), chosen with `--coding-agent`. Each runs inside a
`bubblewrap` sandbox (`src/runtime/agent_sandbox.py`). It can read the
code, write only its own directory, and sees none of the credentials except its
own API login. Each agent has a 15-minute limit.

### 4c. End of the experiment: who survives

After the last round, within the same `5_model_loop` stage:

- **Pruning.** A non-starting model is removed when its score is trustworthy
  and it is clearly worse than the best model whose score is trustworthy
  (a leader with an untrustworthy score does not count): its ELPD-LOO deficit exceeds 2
  standard errors of the difference. That standard error is computed with all
  responses to the same stimulus pair treated as one cluster
  (`src/models/clustered_se.py`). Responses to the same pair are correlated, so
  the ordinary per-response standard error is about half as large as it should
  be.
- **Size limit.** At most 8 models are kept. Models with untrustworthy scores go
  first, then the lowest ELPD-LOO. Starting models are always kept.
- Removed models are moved to `model_loop/models/pruned/`, not deleted.
- **Export.** All survivors, not just the winner, are copied to
  `experimentN/cognitive_models/` with a `models_manifest.yaml` giving each
  model's hypothesis. The attempts record goes next to them. A rival that the
  data cannot yet separate from the winner is kept, and the next design is
  aimed at separating them.
- `model_registry.yaml` then gives every surviving model **equal prior
  weight** for the next design.

Code: `_prune_losers`, `_cap_live_set` (`model_zoo.py`);
`_export_inner_loop_models`, `update_registry_from_interpretation`
(`src/pipelines/outer_loop/model_loop_runner.py`).

## Step 5: the next experiment

Experiment N+1 copies experiment N's `cognitive_models/` (models + manifest +
attempts record) and repeats steps 1–4 (`carry_forward_cognitive_models` in
`orchestrator.py`). It fails loudly if experiment N did not finish. **Only these
files cross the boundary**: model files, the attempts record, and the equal-weight
prior. The data cross implicitly, because every experiment's model stage and
every later design pool all earlier `data/responses.csv` files.

## The final output

For each experiment, under `<output dir>/<project>/experimentN/`:

| file | contents |
|---|---|
| `cognitive_models/` | the surviving models (code + `models_manifest.yaml` with each hypothesis) |
| `model_loop/report.md` | readable summary of the model stage |
| `model_loop/model_posterior.json` | the comparison table (ELPD-LOO, differences, standard errors, reliability and convergence flags), best model, reported posterior |
| `model_loop/history.json` | the best model and scores after every round, and whether each round had a critique |
| `model_loop/best_model.py` | the winning model's code |
| `model_loop/iter_<i>/` | every proposal (code, hypothesis, agent transcript) and every critique |
| `design/stimuli.json` | the stimuli and their information gain |
| `data/responses.csv` | the collected responses, five raw columns (runs collected before 28 September 2026 also contain Prolific IDs) |
| `../raw_collected/experiment<N>_responses.csv` | everything collection returned (**contains Prolific IDs in live runs**; see the privacy rules in the runbook) |
| `token_usage_summary.json` | language-model token use and cost for the experiment |

The answer to "what did the loop discover?" is the best model of the last
experiment, read together with the other survivors and the ELPD differences
between them. To browse all of it, run
`uv run python -m src.viewer.server --data-root <output dir>` and open
`http://127.0.0.1:8000`.
