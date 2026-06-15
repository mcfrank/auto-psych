# Subjective-Randomness Recovery

This directory contains the project-level machinery for checking whether a
subjective-randomness design can recover known parameters — and re-identify
the generating model — from simulated choices.

The importable library code lives in `src/subjective_randomness/`; these are the
runnable command-line entry points. The canonical model families live in
`src/subjective_randomness/model_families/`. Data (stimuli and
responses) lives under `data/subjective_randomness/`.

## Whole Pipeline in One Command

`run_recovery_pipeline.py` chains everything: Bayesian (PyMC) parameter
recovery for every model-family config, the EIG-vs-random stimulus-set
comparison, closed-ended model recovery, and the analysis outputs. It writes
per-family report JSONs, tidy and summary CSVs, correlation figures, the
selection-comparison reports and figures, the confusion matrix
(JSON/CSV/heatmap), and a `key_results.txt` aggregating every stage's summary
table into one output directory:

```bash
uv run python scripts/subjective_randomness/run_recovery_pipeline.py
# quick small-MCMC check:
uv run python scripts/subjective_randomness/run_recovery_pipeline.py \
  --out-dir /tmp/recovery_smoke --n-repeats 3 --draws 100 --tune 100 --chains 1 \
  --skip-model-recovery --skip-selection-comparison
```

Both the parameter-recovery and model-recovery stages fit with MCMC;
`--draws/--tune/--chains` override the sampler settings for both, and
`--n-participants` shrinks model recovery for smoke tests.
`--skip-model-recovery` drops the (slowest) model-recovery stage and
`--skip-selection-comparison` the EIG-vs-random comparison (see below). The
stages can still be run individually with the scripts below.

### EIG-Optimized vs. Random Stimulus Sets

The selection-comparison stage quantifies how much an EIG-optimized stimulus
set improves recovery over a random one. Every stimulus in an auto-generated
candidate pool is scored by the expected information (bits) one response
carries under the prior — parameter-EIG on the family's grid for parameter
recovery, model-discrimination EIG for model recovery. The *eig* arm takes the
top-`n_stimuli`; the *random* arm a uniform same-size draw from the same pool.
For each model family it then samples ground-truth parameter vectors
(`parameter_repeats` of them) and recovers each truth on **both** sets with
the same exact grid posterior (Bayesian, no MCMC), so within a repeat the gap
in `pearson_r`/`rmse` isolates the value of the optimized design. The same
paired design is run for model-identity recovery (`model_repeats` sampled
truths per generating model), reporting accuracy and the mean posterior on the
true model per arm.

Configured by `configs/selection_comparison.yaml` (`model_names`, `pool`,
`n_stimuli`, `points_per_dim`, `seed`, plus per-comparison repeat and
participant counts — `parameter_repeats`/`parameter_participants` and
`model_repeats`/`model_participants`). Outputs per family:
`selection_comparison_{model}.json` (including each arm's chosen stimuli,
annotated with their EIG) and a truth-vs-estimate scatter figure with one row
per arm, plus `selection_comparison_model_recovery.json`/`.png` (side-by-side
confusion heatmaps) and a comparison section in `key_results.txt`. The EIG
set in the JSON is ready to reuse as a `stimuli_path` JSON for the recovery
configs.

## Parameter Recovery

Parameter fitting is **Bayesian only**: simulated choices are fit with the
matching PyMC adapter in `src/subjective_randomness/pymc_model_families/`
(simulate from the pure-Python reference family, featurize the rows, sample
the posterior, compare posterior summaries to the known truths). Typical use
from the repo root:

```bash
uv run python scripts/subjective_randomness/pymc_recover.py \
  --config scripts/subjective_randomness/configs/prototype_similarity.yaml \
  --out /tmp/prototype_recovery.json \
  --tidy-csv /tmp/prototype_recovery.csv
```

By default each repeat draws a fresh ground-truth parameter vector uniformly
from the model family's `PARAM_BOUNDS`, so recovery is evaluated across the
whole parameter space rather than at one hand-picked point. A config can narrow
the sampling with `param_ranges` (e.g. `param_ranges: {beta: [1.0, 8.0]}`;
ranges must stay inside the family bounds) or pin every repeat to a single
vector by giving `true_params` (fixed-truth mode).

MCMC settings come from the config's `mcmc` block (`draws`, `tune`, `chains`,
`cores`) or the matching CLI flags; `--n-repeats` shrinks a run for smoke tests, and
`--work-dir /tmp/rows` keeps the simulated, featurized CSVs for debugging.

Pass `--tidy-csv PATH` to also write a long-format CSV with one row per
`(parameter, repeat)` — columns `model, parameter, repeat, true_value,
estimate, error` — ready for plotting.

To summarize a recovery report — and, for sampled-truth reports, plot
ground-truth vs. recovered correlation scatters (one panel per parameter, with
the identity line and Pearson r) — feed it to the analysis CLI:

```bash
uv run python scripts/subjective_randomness/analyze_recovery.py \
  --results /tmp/prototype_recovery.json \
  --out-csv /tmp/prototype_recovery_summary.csv \
  --figure /tmp/prototype_recovery.png
```

The summary (printed and in the CSV) includes a `pearson_r` column: the
correlation between the sampled ground truths and the recovered estimates for
each parameter. Fixed-truth reports have no truth variance, so `pearson_r` is
empty there and the figure instead shows the estimate spread around the true
value.

## Closed-Ended Model Recovery

Parameter recovery (above) asks whether one model can recover its *own*
parameters. *Closed-ended model recovery* asks a different question: if a known
seed model generated the data, does the inner model loop — comparing the
*closed* set of seed models, with no agent-proposed candidates — put its
posterior mass back on the true model?

For each generating seed model, the pipeline fixes that model's PyMC parameters,
samples synthetic choices over the stimuli, runs the inner loop
(`max_iterations=0`) on the seed set, and records the recovered posterior over
models. The output is a generating-model × recovered-model confusion matrix; a
well-behaved pipeline concentrates posterior mass on the diagonal.

```bash
uv run python scripts/subjective_randomness/model_recovery.py \
  --config scripts/subjective_randomness/configs/model_recovery.yaml \
  --out data/subjective_randomness/model_recovery/confusion.json \
  --tidy-csv data/subjective_randomness/model_recovery/confusion.csv
```

The JSON holds the full result (per generating model: recovered posterior,
ELPD-LOO, and the best model). The tidy CSV has one row per
`(generating_model, recovered_model)` cell — columns `generating_model,
recovered_model, posterior, elpd_loo, is_true_model, is_best_model` — which
drops straight into a confusion-matrix heatmap.

The config's `generating_models` key selects which models generate data and
their fixed parameters; omit it to recover every seed model with its family's
default parameters. MCMC settings (`--draws`, `--tune`, `--chains`) and
`--n-participants` can be overridden on the command line.

By default the synthetic data is generated from the **PyMC seed model** itself
(`generator: pymc`), so the true model and one fitted candidate are identical.
Set `generator: model_family` (or pass `--generator model_family`) to instead
generate from the pure-Python `model_families` family of the same name. Its
functional form differs from the PyMC fit, making recovery a harder, more honest
test of whether the loop can re-identify the generating process.

## Holdout Recovery — the Full Agentic Loop vs. a Held-Out Ground Truth

Closed-ended recovery (above) keeps the true model *in* the candidate set.
*Holdout recovery* removes it: each seed model in turn is held out as the
ground truth — its fixed-param PyMC model generates every synthetic response —
while the full agentic outer+inner loop starts from the *remaining* seed models
and tries to recover the held-out process. Real agents do the work: the design
agent chooses each experiment's stimuli by EIG, the theory agent proposes
models between experiments, and the inner loop's candidate agents conjecture
new PyMC models that are fit by MCMC and compared by ELPD-LOO.

After **every inner-loop scoring step** (the initial seed-set fit and each
candidate round, in every experiment) the then-best model's posterior-predictive
`p_left` is correlated with the ground truth's `p_left` on a large held-out
stimulus set, giving a trajectory of how the loop converges on the true process.

```bash
uv run python scripts/subjective_randomness/holdout_recovery.py \
  --config scripts/subjective_randomness/configs/holdout_recovery.yaml \
  --out data/subjective_randomness/holdout_recovery/holdout.json \
  --tidy-csv data/subjective_randomness/holdout_recovery/holdout.csv \
  --figure data/subjective_randomness/holdout_recovery/holdout.png
```

This spawns real coding agents and runs real MCMC — a full 3-model run takes
hours. Scope a smoke run first:

```bash
# 1 ground truth, 1 experiment, seed-set-only inner loop, tiny MCMC
uv run python scripts/subjective_randomness/holdout_recovery.py \
  --config scripts/subjective_randomness/configs/holdout_recovery.yaml \
  --out /tmp/holdout_smoke/holdout.json \
  --gt-model bayesian_diagnosticity \
  --n-experiments 1 --n-participants 5 --inner-loop-iterations 0 \
  --draws 150 --tune 150 --chains 2
```

Outputs, per held-out model under `<out dir>/<out stem>_runs/<gt_model>/`:
the full `experiment1..N/` pipeline trees, `eval_stimuli.json` (the held-out
evaluation set), and `trajectory.json` (the per-step correlation trajectory,
per-experiment model sets, and a leakage audit). The combined JSON, tidy CSV
(one row per `gt_model × step`: `gt_model, experiment, step, iteration,
global_step, best_model, pearson_r, rmse`), and correlation-vs-step figure land
at the paths you pass.

Details worth knowing:

- **Held-out eval set.** The design agents choose training stimuli freely, so
  holdout is enforced *after* the run: any eval-pool pair that appeared (in
  either order) in any of the run's training data is dropped, and the run fails
  loudly if fewer than `eval_pool.min_remaining` stimuli survive.
- **Per-step history.** The inner loop now writes `model_loop/history.json`
  (best model + posterior after the seed fit and after every candidate round);
  the trajectory evaluation refits each step's best model through the shared
  MCMC cache (`--cache-dir`, default `<out dir>/mcmc_cache`), so evaluation
  costs no new sampling.
- **The cache ignores fit kwargs.** Cached fits are keyed by model file + data
  only, so changing `--draws`/`--tune`/`--chains` for a fresh run requires
  clearing the cache directory first.
- **Leakage is audited, not prevented.** Agents can read the project assets
  dir, which contains the held-out model's source. `trajectory.json` flags
  byte-identical copies, mentions of the ground truth's distinctive parameter
  names, and files named after the ground truth — heuristics for auditing a
  run, not proof it was clean.
- **Design = candidate pool + deterministic EIG.** The design agent's creative
  job is to write `design/candidates.json` (a pool of stimulus pairs); scoring
  it by EIG into the top-N `design/stimuli.json` is deterministic. Agents
  sometimes background that slow scoring and end their turn before
  `stimuli.json` exists — so when the agent leaves a candidate pool but no
  stimuli, the harness runs the EIG annotation itself (top 20 by EIG) rather
  than stalling. Keep candidate pools tractable (the prompt asks for ~100–300
  pairs): EIG scores every candidate via prior-predictive sampling, so a pool
  of thousands takes many minutes.
- **Resume after a failure.** The harness stops loudly at the first agent
  stage whose output doesn't validate. Re-run the same command with `--resume`
  to continue: any ground truth that already has a `trajectory.json` is
  skipped, and within an incomplete run every stage whose output already
  validates is skipped, so work restarts at the first invalid stage. If a
  stopped design stage left a valid `candidates.json`, resume reuses it (it
  scores the EIG and writes `stimuli.json` without re-running the expensive
  design agent). A partial `model_loop/` from a crashed inner loop is wiped and
  rerun (it is fully regenerable, and its MCMC fits are still in `mcmc_cache/`).
  If `--resume` finds a `trajectory.json` whose experiment count disagrees with
  the config's `n_experiments`, it fails loudly rather than mixing runs.

  ```bash
  # after fixing the cause of the failure, continue the same run:
  uv run python scripts/subjective_randomness/holdout_recovery.py \
    --config scripts/subjective_randomness/configs/holdout_recovery.yaml \
    --out data/subjective_randomness/holdout_recovery/holdout.json \
    --tidy-csv data/subjective_randomness/holdout_recovery/holdout.csv \
    --figure data/subjective_randomness/holdout_recovery/holdout.png \
    --resume
  ```

- **opencode permissions.** Agents run from the repo root, so opencode treats
  `/tmp` (and `/private/tmp`, `/var/folders`) as external directories and, in
  non-interactive `opencode run`, auto-rejects writes there — which silently
  breaks design agents that stage temp files. The repo `opencode.json` grants
  `external_directory` allow rules for those paths so this does not recur. If
  you point the pipeline at a different scratch directory, add it there too.

## Analyzing Results

`analyze_recovery.py` summarizes either result type — it auto-detects whether the
JSON is a parameter-recovery report or a model-recovery confusion result.

```bash
uv run python scripts/subjective_randomness/analyze_recovery.py \
  --results data/subjective_randomness/model_recovery/confusion.json \
  --out-csv data/subjective_randomness/model_recovery/summary.csv \
  --figure  data/subjective_randomness/model_recovery/confusion.png
```

For a **parameter-recovery** report it prints, and writes to the summary CSV, one
row per parameter: `true_value, mean_estimate, bias, rmse, estimate_sd,
n_repeats, ci_coverage_95` (the last is the fraction of repeats whose 95% credible
interval contains that run's true value — a calibration check; `n/a` only for
legacy reports whose posterior summaries carry no `q025`/`q975` interval). The
optional figure shows each repeat's estimate against the true value.

For a **model-recovery** confusion result it reports overall posterior- and
ELPD-LOO-based recovery accuracy and the mean posterior on the true model, plus
per generating model the best-fitting model by each criterion (flagging
mis-recoveries). The optional figure is the generating × recovered posterior
confusion heatmap.

It also reports **distinguishability** from the PSIS-LOO comparison table:
`winner_margin` (the runner-up's `elpd_diff`) with its `dse`,
`winner_distinguishable` (margin > 2·dse), and `recovery_clear` (the ELPD winner
*is* the true model **and** is distinguishable). The headline `clear_recovery_rate`
is the fraction of generating models cleanly recovered — which can be far below
the raw accuracy when models are statistically tied. A model that "wins" by a
margin within ~2·dse is a coin flip, not a recovery; bumping MCMC draws will not
change that — see below.

## Improving Recoverability — Discriminating Stimuli

If recovery is poor, the usual cause is **stimulus diagnosticity**, not sample
size: if the candidate models predict nearly the same choices on your stimuli,
they fit any data equally well and no number of participants or MCMC draws will
separate them. `select_stimuli.py` scores each candidate sequence pair by the
expected information (bits) it carries about *which* model generated the
response — the mutual information between model identity and the binary choice —
and keeps the most discriminating ones.

```bash
uv run python scripts/subjective_randomness/select_stimuli.py \
  --candidates pool.json \
  --out data/subjective_randomness/discriminating_stimuli.json \
  --top 20
```

Each output stimulus is annotated with `discrimination_eig` (bits; higher =
better at telling the models apart, ~0 = useless). Feed a large/varied candidate
pool in and the discriminating subset out — then run recovery on that. By
default it scores against all reference model families at their default
parameters; pass `--param-samples N` to average over `N` draws from each
family's parameter bounds (a cheap prior-predictive that accounts for parameter
uncertainty), and `--models` to restrict the set.

This is the fast, MCMC-free design pass. `src/pipelines/outer_loop/eig.py`
computes the same expected-information-gain quantity from the fitted **PyMC**
models' prior predictive when you want the full version.

## Adaptive (Sequential) Recovery — the loop designs its own stimuli

`select_stimuli.py` ranks a fixed pool once. `adaptive_recover.py` goes further:
it runs a **sequential Bayesian optimal-design loop** that auto-generates a
diverse candidate pool and then, each round, picks the stimulus with the highest
EIG *under the current posterior*, simulates the response, updates beliefs, and
repeats — so the experiment designs itself. It runs on the pure-Python families
with an exact grid posterior (no MCMC): fast, deterministic, the
design-evaluation counterpart to the one-shot PyMC recovery.

```bash
# Model recovery: which model generated the data? (confusion over generators)
uv run python scripts/subjective_randomness/adaptive_recover.py \
  --config scripts/subjective_randomness/configs/adaptive_recovery.yaml \
  --out data/subjective_randomness/adaptive/model_recovery.json

# Parameter recovery: recover one model's parameters
uv run python scripts/subjective_randomness/adaptive_recover.py \
  --config scripts/subjective_randomness/configs/adaptive_recovery.yaml \
  --mode parameter \
  --out data/subjective_randomness/adaptive/param_recovery.json \
  --selected-out data/subjective_randomness/adaptive/design.json
```

In **model mode** it recovers each generating model in turn into a confusion of
recovered-vs-true with the model posterior; in **parameter mode** it reports the
posterior mean/sd/error per parameter and the grid-posterior entropy gained, and
`--selected-out` writes the stimuli the loop chose. Config keys: `mode`,
`model_names`, `pool` (`n_pairs`, `lengths`), `n_rounds`, `n_participants`,
`points_per_dim` (grid resolution per parameter), `seed`, and per-mode
`generating_models` / `model` + `true_params`.

This is the answer to "the models won't recover": with self-chosen
high-EIG stimuli, models that are statistically tied under a fixed undiagnostic
set (e.g. `prototype_similarity` vs `bayesian_diagnosticity`) separate cleanly,
and weakly-identified parameters like `beta` are pinned down by stimuli selected
near the choice boundary. Estimates are limited by the grid resolution
(`points_per_dim`); raise it for finer parameter estimates.

## Model Families

All three model families use the same forced-choice observation model. For a
trial with left sequence `A` and right sequence `B`, each model computes a
sequence-level score `S(seq; theta)` and then predicts:

```text
P(choose left | A, B, theta) =
  sigmoid(beta * (S(A; theta) - S(B; theta)) + side_bias)
```

where:

```text
sigmoid(x) = 1 / (1 + exp(-x))
```

`beta` is choice sensitivity. Higher `beta` means more deterministic choices.
`side_bias` is a left/right response bias. Positive values favor the left
sequence, independent of its content.

### 1. Bayesian Diagnosticity

Source: `src/subjective_randomness/model_families/bayesian_diagnosticity.py`

This model is the closest implementation of the Tenenbaum & Griffiths
representativeness account. A sequence looks random when it is better evidence
for a fair/random generator than for salient non-random alternatives.

The model computes:

```text
S(seq) =
  log P(seq | fair)
  - log P(seq | alternatives)
```

Because the experiment allows mixed-length pairs, log probabilities are
length-normalized:

```text
log P_norm(seq | h) = log P(seq | h) / len(seq)
```

The fair generator is an iid fair coin:

```text
P(seq | fair) = product_t P(x_t)
P(H) = P(T) = 0.5
```

The alternative generator is a mixture:

```text
P(seq | alternatives) =
    pi_alt    * P_norm(seq | alternating)
  + pi_bias   * P_norm(seq | biased)
  + pi_streak * P_norm(seq | streaky)
```

implemented in log space with `logsumexp`.

The alternating and streaky generators are first-order Markov generators:

```text
P(first symbol) = 0.5
P(switch) = 0.95   for alternating
P(switch) = 0.15   for streaky
```

So:

```text
log P(seq | Markov(q)) =
  log(0.5)
  + n_switches(seq) * log(q)
  + n_stays(seq)    * log(1 - q)
```

The biased generator is a symmetric mixture of mostly-heads and mostly-tails
coins:

```text
P(seq | biased) =
  0.5 * P(seq | P(H)=0.85)
  + 0.5 * P(seq | P(H)=0.15)
```

The recoverable mixture parameters are represented with a stick-breaking
parameterization:

```text
pi_alt    = alt_prior
pi_bias   = (1 - alt_prior) * bias_share
pi_streak = (1 - alt_prior) * (1 - bias_share)
```

Main parameters:

```text
alt_prior   : weight on the alternating alternative
bias_share  : share of remaining alternative mass assigned to biased coins
beta        : choice sensitivity
side_bias   : left/right response bias
```

Psychological interpretation: people judge randomness by asking whether the
sequence is diagnostic of a fair random process rather than an overly
alternating, biased, or streaky process.

### 2. Prototype Similarity

Source: `src/subjective_randomness/model_families/prototype_similarity.py`

This model treats subjective randomness as similarity to an internal prototype:
random-looking sequences should be close to 50/50 heads/tails and close to an
ideal alternation rate.

Features:

```text
balance_distance(seq) =
  2 * |prop_H(seq) - 0.5|

alternation_rate(seq) =
  n_switches(seq) / (len(seq) - 1)

alternation_distance(seq) =
  |alternation_rate(seq) - theta_alt|
```

The sequence score is:

```text
S(seq) =
  - [
      (1 - alt_weight) * balance_distance(seq)
      + alt_weight     * alternation_distance(seq)
    ]
```

Main parameters:

```text
theta_alt   : ideal alternation rate for a random-looking sequence
alt_weight  : relative weight on alternation distance vs. H/T balance
beta        : choice sensitivity
side_bias   : left/right response bias
```

Psychological interpretation: people compare a sequence to a mental prototype
of randomness. `theta_alt` allows the prototype to prefer overalternation
relative to a true fair coin, while still penalizing perfectly alternating
sequences if they exceed the ideal.

### 3. Encoding Compressibility

Source: `src/subjective_randomness/model_families/encoding_compressibility.py`

This model says that sequences look non-random when they have a short, simple
description. Examples like `HHHHHHHH`, `HTHTHTHT`, and `HHHHTTTT` are easy to
encode, so they receive lower randomness scores.

Features:

```text
max_run_norm(seq) =
  (max_run_length(seq) - 1) / (len(seq) - 1)
```

This is near `0` for fully alternating sequences and `1` for a solid run.

```text
periodicity_score(seq) =
  max over periods p <= len(seq)/2 of template_match(seq, p),
  rescaled so weak periodicity is near 0 and obvious repetition is near 1
```

This penalizes simple repeating templates such as `HTHTHTHT`.

```text
imbalance(seq) =
  2 * |prop_H(seq) - 0.5|
```

The model uses a stick-breaking parameterization for feature weights:

```text
w_longrun  = longrun_weight
w_periodic = (1 - longrun_weight) * periodic_share
w_imbalance =
  (1 - longrun_weight) * (1 - periodic_share)
```

The sequence score is negative compressibility:

```text
S(seq) =
  - [
      w_longrun  * max_run_norm(seq)
      + w_periodic * periodicity_score(seq)
      + w_imbalance * imbalance(seq)
    ]
```

Main parameters:

```text
longrun_weight : weight on long-run compressibility
periodic_share : share of remaining weight assigned to periodic patterns
beta           : choice sensitivity
side_bias      : left/right response bias
```

Psychological interpretation: people judge a sequence as random when it is hard to summarize with a simple rule. This model can penalize both long streaks and perfect alternation, because both are compressible.

## Why Parameter Recovery?

Parameter recovery checks whether a proposed design can estimate the parameters
it claims to measure. The workflow is:

```text
1. Choose a model family and true parameter values.
2. Simulate responses from those known values.
3. Fit the same model family back to the simulated responses.
4. Compare recovered parameters against the known true parameters.
```

If recovery is poor, the stimuli may not separately identify the parameters.
For example, if balance and alternation always favor the same option, a model
may fit choices well while failing to distinguish `alt_weight` from balance
sensitivity.
