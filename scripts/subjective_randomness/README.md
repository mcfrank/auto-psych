# Parameter Recovery

This directory contains the project-level machinery for checking whether a
subjective-randomness design can recover known parameters from simulated
choices.

The importable library code lives in `src/subjective_randomness/`; these are the
runnable command-line entry points. The canonical model families live in
`src/subjective_randomness/model_families/`. Data (stimuli and
responses) lives under `data/subjective_randomness/`.

Typical use from the repo root:

```bash
uv run python scripts/subjective_randomness/recover.py \
  --config scripts/subjective_randomness/configs/prototype_similarity.yaml \
  --out /tmp/prototype_recovery.json
```

The optimizer is intentionally dependency-free. It uses random restarts plus a
bounded coordinate search, which is enough for first-pass design diagnostics.

## PyMC Recovery

The PyMC bridge simulates choices from the pure-Python reference model, converts
the simulated rows to numeric feature columns, fits the matching PyMC adapter in
`src/subjective_randomness/pymc_model_families/`, and reports
posterior means/intervals against the known true parameters.

Quick smoke run:

```bash
uv run python scripts/subjective_randomness/pymc_recover.py \
  --config scripts/subjective_randomness/configs/prototype_similarity.yaml \
  --out /tmp/prototype_pymc_recovery.json \
  --n-repeats 1 \
  --draws 50 \
  --tune 50 \
  --chains 1
```

For debugging, add `--work-dir /tmp/prototype_pymc_recovery_rows` to keep the
simulated, featurized CSVs used by PyMC.

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
