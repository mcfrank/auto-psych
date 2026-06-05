# Subjective randomness: problem definition

The domain of inquiry is the subjective perception of randomness by humans judges. We seek an explantory theory of which sequences humans find more or less random. To explore this we use an forced-choice experimental paradigm, in which participants are asked to choose the more random sequence from a pair.

## Task

On each trial, the participant sees **two sequences of coin flips** (H and T) and chooses which sequence looks **more random**. This is a classic paradigm for studying representativeness and alternation biases (e.g. Griffiths, "The Rational Basis of Representativeness").

## Stimulus design schema

- **Stimulus**: A pair of two sequences of H and T. Each sequence is a string (e.g. `"HHT"`, `"HTHTHT"`, `"HHTHTTHT"`). The two sequences in a pair **may have different lengths** (e.g. length 4 vs length 6).
- **Response**: Which sequence was chosen (e.g. "left" / "right").

## Experiment design constraints

- **Total trials per experiment: 30.** Target duration is about 5 minutes at ~5 seconds per trial (consistent with Prolific).
- **Maximum sequence length: 8.**

## Models

Each model is a **PyMC model** in `cognitive_models/<name>.py`. At module load
the file builds, at top level, `with pm.Model() as model: ...` containing:

- One `pm.Data` container per stimulus feature the theory uses (see the
  feature-column list below). Each container must have a **1-element placeholder**
  of the correct dtype (e.g. `np.zeros(1, dtype="int64")`); the pipeline calls
  `pm.set_data(...)` to swap in real data before sampling.
- Priors over the theory's free cognitive parameters (e.g. softmax temperature,
  Beta prior on bias). MCMC infers these — they are **not** hyperparameters.
- A `pm.Bernoulli("response", p=p_left, observed=chose_left)` likelihood, where
  the `observed=` argument is the **exact** `chose_left = pm.Data("chose_left", ...)`
  tensor (not a derived copy).
- A `pm.Deterministic("p_left", ...)` exposing per-trial P(chose_left=1).

The pipeline fits each model on the preprocessed responses, scores it by
`arviz.loo` (ELPD-LOO), and uses posterior-predictive samples for correlations
and posterior predictive checks. No callable-style `def model_name(stimulus,
response_options)` functions — that is the old contract and is no longer used.

## Preprocessed data schema

`data/responses.csv` is produced by `preprocess_data.py`. It carries the raw
columns (`participant_id`, `trial_index`, `sequence_a`, `sequence_b`,
`chose_left`) **plus** the following numeric feature columns, one per sequence
(`_a` and `_b`). Use these names verbatim in your `pm.Data(...)` containers.

| Column           | Type  | Meaning                                          |
| ---------------- | ----- | ------------------------------------------------ |
| `n_a`, `n_b`     | int   | Length of the sequence                           |
| `h_a`, `h_b`     | int   | Number of H's                                    |
| `p_a`, `p_b`     | float | Head proportion (`h / n`)                        |
| `alts_a`, `alts_b`     | int   | Alternation count (transitions H↔T)        |
| `p_alts_a`, `p_alts_b` | float | Alternation proportion (`alts / (n-1)`)    |
| `max_run_a`, `max_run_b` | int | Longest constant-character run                |

And the observed response: `chose_left` ∈ {0, 1} — 1 means the participant
chose sequence A. Use `chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))`
in every model and pass it to `pm.Bernoulli(..., observed=chose_left)`.

You may pick any subset of these feature columns per theory. You do not need
to use them all — only the ones the theory commits to.

## Running the analysis (existing-data flow)

```bash
# 1. Preprocess a raw responses CSV (adds the feature columns above)
uv run python scripts/subjective_randomness/preprocess.py \
    --input-csv data/subjective_randomness/experiment1/responses.csv \
    --output-csv data/subjective_randomness/responses.csv

# 2. Run the outer loop (theory → design → collect → model loop)
uv run python -m src.pipelines.outer_loop.run \
    --project subjective_randomness \
    --experiment 1
```
