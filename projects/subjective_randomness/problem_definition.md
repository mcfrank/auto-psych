# Subjective randomness: problem definition

## Task

On each trial, the participant sees **two sequences of coin flips** (H and T) and chooses which sequence looks **more random**. This is a classic paradigm for studying representativeness and alternation biases (e.g. Griffiths, "The Rational Basis of Representativeness").

## Experiment constraints

- **Total trials per experiment: 30.** Target duration is about 5 minutes at ~5 seconds per trial (consistent with Prolific).
- **Allowed sequence lengths: 4, 6, 8.** Pairs may mix lengths (e.g. a 4-symbol sequence vs a 6-symbol sequence).

## Stimulus design schema

- **Stimulus**: A pair of two sequences of H and T. Each sequence is a string (e.g. `"HHT"`, `"HTHTHT"`, `"HHTHTTHT"`). The two sequences in a pair **may have different lengths** (e.g. length 4 vs length 6).
- **Response**: Which sequence was chosen (`chose_left` ∈ {0, 1}; 1 means the participant chose sequence A).
- **Stimulus space**:
  - Allowed sequence lengths: 4, 6, 8 (see Experiment constraints).
  - Candidate stimuli: pairs of sequences (same-length or mixed-length) chosen by optimal design. The designer must select **exactly 30** stimuli per experiment.

## Suggested theories to consider

When adding theories, the theorist should take these suggestions into account (include or favor them when appropriate):

- Include **at least one theory based on the rational basis of representativeness**: a likelihood comparison between well-specified generative models of the data (e.g. Griffiths-style: compare sequences under different generative models and choose the one with higher likelihood under the preferred model).
- A simple **alternation heuristic** (preference for sequences with more H↔T transitions).
- A **representativeness heuristic** (preference for sequences whose head proportion is closer to 0.5).

## Theoretical models (PyMC contract)

Each model is a **PyMC model** in `cognitive_models/<name>.py`. At module load
the file builds, at top level, `with pm.Model() as model: ...` containing:

- One `pm.Data` container per stimulus feature the theory uses (see the
  feature-column list below). Initialize with a **1-element placeholder** of
  the correct dtype (e.g. `np.zeros(1, dtype="int64")`); the pipeline calls
  `pm.set_data(...)` to swap in real data before sampling.
- Priors over the theory's free cognitive parameters (e.g. softmax temperature,
  Beta prior on bias). MCMC infers these — they are **not** hyperparameters.
- A `pm.Bernoulli("response", p=p_left, observed=chose_left)` likelihood,
  where the `observed=` argument is the **exact** `chose_left = pm.Data("chose_left", ...)`
  tensor (not a derived copy).
- A `pm.Deterministic("p_left", ...)` exposing per-trial P(chose_left=1).

No callable-style `def model_name(stimulus, response_options)` functions — that
contract is no longer used. The pipeline fits each model via MCMC, scores it by
`arviz.loo` (ELPD-LOO), and uses posterior- and prior-predictive samples for
EIG, correlations, and PPCs.

## Preprocessed data schema (pm.Data column names)

`projects/subjective_randomness/preprocess_data.py` adds these numeric feature
columns to `responses.csv` after collect. Use these names verbatim in your
`pm.Data(...)` containers — the bridge auto-maps them by name.

| Column           | Type  | Meaning                                          |
| ---------------- | ----- | ------------------------------------------------ |
| `n_a`, `n_b`     | int   | Length of the sequence                           |
| `h_a`, `h_b`     | int   | Number of H's                                    |
| `p_a`, `p_b`     | float | Head proportion (`h / n`)                        |
| `alts_a`, `alts_b`       | int   | Alternation count (transitions H↔T)      |
| `p_alts_a`, `p_alts_b`   | float | Alternation proportion (`alts / (n-1)`)  |
| `max_run_a`, `max_run_b` | int | Longest constant-character run                |

Observed response: `chose_left` ∈ {0, 1}. Use
`chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))` and pass it
to `pm.Bernoulli(..., observed=chose_left)`.

You may pick any subset of these feature columns per theory — only the ones
the theory commits to.

## Optional references

- PDFs or papers in `references/` may be cited for scientific background (e.g. Griffiths on representativeness). Agents may read them if needed.
