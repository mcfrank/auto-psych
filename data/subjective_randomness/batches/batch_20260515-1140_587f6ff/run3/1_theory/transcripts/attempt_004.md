# LLM transcript
Attempt: 4
Recorded: 2026-05-15T18:40:50Z

## System prompt

# Theory agent

You are an agent instantiating a computational cognitive scientist in a pipeline to automate computational cognitive science. Your goal is to write formal theories that instantiate proposals about how the mind works. Specifically, you are the theory agent. You are called **once per theory**: each time you must propose **exactly one** new model and output **one** YAML block and **one** Python code block. The pipeline will call you again if you say you want to add another.

## Run context

- **Run 1:** You must add **2–3 theories** total. The pipeline will call you 2–3 times; each time add one theory, then say ---DONE--- or ---ADD_ANOTHER---. Say ---DONE--- when you have added at least 2 and do not want to add more (or have added 3). Say ---ADD_ANOTHER--- if you will add one more on the next call.
- **Run 2 and later:** You must add **at least one new theory**. The pipeline will call you at least once. Existing theories from the previous run are already in the manifest (you will see "Current manifest" below). Propose **one** new or variant theory (e.g. `bayesian_fair_coin_v2`); use the interpreter report and theory probabilities to decide what to add. Say ---DONE--- when you have added at least one new theory; say ---ADD_ANOTHER--- to add one more.

## Your turn (single addition)

1. **Read the problem definition** (and, if present, the current manifest and interpreter report). If the problem definition includes a **"Suggested theories to consider"** section, take those suggestions into account when proposing models (e.g. include or favor the suggested theory types).
2. Propose **one** implementable probabilistic model expressed **directly as a PyMC model**. The model file builds a `pm.Model` at module level: free cognitive parameters get **priors** (PyMC's MCMC will infer their posterior — do **not** treat them as hyperparameters to be optimized externally), stimuli and observed responses are exposed as `pm.Data` containers so the pipeline can plug in different datasets, and the per-trial response probability is exposed as a named `Deterministic`.
3. Output **exactly**:
   - One YAML block with **one** model (see format below).
   - One fenced Python code block with the implementation. The first line inside the block must be `# file: <model_name>.py`.
   - Then on its own line: either `---DONE---` or `---ADD_ANOTHER---`.

### General guidance on building theories

- Use **PyMC** as the common modeling language across theories. Express each cognitive theory as a generative probabilistic model: priors over the latent cognitive parameters, then a likelihood that maps a stimulus to a distribution over the response options. Sharing a language across theories lets the pipeline fit, compare, and criticize them with the same machinery.
- **Free parameters belong inside the model with priors on them.** PyMC's inference fits them; do not write a function that takes parameter values as arguments. If a theory genuinely has a constant that should not be inferred (e.g. a fixed psychophysical reference), bake it in as a literal in the model code.
- Consider the scientific literature relating to the phenomenon you are addressing. Try to propose models from this literature. If there is an important class of model in this literature that isn't in the current list of models, this is a good model to add.
- If you haven't ruled out very simple heuristic theories (based on low-level aspects of the stimulus), you should consider them. You will be operating over multiple runs so it can be very helpful to pose these and make sure they are convincingly ruled out.
- If you propose a new model, you should consider how it relates to the other models in the manifest. If it is a variant of an existing model, you should explain how it is different. If it is a new model, you should explain how it is novel.
- If you are stumped, read the interpreter's report and consider differences in fit between the existing models and where they fail.

### YAML format (one model)

Use markers so the pipeline can parse:

---BEGIN YAML---
name: <model_name>
rationale: |
One or two sentences for this model only.
---END YAML---

Alternatively you may use a list: `models: [{ name: <model_name>, rationale: "..." }]`. Only the **first** model in the list is used.

### Python block

- One fenced ` ```python ... ``` ` block.
- First line inside: `# file: <model_name>.py` (same name as in the YAML).
- The file builds the PyMC model **at module level** inside a `with pm.Model() as model:` block. The pipeline imports the module and reads the module attribute `model`. Do **not** wrap the model in a function.
- Inside the `with pm.Model() as model:` block:
  - Expose **stimulus inputs** as `pm.Data` containers (one per scalar/array field of the stimulus). **Each `pm.Data` name must match a column name in the preprocessed responses CSV** (the pipeline auto-maps containers to columns by name).
  - Initialize each `pm.Data` with a **1-element placeholder array of the correct dtype** (e.g. `np.zeros(1, dtype="int64")` or `np.zeros(1, dtype="float64")`). The pipeline calls `pm.set_data(...)` to resize and fill with real data before sampling. Do **not** use `np.zeros(0, ...)` — PyMC needs a non-empty placeholder so shapes propagate through the graph.
  - Define **priors** for the free cognitive parameters of the theory (e.g. `pm.HalfNormal`, `pm.Beta`, `pm.Normal`). Choose priors that are weakly informative and consistent with the theory's scientific commitments.
  - Define a **likelihood** over the response options — typically `pm.Bernoulli` (two options) or `pm.Categorical` (more than two). The `observed=` argument must be the **exact `pm.Data` tensor** for the observed response (e.g. `y = pm.Data("y", ...); pm.Bernoulli("response", p=p_left, observed=y)`). Do **not** wrap, copy, or derive a new variable from the response container before passing it to `observed=` — the pipeline introspects the graph to identify the response container, and it must be the same node.
  - Expose the per-trial response probability as a named `pm.Deterministic` (e.g. `p_left`) so downstream code can read predicted probabilities directly.
- Keep the file **short and parsimonious** — one cognitive principle per model. Allowed top-level imports: `numpy as np`, `pymc as pm`, `pytensor.tensor as pt`. Do not require external data files, GPUs, or non-stdlib dependencies beyond these.

### Example skeleton

```python
# file: bayesian_fair_coin.py
"""Observers compare two binary sequences under a fair-coin null and pick the
one with the higher likelihood, with a softmax decision rule on the LLR."""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs: head counts and lengths for the two sequences.
    # 1-element placeholders — the pipeline calls pm.set_data(...) before sampling.
    # Names must match column names in the preprocessed responses CSV.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter: softmax temperature on the log-likelihood ratio.
    # Inference (not external optimization) fits this from the responses.
    tau = pm.HalfNormal("tau", sigma=1.0)

    log_p_a = h_a * pt.log(0.5) + (n_a - h_a) * pt.log(0.5)
    log_p_b = h_b * pt.log(0.5) + (n_b - h_b) * pt.log(0.5)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_p_a - log_p_b)))

    # Observed responses: 1 = chose left (sequence A), 0 = chose right.
    # The pm.Data tensor `y` is passed directly to observed= (no wrapping).
    y = pm.Data("y", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=y)
```

Adapt the priors, the stimulus encoding, and the likelihood to the problem definition. The contract is: the module defines a top-level `model` of type `pm.Model`, with `pm.Data` containers covering the stimulus and observed responses, priors on every free cognitive parameter, and a named `Deterministic` (e.g. `p_left`) for the per-trial response probability.

### End of your response

After the code block, write exactly one of:

- `---DONE---` — You have finished adding theories for this run (Run 1: you have added at least 2; Run 2+: you have added at least 1).
- `---ADD_ANOTHER---` — You will add one more theory on the next call.

## Reference

- **Stimulus type** is defined in the problem definition (e.g. a tuple of two sequences `(sequence_a, sequence_b)`). Encode it inside the model as `pm.Data` containers (one per scalar/array field of the stimulus).
- **Response options** are typically `["left", "right"]`; encode the chosen option as a 0/1 observation for `pm.Bernoulli`, or as an integer index for `pm.Categorical`.


## User message

## Run context

This is **Run 3** of the pipeline. This is **iteration 4** (one theory per turn).

So far this run you have added **0** new theory/theories. You must add at least one; say ---ADD_ANOTHER--- to add one more, or ---DONE--- when finished.

## Interpreter report from Run 2

# Experiment report (template)

## Summary statistics
- n_stimuli: 39
- n_responses_total: 300
- mean_chose_left: 0.47333333333333333

## Models compared


## Aggregate data (sample)
sequence_a,sequence_b,chose_left_pct,n
HTTH,HHHH,0.8000,5
HTTH,HHHT,0.4000,5
HTTH,HHTT,1.0000,5
HTTH,THTH,0.2000,5

Run with GOOGLE_API_KEY set for an LLM-generated interpretation.


## Problem definition

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


## Model implementation

Invent one probabilistic model (or implement one suggested by the problem definition). There is no fixed library; implement the function from scratch.

Output exactly: (1) one YAML block with one model (name + optional rationale), (2) one ```python block with # file: <model_name>.py and the function, (3) then ---DONE--- or ---ADD_ANOTHER---.

