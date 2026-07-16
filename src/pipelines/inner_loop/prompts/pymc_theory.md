# Cognitive Hypothesis → PyMC Model (inner loop)

You are proposing one candidate **hypothesis** about how people make these
judgments. You do this in three steps: state the hypothesis in plain English,
give your model a short descriptive name, then translate that single hypothesis
into a PyMC model the pipeline fits with MCMC and compares by ELPD-LOO.

Read these files in the current working directory before deciding what to write:

1. `CONTEXT.md` — paths, the responses CSV schema (the feature columns your
   model may read), and the inner-loop round number.
2. `CANDIDATE_BRIEF.md` — what kind of hypothesis to attempt this round.
3. `existing_hypotheses.md` — the hypotheses already in the model set and how
   well each fits the data. Use it to pick a hypothesis that is genuinely
   different, or a refinement of a single existing one.

## Goal

Each model is one specific, falsifiable hypothesis about the cognitive process
people use — **not** a fit-maximizing combination of cues. Articulate one such
hypothesis and implement exactly that.

Do **NOT** build a mixture-of-heuristics: no averaging, weighting, or Dirichlet-
/ softmax-blending of cues or mechanisms drawn from several hypotheses into one
model. A model that bolts together many heuristics to fit better is not a
hypothesis and will be rejected. Refining a *single* existing hypothesis — a
different functional form, prior, or normalization of the **same** mechanism —
is encouraged.

## Step 1 — `hypothesis.md`

Write `hypothesis.md`: 1–3 plain-English sentences naming the single cognitive
mechanism you claim people use and how it drives their choice. No code, no math
notation — a psychologist should read it as one clear, testable claim.

## Step 2 — `model_name.txt`

Write `model_name.txt`: one line, a short **snake_case** name for your model
(lowercase letters, digits, underscores; 3–40 characters), e.g.
`recency_weighted_runs` or `motif_surprise_accumulator`. This becomes the
model's identifier in the manifest, the reports, and — if it wins — its
exported filename, so make it say what the mechanism *is*. Do not reuse a name
from `existing_hypotheses.md`, and do not use generic names like `new_model`.

## Step 3 — `candidate.py`

Translate the hypothesis in `hypothesis.md` into a PyMC model. It must build a
PyMC model **at module level** inside a `with pm.Model() as model:` block. The
pipeline imports the module and reads the module attribute `model`. Do **not**
wrap the model in a function. Start the file with a module docstring restating
the hypothesis (the same claim as `hypothesis.md`).

Inside the `with pm.Model() as model:` block:

- Expose **stimulus inputs** as `pm.Data` containers, one per scalar field of
  the stimulus. **Each `pm.Data` name must match a numeric column** the pipeline
  can supply — either a precomputed feature column in the responses CSV or a
  feature you derive yourself (see *Extending the feature space* below). The
  pipeline auto-maps containers to columns by name. Initialize each with a
  **1-element placeholder of the correct dtype** (e.g. `np.zeros(1,
  dtype="int64")`); the pipeline calls `pm.set_data(...)` to fill in real data
  before sampling. Do **not** use `np.zeros(0, ...)`. The raw H/T sequence
  strings `sequence_a`/`sequence_b` are **not** numeric and cannot be a
  `pm.Data` directly — derive numbers from them as below.
- Put **priors** on every free cognitive parameter (e.g. `pm.HalfNormal`,
  `pm.Beta`, `pm.Normal`). MCMC infers their posterior — do **not** take
  parameter values as function arguments or optimize them externally.
- Expose the per-trial response probability as a named `pm.Deterministic`
  (e.g. `p_left`) so downstream code can read predictions.
- Define a **likelihood** over the response — typically `pm.Bernoulli` (two
  options) or `pm.Categorical`. The `observed=` argument must be the **exact
  `pm.Data` tensor** for the observed response (e.g.
  `y = pm.Data("chose_left", ...); pm.Bernoulli("response", p=p_left, observed=y)`).
  Do **not** wrap, copy, or derive a new variable from the response container
  before passing it to `observed=` — the pipeline introspects the graph to
  identify the response container, and it must be the same node.

Allowed top-level imports: `numpy as np`, `pymc as pm`, `pytensor.tensor as pt`.
Keep the file short and parsimonious — **one cognitive mechanism per model**.
The number of free parameters and feature columns a model reads should match the
single hypothesis; a model that needs many weighted cues to fit is a blend, not
a hypothesis.

### Extending the feature space (optional)

The precomputed feature columns are order-destroying aggregates: they cannot see
where in a sequence something happens, the specific sub-sequences it contains, or
recency. If your hypothesis depends on such an aspect of the raw sequence, do
**not** try to force it from the existing columns — derive the exact statistic
your hypothesis needs by adding a module-level featurizer to `candidate.py`:

```python
def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return new numeric feature columns for one stimulus pair."""
    ...
```

The pipeline calls it on the raw `sequence_a`/`sequence_b` strings for every
trial and exposes each returned key as a new column you read with a matching
`pm.Data`. Use this to express a hypothesis the precomputed features cannot — for
example, whether the *last* toss of each sequence is heads (a recency cue):

```python
def compute_features(sequence_a, sequence_b):
    def ends_heads(seq):
        return 1.0 if seq.strip().upper().endswith("H") else 0.0
    return {"ends_heads_a": ends_heads(sequence_a), "ends_heads_b": ends_heads(sequence_b)}

# ... then inside the model:
#   ends_heads_a = pm.Data("ends_heads_a", np.zeros(1, dtype="float64"))
```

Rules for `compute_features`: it must return a dict of **finite numbers** with
the **same keys for every sequence pair**, and those keys must be **new names**
(not collisions with existing columns). It is still **one hypothesis** — add only
the feature(s) the single mechanism needs, not a grab-bag of cues to fit better.

### Numerical safety (required)

Your model must evaluate to a **finite** log-probability — a model whose `p_left`
or likelihood is NaN or `-inf` is rejected (it would crash MCMC at its
start-value check). So:

- Keep `p_left` strictly inside `(0, 1)`. A `sigmoid`/`softmax` already does
  this; if you build a probability another way, clamp it with
  `pt.clip(p, 1e-6, 1 - 1e-6)`.
- Use `pt.abs(x)` for absolute value — **not** `pt.sqrt(x ** 2)`, which returns
  NaN in PyTensor for some inputs.
- Avoid `log(0)`, division by zero, and unbounded exponentials of large scores.

## Example skeleton

```python
import numpy as np
import pymc as pm
import pytensor.tensor as pt

with pm.Model() as model:
    # Stimulus inputs — names match responses CSV columns.
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    h_a = pm.Data("h_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))
    h_b = pm.Data("h_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameter with a prior (inference fits it).
    tau = pm.HalfNormal("tau", sigma=1.0)

    log_p_a = h_a * pt.log(0.5) + (n_a - h_a) * pt.log(0.5)
    log_p_b = h_b * pt.log(0.5) + (n_b - h_b) * pt.log(0.5)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (log_p_a - log_p_b)))

    # Observed response: the pm.Data tensor is passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
```

## Self-check

Before stopping, confirm all three files exist — `hypothesis.md` (non-empty),
`model_name.txt` (one snake_case line), and a `candidate.py` that imports and
exposes a `pm.Model`. A candidate with no `hypothesis.md` is rejected; a
missing or invalid `model_name.txt` demotes your model to an auto-generated
name.

```bash
test -s hypothesis.md && echo "hypothesis.md OK"
test -s model_name.txt && echo "model_name.txt OK"
python3 -c "
from pathlib import Path
from src.models.pymc_inference import load_pymc_model, observed_response_data
m = load_pymc_model('candidate', Path('.'))
print('observed:', observed_response_data(m))
"
```
