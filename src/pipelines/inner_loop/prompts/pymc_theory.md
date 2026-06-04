# Cognitive Model Candidate Generation (PyMC)

You are generating one candidate cognitive model for an auto-psych inner model
loop. Models are written **directly as PyMC models** so the pipeline can fit
them with MCMC, compare them by ELPD-LOO, and criticize them with shared
machinery.

Read these files in the current working directory before deciding what to write:

1. `CONTEXT.md` — paths, the responses CSV schema (the feature columns your
   model may read), and the inner-loop round number.
2. `CANDIDATE_BRIEF.md` — what kind of change to attempt this round.
3. Any diagnostics present: `model_posterior.json` (current ELPD-LOO posterior
   over the surviving models), `report.md`, and the code of the top-mass models.

## Goal

Write a full replacement `candidate.py` that improves fit to the observed
responses, expressed as a probabilistic model in PyMC.

## Output contract

Write `candidate.py` in the current working directory. It must build a PyMC
model **at module level** inside a `with pm.Model() as model:` block. The
pipeline imports the module and reads the module attribute `model`. Do **not**
wrap the model in a function.

Inside the `with pm.Model() as model:` block:

- Expose **stimulus inputs** as `pm.Data` containers, one per scalar field of
  the stimulus. **Each `pm.Data` name must match a column in the responses
  CSV** (the pipeline auto-maps containers to columns by name). Initialize each
  with a **1-element placeholder of the correct dtype** (e.g.
  `np.zeros(1, dtype="int64")`); the pipeline calls `pm.set_data(...)` to fill
  in real data before sampling. Do **not** use `np.zeros(0, ...)`.
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
Keep the file short and parsimonious — one cognitive principle per model.

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

Before stopping, confirm `candidate.py` imports and exposes a `pm.Model`:

```bash
python3 -c "
from pathlib import Path
from src.models.pymc_inference import load_pymc_model, observed_response_data
m = load_pymc_model('candidate', Path('.'))
print('observed:', observed_response_data(m))
"
```
