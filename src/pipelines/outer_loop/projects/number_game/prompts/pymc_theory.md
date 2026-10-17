# Cognitive Hypothesis → PyMC Model (inner loop)

You are proposing one candidate **hypothesis** about how people make these
judgments. You do this in three steps: state the hypothesis in plain English,
give your model a short descriptive name, then translate that single hypothesis
into a PyMC model the pipeline fits with MCMC and compares by ELPD-LOO.

Read these documents before deciding what to write. Each is included below in
this prompt and saved in your candidate directory:

1. `CONTEXT.md` — paths, the responses CSV's columns (the raw example sets and
   targets and the response only), and the inner-loop round number.
2. `CANDIDATE_BRIEF.md` — what kind of hypothesis to attempt this round.
3. `existing_hypotheses.md` — the hypotheses already in the model set and how
   each stands on the current data by ELPD-LOO rank (`elpd_diff ± dse` against
   the best model). Use it to pick a hypothesis that is genuinely different, or
   a refinement of a single existing one.
4. `attempted_hypotheses.md` — in an *exploratory* slot: hypotheses tried
   earlier that are no longer in the model set, with what happened to each:
   pruned after losing to the best model by a stated margin, or rejected at
   admission (most often as a near-duplicate of a model still in the set). Do
   not propose any of them again under a new name; a mechanism that lost by a
   wide margin is ruled out, and one already covered by a live model needs no
   second copy.
5. `refinement_menu.md` — in a *refinement* slot (your brief says which kind
   of slot this is): the models you may refine — the live models other than
   the incumbent and the models pruned earlier — each with its full
   hypothesis, its standing or the margin by which it lost, and its source
   file. A menu, not a blacklist.

## Goal

Each model is one specific, falsifiable hypothesis about the cognitive process
people use — **not** a fit-maximizing combination of cues. Articulate one such
hypothesis and implement exactly that.

Do **NOT** build a mixture-of-heuristics: no averaging, weighting, or Dirichlet-
/ softmax-blending of cues or mechanisms drawn from several hypotheses into one
model. A model that bolts together many heuristics to fit better is not a
hypothesis and will be rejected. Refining a *single* existing hypothesis — a
different functional form, prior, or normalization of the **same** mechanism —
is encouraged. In a *refinement* slot (`CANDIDATE_BRIEF.md` says so) the rule
against grafting is lifted: you may extend the model you refine with one
component from another model, as a single stated change. A model that bolts
together many heuristics to fit better is not a hypothesis in any slot.

## Step 1 — `hypothesis.md`

Write `hypothesis.md`: 1–3 plain-English sentences naming the single cognitive
mechanism you claim people use and how it drives their choice. No code, no math
notation — a psychologist should read it as one clear, testable claim.

## Step 2 — `model_name.txt`

Write `model_name.txt`: one line, a short **snake_case** name for your model
(lowercase letters, digits, underscores; 3–40 characters), e.g.
`magnitude_band_generalization` or `last_example_anchoring`. This becomes the
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
  the stimulus. The responses CSV has no feature columns — only the raw
  stimulus columns `number_set` (the example numbers, a comma-separated string
  such as `"16,8,64"`, in the order shown) and `target` (the number asked
  about, 1-100), and the response `rating` (1 = yes). Every value reaches your
  model as the string the CSV holds. Your model computes every stimulus
  quantity it needs from the raw columns itself (see *Computing features from
  the raw stimulus* below), and **each `pm.Data` name must match a feature
  your hook returns** (or be the response `rating`); the pipeline maps
  containers to features by name. Initialize each with a **1-element
  placeholder of the correct dtype** (e.g. `np.zeros(1, dtype="int64")`); the
  pipeline calls `pm.set_data(...)` to fill in real data before sampling. Do
  **not** use `np.zeros(0, ...)`.
- Put **priors** on every free cognitive parameter (e.g. `pm.HalfNormal`,
  `pm.Beta`, `pm.Normal`). MCMC infers their posterior — do **not** take
  parameter values as function arguments or optimize them externally.
- Expose the per-trial probability of answering yes as
  `pm.Deterministic("p_yes", ...)`, one value per trial: the design, the
  novelty check and the evaluation read predictions from it.
- Define the **likelihood** as one `pm.Bernoulli` over the response whose
  probability is exactly `p_yes` (e.g.
  `y = pm.Data("rating", ...); pm.Bernoulli("response", p=p_yes, observed=y)`).
  The `observed=` argument must be the **exact `pm.Data` tensor** for the
  response. Do **not** wrap, copy, reorder or derive a new variable from it
  (`1 - y` is not the responses), and apply any lapse, mixture or bias to
  `p_yes` itself, not only inside the likelihood. Admission checks this data
  contract without sampling — the observed data are `rating` in row order,
  `p_yes` has one entry per trial, and the likelihood's probability of each
  response is Bernoulli(`rating`; `p_yes`) — and rejects a model that breaks
  it. No `pm.Potential` may depend on the responses.

**Allowed imports (enforced — candidates that import anything else are rejected
at admission):** `arviz`, `collections`, `dataclasses`, `functools`,
`itertools`, `math`, `numpy`, `operator`, `pymc`, `pytensor`, `re`, `scipy`,
`statistics`, `typing`. No other imports — no `pandas`, no `src.*`, no project
library modules. Every helper your model needs must be written in the file
itself (self-contained code only).

Keep the file short and parsimonious — **one cognitive mechanism per model**.
The number of free parameters and features a model computes should match the
single hypothesis; a model that needs many weighted cues to fit is a blend, not
a hypothesis.

### Computing features from the raw stimulus (required)

Every model computes its own features from the raw stimulus, so that it can
express exactly the statistic its hypothesis needs — the examples' magnitude,
spacing, digits, order or number, and how the target relates to them. Add one
of two module-level hooks to `candidate.py`:

```python
def compute_features(number_set: str, target: str) -> dict:
    """Return numeric features for one (set, target) stimulus."""
    ...
```

The pipeline calls it on the raw `number_set`/`target` strings for every
trial and exposes each returned key as a value you read with a matching
`pm.Data`. For example, whether the target shares its final digit with the
last example shown (a recency cue):

```python
def compute_features(number_set, target):
    examples = [int(x) for x in number_set.split(",")]
    return {"same_last_digit": 1.0 if int(target) % 10 == examples[-1] % 10 else 0.0}


# ... then inside the model:
#   same_last_digit = pm.Data("same_last_digit", np.zeros(1, dtype="float64"))
```

Rules for `compute_features`: it must return a dict of **finite numbers** with
the **same keys for every stimulus**, and those keys must be **new names** (not
the CSV's own column names). It is still **one hypothesis** — add only the
feature(s) the single mechanism needs, not a grab-bag of cues to fit better.

The alternative hook, `prepare_observed(rows: list[dict]) -> dict[str,
np.ndarray]`, builds every `pm.Data` array (the response `rating` included)
at once from the full list of rows, for a model whose data do not fit one
value per container per trial — for example a table of the distinct example
sets plus per-trial indices into it. Use one hook, not both.

**Keep the fit fast.** Each sampling run of admission's fit must finish within
the time limit in `CONTEXT.md`, and one example set is asked about with many
targets. Anything that does not depend on the model's parameters — which
numbers a candidate concept contains, how likely the examples are under it —
belongs in `prepare_observed`, computed once per distinct set, not inside the
model where every gradient step recomputes it.

### Numerical safety (required)

Your model must evaluate to a **finite** log-probability — a model whose `p_yes`
or likelihood is NaN or `-inf` is rejected (it would crash MCMC at its
start-value check). So:

- Keep `p_yes` strictly inside `(0, 1)`. A `sigmoid`/`softmax` already does
  this; if you build a probability another way, clamp `p_yes` itself with
  `pt.clip(p, 1e-6, 1 - 1e-6)`. People answer "no" even to targets that are
  among the examples, so a model that puts `p_yes` at exactly 1 anywhere has a
  log-likelihood of `-inf`.
- Use `pt.abs(x)` for absolute value — **not** `pt.sqrt(x ** 2)`, which returns
  NaN in PyTensor for some inputs.
- Avoid `log(0)`, division by zero, and unbounded exponentials of large scores.

## Example skeleton

```python
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    """How far the target is from the nearest example, on the 1-100 scale."""
    examples = [int(x) for x in number_set.split(",")]
    return {"distance": min(abs(int(target) - x) for x in examples) / 100.0}


with pm.Model() as model:
    # Stimulus input — the name matches the key compute_features returns.
    distance = pm.Data("distance", np.zeros(1, dtype="float64"))

    # Free cognitive parameters with priors (inference fits them): how likely
    # a target right at an example is judged to belong, and how fast that
    # falls off with distance.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    falloff = pm.HalfNormal("falloff", sigma=20.0)

    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept - falloff * distance))

    # Observed response: the pm.Data tensor is passed directly to observed=.
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
```

## Self-check

Before stopping, confirm all three files exist at the absolute paths given in
your instructions — `hypothesis.md` (non-empty), `model_name.txt` (one
snake_case line) and `candidate.py` — then run the check command documented in
`CONTEXT.md`. It loads `candidate.py` the way admission does (a module-level
`model: pm.Model`), binds the real responses, checks the log-probability is
finite and the data contract holds, completes a short MCMC fit and checks ELPD-LOO is finite, printing
`OK` or the exact reason admission would reject the file. Fix anything it
reports before you finish. A candidate with no `hypothesis.md` is rejected; a
missing or invalid `model_name.txt` demotes your model to an auto-generated
name.
