# Cognitive Hypothesis → PyMC Model (inner loop)

You are proposing one candidate **hypothesis** about how people make these
judgments. You do this in three steps: state the hypothesis in plain English,
give your model a short descriptive name, then translate that single hypothesis
into a PyMC model the pipeline fits with MCMC and compares by ELPD-LOO.

Read these documents before deciding what to write. Each is included below in
this prompt and saved in your candidate directory:

1. `CONTEXT.md` — paths, the responses CSV's columns (the raw H/T sequences and
   the response only), and the inner-loop round number.
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
  the stimulus. The responses CSV has no feature columns — only the raw H/T
  sequence strings `sequence_a`/`sequence_b`, which are **not** numeric and
  cannot be a `pm.Data` directly, and the response `chose_left`. Your model
  computes every stimulus quantity it needs from the raw sequences itself (see
  *Computing features from the raw sequences* below), and **each `pm.Data`
  name must match a feature your hook returns** (or be the response
  `chose_left`); the pipeline maps containers to features by name. Initialize
  each with a **1-element placeholder of the correct dtype** (e.g.
  `np.zeros(1, dtype="int64")`); the pipeline calls `pm.set_data(...)` to fill
  in real data before sampling. Do **not** use `np.zeros(0, ...)`.
- Put **priors** on every free cognitive parameter (e.g. `pm.HalfNormal`,
  `pm.Beta`, `pm.Normal`). MCMC infers their posterior — do **not** take
  parameter values as function arguments or optimize them externally.
- Expose the per-trial probability of choosing the left sequence as
  `pm.Deterministic("p_left", ...)`, one value per trial: the design, the
  novelty check and the evaluation read predictions from it.
- Define the **likelihood** as one `pm.Bernoulli` over the response whose
  probability is exactly `p_left` (e.g.
  `y = pm.Data("chose_left", ...); pm.Bernoulli("response", p=p_left, observed=y)`).
  The `observed=` argument must be the **exact `pm.Data` tensor** for the
  response. Do **not** wrap, copy, reorder or derive a new variable from it
  (`1 - y` is not the responses), and apply any lapse, mixture or bias to
  `p_left` itself, not only inside the likelihood. Admission checks this data
  contract without sampling — the observed data are `chose_left` in row
  order, `p_left` has one entry per trial, and the likelihood's probability of
  each response is Bernoulli(`chose_left`; `p_left`) — and rejects a model that
  breaks it. No `pm.Potential` may depend on the responses.

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

### Computing features from the raw sequences (required)

Every model computes its own features from the raw sequences, so that it can
express exactly the statistic its hypothesis needs — including where in a
sequence something happens, the specific sub-sequences it contains, or recency.
Add one of two module-level hooks to `candidate.py`:

```python
def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return numeric features for one stimulus pair."""
    ...
```

The pipeline calls it on the raw `sequence_a`/`sequence_b` strings for every
trial and exposes each returned key as a value you read with a matching
`pm.Data`. For example, whether the *last* toss of each sequence is heads (a
recency cue):

```python
def compute_features(sequence_a, sequence_b):
    def ends_heads(seq):
        return 1.0 if seq.strip().upper().endswith("H") else 0.0

    return {
        "ends_heads_a": ends_heads(sequence_a),
        "ends_heads_b": ends_heads(sequence_b),
    }


# ... then inside the model:
#   ends_heads_a = pm.Data("ends_heads_a", np.zeros(1, dtype="float64"))
```

Rules for `compute_features`: it must return a dict of **finite numbers** with
the **same keys for every sequence pair**, and those keys must be **new names**
(not the CSV's own column names). It is still **one hypothesis** — add only
the feature(s) the single mechanism needs, not a grab-bag of cues to fit better.

The alternative hook, `prepare_observed(rows: list[dict]) -> dict[str,
np.ndarray]`, builds every `pm.Data` array (the response `chose_left`
included) at once from the full list of rows, for a model whose data do not
fit one value per container per trial — for example a table of distinct
sequences plus per-trial indices into it. Use one hook, not both.

### Participant effects (if your hypothesis has person-specific parameters)

The responses CSV's `participant_id` is an integer that keeps counting across
the run's experiments, so each experiment adds larger ids. A model with
person-specific parameters binds it as
`participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))` and
indexes its per-person parameters with it directly: give those parameters a
population distribution (e.g. `mu + sigma * z` with `z` of `shape=400`), size
them well beyond the participants so far (`shape=400`), and never renumber
participants inside a hook. The next experiment is designed for **new**
participants: the design predicts your model with the slots no participant's
data have reached, which are draws from your population distribution. A model
without spare slots, or one whose hook renumbers participants, cannot predict
a new participant and is left out of the design.

### Numerical safety (required)

Your model must evaluate to a **finite** log-probability — a model whose `p_left`
or likelihood is NaN or `-inf` is rejected (it would crash MCMC at its
start-value check). So:

- Keep `p_left` strictly inside `(0, 1)`. A `sigmoid`/`softmax` already does
  this; if you build a probability another way, clamp `p_left` itself with
  `pt.clip(p, 1e-6, 1 - 1e-6)`.
- Use `pt.abs(x)` for absolute value — **not** `pt.sqrt(x ** 2)`, which returns
  NaN in PyTensor for some inputs.
- Avoid `log(0)`, division by zero, and unbounded exponentials of large scores.

## Example skeleton

```python
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    """Each sequence's alternation rate: the share of adjacent flips that differ."""

    def alternation_rate(seq):
        seq = seq.strip().upper()
        switches = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
        return switches / (len(seq) - 1)

    return {"alt_a": alternation_rate(sequence_a), "alt_b": alternation_rate(sequence_b)}


with pm.Model() as model:
    # Stimulus inputs — names match the keys compute_features returns.
    alt_a = pm.Data("alt_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameter with a prior (inference fits it): how strongly
    # more alternation makes a sequence look random (negative: less random).
    beta = pm.Normal("beta", mu=0.0, sigma=2.0)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (alt_a - alt_b)))

    # Observed response: the pm.Data tensor is passed directly to observed=.
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
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
