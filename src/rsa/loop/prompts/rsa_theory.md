# Cognitive hypothesis → memo model (RSA inner loop)

You are proposing one candidate **hypothesis** about how people interpret a
speaker in a reference game. You state the hypothesis in plain English, name
it, and translate that single hypothesis into a model written in **memo**, a
probabilistic programming language for agents reasoning about agents. The
pipeline fits your model to every trial with MCMC (numpyro NUTS) and compares
it with the other models by ELPD-LOO.

Read these documents before deciding what to write. Each is included below in
this prompt and saved in your candidate directory:

1. `CONTEXT.md`: paths, the experiments, what a trial looks like, the
   responses CSV, and the inner-loop round.
2. `CANDIDATE_BRIEF.md`: what kind of hypothesis to attempt in this slot.
3. `existing_hypotheses.md`: the models in the set, with each one's
   hypothesis, standing (ELPD-LOO behind the best model ± SE) and source file.
4. `attempted_hypotheses.md` (exploratory slots): hypotheses tried earlier
   that are no longer in the set, and what happened to each. Do not propose
   any of them again under a new name.
5. `refinement_menu.md` (refinement slots): the models you may refine.
6. `memo_handbook.md`: the complete memo language reference. Read the parts
   you need. The "memo in this pipeline" section below says how this pipeline
   uses memo, and it takes precedence.

## The task people did

On each trial a participant sees a few objects side by side (e.g. three faces;
each face has or lacks some features such as a hat, glasses or a mustache).
They are told that someone ("Bob") can say only one word and that he said,
e.g., **"glasses"**, and they click the object they think he means. On *prior*
trials no informative word is given ("mumble mumble", "which will he pick
next?"), so the choice shows which objects people expect a priori.

The classic finding: with a plain face, a face with glasses, and a face with
glasses and a hat, people hearing "glasses" mostly pick the face with only
glasses, because a speaker meaning the other one would have said "hat". Some
experiments manipulate how often each object was seen before (familiarization
base rates), which object is shown in colour, or whether Bob describes his
*favorite* or *least favorite* one.

## Goal

Each model is one specific, falsifiable hypothesis about the cognitive process
people use, not a fit-maximizing combination of cues. Articulate one such
hypothesis and implement exactly that.

**Do not build a mixture of heuristics.** That means no averaging, weighting or
softmax-blending of mechanisms drawn from several hypotheses into one model; it
is not a hypothesis and will be rejected. Refining a *single* hypothesis is
encouraged: a different functional form, prior, depth or normalization of the
**same** mechanism. In a *refinement* slot (your brief says so) you may extend
the model you refine with one component from another model, as a single
stated change.

## Step 1: `hypothesis.md`

Write `hypothesis.md`: 1–3 plain-English sentences naming the single mechanism
you claim people use and how it drives their choice. Use no code or math; a
psychologist should read it as one clear, testable claim.

## Step 2: `model_name.txt`

Write `model_name.txt`: one line, a short **snake_case** name (lowercase
letters, digits, underscores; 3–40 characters) that says what the mechanism is,
e.g. `costly_feature_speaker` or `valence_salience_listener`. Do not reuse a
name from `existing_hypotheses.md`, and do not use a generic name like `new_model`.

## Step 3: `candidate.py`

A model file defines exactly two things the pipeline reads:

```python
PARAMS = {"alpha": dist.LogNormal(0.0, 1.0), "lapse": dist.Beta(1.0, 9.0)}

def choice_probs(params, ctx):
    ...  # -> probability of choosing each object on this trial, shape (N_OBJ,)
```

* `PARAMS` is a non-empty dict from parameter name to a **numpyro
  distribution**, the prior. Every parameter is a scalar shared by all trials.
* `choice_probs(params, ctx)` is called for **one trial**. `params[name]` is a
  scalar, and `ctx` holds that trial's arrays. It returns a JAX array of shape
  `(N_OBJ,)`: non-negative, summing to 1, the probability that the participant
  clicks each object. The pipeline owns the likelihood (a categorical choice
  per trial), vectorizes your function over trials with `jax.vmap`, and
  differentiates through it with NUTS, so it must be pure JAX: no Python `if`
  on array values (use `jnp.where`), no loops over traced values, and no
  printing or file access.

### What `ctx` holds (one trial)

| field | shape | meaning |
|---|---|---|
| `ctx.lex` | `(N_UTT, N_OBJ)` | 1.0 where utterance `u` is literally true of object `r` |
| `ctx.is_sink` | `(N_UTT,)` | 1.0 for the *sink* utterance (see below) |
| `ctx.feature_count` | `(N_OBJ,)` | how many of the context's words are true of each object |
| `ctx.utterance` | `()` int | index of the utterance heard (0 on a prior trial; ignore it there) |
| `ctx.is_prior` | `()` | 1.0 on a prior trial (no informative word) |
| `ctx.familiarization` | `(N_OBJ,)` | base rate at which each object was seen before (0..1; all 0 when there was no familiarization phase) |
| `ctx.has_familiarization` | `()` | 1.0 when there was a familiarization phase |
| `ctx.grayscale` | `(N_OBJ,)` | 1.0 for an object shown in grayscale (all 0 when everything was in colour) |
| `ctx.valence` | `()` | +1 "my favorite …", −1 "my least favorite …", 0 neutral |

**Utterances.** A context's utterances are the features true of at least one
object, in feature order. When some object has none of them (the plain face),
a final **sink** utterance ("no word applies") is added, true of exactly those
objects. The sink is never heard. It lets a speaker who means a featureless
object have an option, which keeps every speaker distribution normalisable. A
speaker model should let the speaker choose it like any other utterance; do
not treat it as a real word.

**Shapes.** Contexts have 2–4 objects and 2–5 utterances. The pipeline runs
your file once per context shape with the memo domains `OBJ =
jnp.arange(N_OBJ)` and `UTT = jnp.arange(N_UTT)` **already defined**. Use
them as your memo domains, and **do not define or import `OBJ` or `UTT`**.

**Identical objects.** Two objects with the same features (and the same
familiarization and colour) are one choice for the likelihood: their
probabilities are summed. Your model need not, and cannot, tell them apart.

**People sometimes choose an object the word is false of.** A model that gives
any observed choice probability zero is refused. Include an explicit noise or
lapse process (e.g. `with_lapse(p, params["lapse"])` from the kit) or another
principled source of error.

### memo in this pipeline (read this)

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {"alpha": dist.LogNormal(0.0, 1.0), "lapse": dist.Beta(1.0, 9.0)}

@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]

@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
```

* **Array parameters.** An array a memo needs (the lexicon, a prior over
  objects, per-utterance costs) is a memo parameter declared with `: ...`
  (e.g. `lex: ...`). Inside memo you read it **only** through a jitted helper
  call: `at(lex, u, r)` for a matrix and `vec(prior, r)` for a vector, both
  from `src.rsa.memo_kit`. `lex[u, r]` inside memo is a syntax error, because
  memo parses it as a query about an agent. Scalar parameters (`alpha`) are
  plain.
* **Python constants inside memo** must be escaped in braces: `log(p + {EPS})`.
  A bare `EPS` is read as an agent's choice and memo refuses to compile.
* **`log(0)`.** It gives an infinite value and a NaN gradient, and NUTS cannot
  sample through it. Always `log(p + {EPS})`.
* **All-zero weights.** A `chooses(..., wpp=...)` whose weights are all zero
  for some value of the variables gives zeros in the forward pass and NaN
  gradients. The sink utterance prevents this for speakers over `ctx.lex`.
  When you add your own weights, keep them strictly positive (e.g.
  `exp(...)`) or add `{EPS}`.
* **Recursion depth** must be a fixed Python integer. Write each level as its
  own memo (`L0`, `L1`, `L2`, …) or use a static ternary on a Python int
  parameter; a depth cannot be a fitted parameter.
* **Observing.** `observes [x.u] is u` conditions on a choice name. To condition
  on a value use `observes_that [x.u == 2]`.
* **What a memo returns.** It returns a dense array over its axes, e.g.
  `L1(...)` has shape `(N_UTT, N_OBJ)`. Index it in `choice_probs` with
  `[ctx.utterance]`.
* **Python code outside memo.** Ordinary jitted JAX helpers can compute
  vectors to pass into memo, e.g. a salience prior
  `softmax_prior(w * ctx.feature_count)`, passed as `prior: ...`.
* **Do not use `@memo(cache=True)`.** It breaks differentiation.

### Allowed imports

`jax`, `memo`, `numpyro`, `numpy`, `math`, `itertools`, `functools`,
`collections`, `typing`, `dataclasses`, `enum`, `operator`, and from this
repository **only** `src.rsa.memo_kit`. No file access, no `open`, no
`eval`/`exec`, no `getattr`, no reading the data inside the model. Anything else
is refused before your code runs.

### Parameters and convergence

The fit must pass a convergence gate: at most 0.1% divergent transitions,
R-hat ≤ 1.05, bulk ESS ≥ 100. Give every parameter a proper, weakly
informative prior on a sensible scale. Avoid parameters the data cannot
identify (two parameters that only ever appear as a product; a weight on a
manipulation no trial varies). Rationality `alpha` and lapse rates are
typically identified.

## What the pipeline checks (in order)

1. The import and code gate.
2. The contract, run at several prior draws on the training displays **and on
   every display shape of a broad pool** (2–4 objects × 2–5 utterances, plus
   valence, familiarization and grayscale variants): shape `(N_OBJ,)`, finite,
   non-negative, sums to 1, finite gradient for every parameter.
3. No observed choice gets probability 0.
4. NUTS converges within 30 minutes.
5. ELPD-LOO is finite.
6. **Novelty.** Your posterior-mean predictions must differ from every model
   already in the set (RMSE ≥ 0.002 over the pool). A re-parameterisation of
   an existing mechanism is refused.

## Self-check (do this before you finish)

Run the self-check on your candidate directory from the repository root. It
runs gates 1–5 with a short fit and prints what fails:

```bash
uv run python -m src.rsa.loop.check_candidate <your candidate directory>
```

Fix everything it reports. A candidate that fails a gate is rejected and sent
back once for repair with the reason. A second failure is final.
