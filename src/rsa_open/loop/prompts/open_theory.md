# Cognitive hypothesis → a theory of reference across paradigms (open campaign)

You are proposing one candidate **theory** of how people interpret a speaker, and choose
what to say, in reference games. You state the hypothesis in plain English, name it, and
translate that single hypothesis into **memo**, a probabilistic programming language for
agents reasoning about agents. The pipeline fits your theory to **every paradigm of this
chain at once** with MCMC (numpyro NUTS) and compares it with the other theories by
cross-validation on each paradigm.

Read these documents before deciding what to write. Each is included below in this prompt
and saved in your candidate directory:

1. `CONTEXT.md`: paths, the paradigms and their data, and the inner-loop round.
2. `CANDIDATE_BRIEF.md`: what kind of hypothesis to attempt in this slot.
3. `existing_hypotheses.md`: the theories in the set, with each one's hypothesis, standing
   and source file.
4. `attempted_hypotheses.md` (exploratory slots): hypotheses tried earlier that are no
   longer in the set. Do not propose any of them again under a new name.
5. `refinement_menu.md` (refinement slots): the theories you may refine.
6. `memo_handbook.md`: the complete memo language reference. The "memo in this pipeline"
   section below takes precedence over it.

## A theory, not a model of one task

The same people-level ability shows up in different tasks. In this campaign each experiment
uses a **paradigm** chosen to test the theories: forced choice (click the object the speaker
means), ratings or a slider (how likely is the speaker to mean this object), a bet (split
points over the objects), production (which word would you use for this object), or a
judgment (would a speaker say this word for this object). `CONTEXT.md` lists this chain's
paradigms, the literature's forced-choice data (E) first.

Your theory says what is in the person's head, once, for every paradigm:
- the **listener**: given a scene and a heard word, a posterior over which object the speaker
  means;
- the **speaker**: given a scene and a target object, a distribution over which word to say.

Each paradigm's **link** turns those into responses, with parameters of its own that the
pipeline fits per paradigm: a decision temperature (how sharply a person's responses follow
the theory's probabilities), a lapse (random responding), and for ratings and sliders their
cutpoints. **The response process is the link's, not the theory's**: a theory with a lapse
or cutpoints is refused, and a theory need not guard against responses it gives probability
zero (the link's lapse does).

**E is the reference paradigm: its temperature is fixed at 1.** On E the theory's listener
probabilities are the choice probabilities (with E's lapse), and each later paradigm's
temperature is measured relative to E. So if you hold that people sharpen or soften their
beliefs into a choice (they maximize rather than probability-match, say), that is a claim
about people, not about one task: put it in the theory as a parameter of its own (any name
but the links', e.g. `beta`), and it then applies, and is tested, in every paradigm.

A theory that fits one paradigm by giving up on another is not progress. A theory is ranked
on the newest paradigm, and it may lead only while, on **every earlier paradigm, E
included**, it is within 4 standard errors of the best theory there.

Each model is one specific, falsifiable hypothesis about the cognitive process, not a
fit-maximizing combination of cues. **Do not build a mixture of heuristics**: no averaging,
weighting or softmax-blending of mechanisms drawn from several hypotheses into one theory.
Refining a *single* hypothesis is encouraged. In a *refinement* slot (your brief says so) you
may extend the theory you refine with one component from another, as a single stated change.

## Step 1: `hypothesis.md`

1–3 plain-English sentences naming the single mechanism you claim people use, as listeners
and as speakers. No code or math.

## Step 2: `model_name.txt`

One line, a short **snake_case** name (lowercase letters, digits, underscores; 3–40
characters) that says what the mechanism is. Do not reuse a name from
`existing_hypotheses.md`.

## Step 3: `candidate.py`

```python
PARAMS = {"alpha": dist.LogNormal(0.0, 1.0)}

def listener(params, ctx):
    ...  # -> P(referent | heard word) for this trial, shape (N_OBJ,)

def speaker(params, ctx):   # optional
    ...  # -> column r is P(word | target r), shape (N_UTT, N_OBJ)
```

* `PARAMS` maps parameter names to **numpyro distributions** (the priors); every parameter
  is a scalar shared by all trials of every paradigm. It may be empty.
* `listener(params, ctx)` is called for **one trial** and returns a JAX array of shape
  `(N_OBJ,)`: non-negative, summing to 1. On a prior trial (`ctx.is_prior`) it is the
  listener's expectation before any word.
* `speaker(params, ctx)` is optional. Column `r` is the speaker's distribution over the
  scene's words when the target is object `r`: non-negative, summing to 1 over the real
  words, with probability 0 on the sink (the last utterance when `ctx.is_sink` marks one).
  **Without it, the speaker is derived from your listener**: S(word | r) ∝ listener(r |
  word), over the real words, the speaker who expects your listener. Define your own when
  your hypothesis says something about speakers that this does not capture (costs, a
  different goal, alternatives they consider).
* Both are vectorized over trials with `jax.vmap` and differentiated by NUTS, so they must be
  pure JAX: no Python `if` on array values (use `jnp.where`), no loops over traced values, no
  printing or file access.

### What `ctx` holds (one trial)

| field | shape | meaning |
|---|---|---|
| `ctx.lex` | `(N_UTT, N_OBJ)` | 1.0 where utterance `u` is literally true of object `r` |
| `ctx.is_sink` | `(N_UTT,)` | 1.0 for the *sink* utterance (see below) |
| `ctx.feature_count` | `(N_OBJ,)` | how many of the scene's words are true of each object |
| `ctx.utterance` | `()` int | index of the utterance heard (0 on a prior trial; ignore it there) |
| `ctx.is_prior` | `()` | 1.0 on a prior trial (no informative word) |
| `ctx.familiarization`, `ctx.has_familiarization`, `ctx.grayscale`, `ctx.valence` | | always zero in this campaign's scenes (plain displays) |

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

**Identical objects.** Two objects with the same features are one referent for every
response: their probabilities are summed. Your theory need not, and cannot, tell them
apart.

### memo in this pipeline (read this)

```python
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, softmax_prior

PARAMS = {"alpha": dist.LogNormal(0.0, 1.0)}

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

def listener(params, ctx):
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return jnp.where(ctx.is_prior > 0, uniform, heard)
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
  `L1(...)` has shape `(N_UTT, N_OBJ)`. Index it in `listener` with
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

The fit must pass a convergence gate: at most 0.1% divergent transitions, R-hat ≤ 1.05, bulk
ESS ≥ 100. Give every parameter a proper, weakly informative prior on a sensible scale. Avoid
parameters the data cannot identify: two of the theory's own that rescale the same
probabilities (a speaker's rationality and a listener's sharpness can trade off on
forced-choice data alone) need a reason in the data, or one of them should go.

## What the pipeline checks (in order)

1. The import and code gate (`jax`, `memo`, `numpyro`, `numpy`, a few stdlib modules and
   `src.rsa.memo_kit` only; no file access).
2. It is a theory: it defines `listener`, no `choice_probs`, and no parameter named like a
   link's (`temperature`, `lapse`, `center`, `spread`, `kappa`, `cut_low`, `cut_gap`, `phi`).
3. The listener contract and the speaker contract (your own speaker or the derived one), at
   several prior draws, on every scene of every paradigm **and on a broad pool of scenes**
   (2-4 objects × 2-4 features, with prior trials and features that share an extension):
   shapes, finite, non-negative, sums to 1, the sink never said, finite gradients.
4. The fit to every paradigm at once converges within 30 minutes.
5. **Novelty.** Your posterior-mean listener and speaker predictions on the pool must differ
   from every theory already in the set (RMSE ≥ 0.002).

## Self-check (do this before you finish)

Run the self-check with the exact command your `CONTEXT.md` gives (the pipeline's own Python;
`uv` is not available to you). It runs gates 1–4 with a short fit and prints what fails. It
compiles and fits your theory on every paradigm, which takes 3–15 minutes: **give the shell
tool a timeout of 1200000 milliseconds**. Run it once per change and read its PASS/FAIL line.

Write, edit and check your three files **in your candidate directory**, from the start. The
loop reads only that directory. A candidate that fails a gate is sent back once for repair
with the reason; a second failure is final.
