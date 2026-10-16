"""Consolidated RSA listener: the mechanisms of the campaign's three final models, each switchable.

Written for `src.rsa.consolidate`: a variant model file is this source plus a
footer that calls ``make(MECHANISMS)`` (`src.rsa.consolidate.variants`), so a
variant's file hash covers this code and the fit cache stays honest.

The core is plain RSA (Frank & Goodman, 2012): a literal listener, a speaker
choosing a true word with probability proportional to exp(alpha * utility), a
listener inverting that speaker under a prior over objects, and a lapse to a
uniform guess. With no mechanism on it is the starting model ``rsa_l1``.

The mechanisms (``MECHANISMS``; where each comes from):

* ``depth2``: a second level of recursion (chains 0 and 2).
* prior over objects (all three chains, each in its own form):
  ``dup`` duplicated objects are less likely referents (log of copies);
  ``richness`` feature count, linear, and its within-display extremity (a
  U-shape over the range, chain 0's interior suppression, chain 1's focal and
  chain 2's endpoint terms as one curve); ``empty`` a bonus for an object with
  no features (chains 1 and 2); ``surprisal`` summed feature surprisal (chain 1).
* speaker: ``redundancy`` synonyms count as k^-w alternatives (chain 2);
  ``aspect`` a speaker who, with probability p, names the sparsest-matching
  object's feature without trying to discriminate (chain 1); ``interior`` a
  penalty on words for objects inside a word's feature-count range (chain 0);
  ``avoidable`` a penalty on an ambiguous word when a unique one existed
  (chain 0); ``competition`` lower speaker precision for objects with more
  features (chain 2).
* noise: ``confusion`` a literal listener that confuses similar objects
  (chain 1; lambda 0 is plain RSA); ``binding`` illusory feature binding onto
  non-matching objects (chain 2); ``attention`` inattention that grows with
  display size, captured by the prior over matching objects (chain 0).

A mechanism that is off takes its neutral value (``OFF``): every weight 0, no
aspect speaker, lambda 0, no binding, full attention.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist

from src.rsa.memo_kit import EPS, softmax_prior, with_lapse

CORE_PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}
MECHANISM_PARAMS = {
    "depth2": {},
    "dup": {"w_dup": dist.Normal(0.0, 1.0)},
    "richness": {"w_features": dist.Normal(0.0, 1.0), "w_extreme": dist.Normal(0.0, 1.0)},
    "empty": {"w_empty": dist.Normal(0.0, 1.0)},
    "surprisal": {"w_surprisal": dist.Normal(0.0, 1.0)},
    "redundancy": {"w_redundancy": dist.Normal(0.0, 1.0)},
    "aspect": {"p_aspect": dist.Beta(2.0, 2.0)},
    "interior": {"w_interior": dist.Normal(0.0, 1.0)},
    "avoidable": {"c_avoid": dist.Normal(0.0, 1.0)},
    "competition": {"w_competition": dist.LogNormal(-1.0, 1.0)},
    "confusion": {"lambda_sim": dist.LogNormal(0.0, 0.5)},
    "binding": {"w_binding": dist.Beta(1.0, 24.0)},
    "attention": {"logit_p_attend": dist.Normal(2.0, 1.0), "w_load": dist.Normal(0.5, 0.5)},
}
MECHANISMS = tuple(MECHANISM_PARAMS)
OFF = {
    "w_dup": 0.0, "w_features": 0.0, "w_extreme": 0.0, "w_empty": 0.0, "w_surprisal": 0.0,
    "w_redundancy": 0.0, "p_aspect": 0.0, "w_interior": 0.0, "c_avoid": 0.0, "w_competition": 0.0,
    "lambda_sim": 0.0, "w_binding": 0.0, "logit_p_attend": 0.0, "w_load": 0.0,
}
NO_WORD = -1e30  # a speaker's log weight for a word false of the object


def make(mechanisms):
    """``(PARAMS, choice_probs)`` of the model with ``mechanisms`` on."""
    on = frozenset(mechanisms)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")
    params = dict(CORE_PARAMS)
    for m in MECHANISMS:
        if m in on:
            params.update(MECHANISM_PARAMS[m])

    def choice_probs(p, ctx):
        return _choice(p, ctx, on)

    return params, choice_probs


def equivalence_class(lex, is_sink):
    """Each word's number of words with the same extension (1 for the sink; chain 2)."""
    real = lex * (1.0 - is_sink)[:, None]
    same = jnp.all(jnp.abs(real[:, None, :] - real[None, :, :]) == 0, axis=-1).astype(jnp.float32)
    same = same * (1.0 - is_sink[:, None]) * (1.0 - is_sink[None, :])
    return jnp.where(is_sink > 0, 1.0, jnp.maximum(jnp.sum(same, axis=1), 1.0))


def interior_penalty(real, feature_count):
    """Chain 0's interior suppression: for a word true of 3+ objects whose feature
    counts differ, 4 x(1 - x) at each object's position x in that range."""
    k_u = jnp.sum(real, axis=1)
    hi = jnp.max(jnp.where(real > 0, feature_count[None, :], -999.0), axis=1, keepdims=True)
    lo = jnp.min(jnp.where(real > 0, feature_count[None, :], 999.0), axis=1, keepdims=True)
    span = hi - lo
    x = (feature_count[None, :] - lo) / jnp.maximum(1.0, span)
    eligible = (k_u[:, None] >= 3.0) & (span > 0.0) & (real > 0)
    return (4.0 * x * (1.0 - x) * eligible).astype(jnp.float32)


def avoidable_ambiguity(real, is_sink):
    """Chain 0's avoidable ambiguity: a word's ambiguity, for objects that had a unique word."""
    n_obj = real.shape[1]
    k_u = jnp.sum(real, axis=1)
    ambiguity = (k_u - 1.0) / jnp.maximum(1.0, n_obj - 1.0) * (1.0 - is_sink)
    unique = (k_u == 1.0) * (1.0 - is_sink)
    has_unique = jnp.max(real * unique[:, None], axis=0)
    return real * ambiguity[:, None] * has_unique[None, :]


def _choice(p, ctx, on):
    q = {**OFF, **p}
    lex, sink, fc = ctx.lex, ctx.is_sink, ctx.feature_count
    n_obj = lex.shape[1]
    real = lex * (1.0 - sink)[:, None]

    # Objects that look identical (features, colour, familiarization): as every final model.
    pair = jnp.sum(jnp.abs(lex[:, :, None] - lex[:, None, :]) * (1.0 - sink)[:, None, None], axis=0)
    total = (pair + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
             + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :]))
    copies = jnp.sum(total == 0, axis=1)

    # The prior over objects.
    lo, hi = jnp.min(fc), jnp.max(fc)
    span = hi - lo
    x = (fc - lo) / jnp.maximum(span, 1.0)
    extremity = jnp.where(span > 0, 1.0 - 4.0 * x * (1.0 - x), 0.0)
    surprisal = jnp.sum(real * -jnp.log(jnp.mean(real, axis=1, keepdims=True) + 1e-4), axis=0)
    prior = softmax_prior(
        -q["w_dup"] * jnp.log(copies)
        + q["w_features"] * fc + q["w_extreme"] * extremity
        + q["w_empty"] * (fc == 0)
        + q["w_surprisal"] * surprisal
    )

    # The literal listener; with confusion, similar objects share a word's mass.
    sim = jnp.exp(-q["lambda_sim"] * total)
    l0 = lex / (lex @ sim + EPS)

    # The speaker.
    alt = -q["w_redundancy"] * jnp.log(equivalence_class(lex, sink))
    cost = q["w_interior"] * interior_penalty(real, fc) + q["c_avoid"] * avoidable_ambiguity(real, sink)
    alpha = q["alpha"] / (1.0 + q["w_competition"] * (jnp.maximum(fc, 1.0) - 1.0))
    sparsest = jnp.min(jnp.where(lex > 0, fc[None, :], 999.0), axis=1, keepdims=True)
    aspect_util = -jnp.log(1.0 + jnp.maximum(fc[None, :] - sparsest, 0.0))
    aspect = jax.nn.softmax(jnp.where(lex > 0, alpha[None, :] * aspect_util, NO_WORD), axis=0)

    def speaker(utility):
        ident = jax.nn.softmax(jnp.where(lex > 0, alt[:, None] + alpha[None, :] * (utility - cost), NO_WORD), axis=0)
        return (1.0 - q["p_aspect"]) * ident + q["p_aspect"] * aspect

    def listener(s):
        joint = prior[None, :] * s
        return joint / (jnp.sum(joint, axis=1, keepdims=True) + EPS)

    heard_all = listener(speaker(jnp.log(l0 + EPS)))
    if "depth2" in on:
        heard_all = listener(speaker(jnp.log(heard_all + EPS)))
    heard = heard_all[ctx.utterance]

    # Illusory binding onto objects the word is false of (chain 2).
    false_of = (1.0 - sink[ctx.utterance]) * (1.0 - lex[ctx.utterance])
    bind = false_of * (fc / jnp.maximum(hi, 1.0)) * prior
    bind_total = jnp.sum(bind)
    w_bind = q["w_binding"] * (bind_total > 0)
    heard = (1.0 - w_bind) * heard + w_bind * bind / jnp.maximum(bind_total, EPS)

    # Inattention growing with display size, captured by the prior over matching objects (chain 0).
    if "attention" in on:
        attend = jax.nn.sigmoid(q["logit_p_attend"] - q["w_load"] * (n_obj - 2.0))
        capture = prior * real[ctx.utterance] + 1e-6
        heard = attend * heard + (1.0 - attend) * capture / jnp.sum(capture)

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), q["lapse"])
