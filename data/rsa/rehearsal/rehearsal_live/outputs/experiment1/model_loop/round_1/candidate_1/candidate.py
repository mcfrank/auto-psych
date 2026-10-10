"""Informational preemption listener.

Listeners resolve referring expressions through informational preemption rather than
recursive Theory of Mind. When hearing a word that applies to multiple referents,
listeners penalize candidate objects in proportion to the discriminative specificity
of their unmentioned alternative features, inferring that an object possessing a
superior, uniquely diagnostic description would not be referred to with a shared
label. On uninformative trials with no informative word, choices follow intrinsic
feature distinctiveness, favoring unique items over featureless or duplicate items.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_preempt[u: UTT, r: OBJ](beta, lex: ..., preemption: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta * at(preemption, u, r)),
    )
    return Pr[listener.r == r]


def compute_preemption(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1)
    safe_ext = jnp.maximum(ext, 1.0)
    spec = jnp.where(ext > 0, 1.0 / safe_ext, 0.0)
    distinctiveness = jnp.sum(real_lex * spec[:, None], axis=0)
    preemption = jnp.maximum(distinctiveness[None, :] - real_lex * spec[:, None], 0.0)
    return preemption, distinctiveness


def choice_probs(params, ctx):
    preemption, distinctiveness = compute_preemption(ctx)
    prior = softmax_prior(params["w_prior"] * distinctiveness)
    heard = L_preempt(params["beta"], ctx.lex, preemption)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
