"""Most distinctive listener: non-Bayesian heuristic listener choosing the most contextually distinctive referent.

Listeners interpret referring expressions using a non-Bayesian contextual distinctiveness heuristic
rather than recursive Theory of Mind. When hearing a word that applies to multiple referents, listeners
select among matching candidates by choosing the object that is most contextually distinctive—exhibiting
the greatest average feature contrast relative to the other objects in the visual scene. On uninformative
prior trials without an informative word, choices are similarly governed by contextual distinctiveness
across all scene objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_distinct[u: UTT, r: OBJ](beta, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * vec(distinct, r)),
    )
    return Pr[listener.r == r]


def compute_distinctiveness(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    n_obj = ctx.lex.shape[1]
    n_other = jnp.maximum(1.0, n_obj - 1.0)
    return jnp.sum(diff, axis=(0, 2)) / n_other


def choice_probs(params, ctx):
    distinct = compute_distinctiveness(ctx)
    prior = softmax_prior(params["beta"] * distinct)
    heard = L_distinct(params["beta"], ctx.lex, distinct)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
