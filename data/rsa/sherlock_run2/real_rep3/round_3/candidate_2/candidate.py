"""Perceptual distinctiveness heuristic listener.

Listeners interpret referring expressions using a direct perceptual distinctiveness
heuristic rather than recursive Theory of Mind. When hearing a word, listeners
select among the objects of which the word is literally true by choosing the
object that is most visually distinctive in the context, defined by continuous
feature contrast against the surrounding items. On uninformative trials without
an informative word, listeners choose the most visually distinctive object across
the entire display.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta, lex: ..., distinctiveness: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(beta * vec(distinctiveness, r)))
    return Pr[listener.r == r]


def compute_distinctiveness(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = ctx.lex.shape[1]
    mean_dist = jnp.sum(dist, axis=1) / (n_obj - 1.0)
    return mean_dist


def choice_probs(params, ctx):
    dist_scores = compute_distinctiveness(ctx)
    prior = softmax_prior(params["beta"] * dist_scores)
    heard = L_heur(params["beta"], ctx.lex, dist_scores)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
