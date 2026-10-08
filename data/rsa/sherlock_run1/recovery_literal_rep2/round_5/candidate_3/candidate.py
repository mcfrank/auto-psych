"""Pragmatic listener with contextual typicality prior grounded in prototype theory.

Listeners evaluate candidate referents through contextual typicality, expecting speakers
to refer to the prototypical exemplar of the visual ensemble. Grounded in prototype
theory (Rosch & Mervis, 1975), an object's typicality reflects its feature overlap with
the other objects in the display: objects that share features widely with competitors
are typical exemplars, whereas objects possessing idiosyncratic or unshared features are
atypical outliers. Before hearing an utterance, listeners expect the speaker to refer to
the prototypical object; upon hearing a referring expression, listeners integrate this
contextual typicality prior with the literal meaning of the word, favoring typical
exemplars over atypical competitors that carry extraneous unshared features.
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
def L_typical[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_feat = jnp.sum(real_lex, axis=-1)
    shared_count = jnp.maximum(n_feat - 1.0, 0.0)
    is_unique = jnp.where((n_feat == 1.0) & (ctx.is_sink == 0.0), 1.0, 0.0)
    shared_score = jnp.sum(real_lex * shared_count[:, None], axis=0)
    unique_score = jnp.sum(real_lex * is_unique[:, None], axis=0)
    typicality = shared_score - unique_score

    prior = softmax_prior(params["beta"] * typicality)
    heard = L_typical(ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
