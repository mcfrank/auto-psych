"""Pragmatic listener reasoning about an aspect-focused Question Under Discussion (QUD).

Speakers in reference games communicate to address a Question Under Discussion regarding
a salient, distinguishing aspect of the target referent rather than its coordinate identity.
When an intended referent possesses multiple candidate features, communicative speakers select
which aspect to highlight in proportion to its contextual distinctiveness across display competitors,
and then choose referring expressions that are informative about that chosen aspect. Pragmatic
listeners invert this aspect-guided speaker, jointly inferring the speaker's communicative goal and
intended referent.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_aspect": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_qud[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., aspect_prob: ..., aspect_salience: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) * vec(aspect_salience, g)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(aspect_prob, u, g)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    word_counts = jnp.sum(ctx.lex, axis=1, keepdims=True)
    L0_mat = ctx.lex / jnp.maximum(word_counts, 1.0)
    aspect_prob = L0_mat @ ctx.lex.T

    counts = jnp.sum(ctx.lex, axis=1)
    aspect_distinctness = -jnp.log(jnp.maximum(counts, 1.0))
    aspect_salience = jnp.exp(params["w_aspect"] * aspect_distinctness)

    aspect_prominence = jnp.sum(ctx.lex * aspect_distinctness[:, None], axis=0)
    prior = softmax_prior(params["w_aspect"] * aspect_prominence)

    heard = L_qud(
        params["alpha"], ctx.lex, prior, aspect_prob, aspect_salience
    )[ctx.utterance]

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
