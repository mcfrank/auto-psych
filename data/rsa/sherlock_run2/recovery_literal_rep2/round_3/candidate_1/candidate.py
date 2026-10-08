"""Evaluative prominence listener: reference resolution via contextual prominence alignment.

Listeners evaluate candidate referents through contextual prominence alignment,
where the pragmatic role of unmentioned features is modulated by communicative framing.
Under neutral framing, unmentioned features incur an omission penalty; under evaluative
framing ('least favorite'), unmentioned features represent compounding negative attributes
that reverse this penalty and favor feature-rich referents. Visual color contrast provides
an orthogonal perceptual prominence boost across both prior and communicative settings.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.LogNormal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_evaluative[u: UTT, r: OBJ](coeff, lex: ..., unmentioned: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(coeff * at(unmentioned, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_familiar"] * ctx.familiarization
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    unmentioned = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    coeff = -params["theta"] - params["w_valence"] * ctx.valence
    heard = L_evaluative(coeff, ctx.lex, unmentioned, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
