"""RSA pragmatic listener with an object prior grounded in diagnostic feature contrast.

People evaluate candidate referents against a prior grounded in diagnostic
feature contrast rather than raw feature count. In visual reference displays,
distinctive features unique to an object heighten its perceptual and
communicative salience, whereas features shared with competitor objects create
ambiguity and dilute attention. Pragmatic listeners treat this balance of
unique versus shared features as common-knowledge prior expectations over
topics, favoring distinctive referents before and during pragmatic
communication.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    non_sink_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_u = jnp.sum(non_sink_lex, axis=-1, keepdims=True)
    is_unique = jnp.where(n_u == 1.0, 1.0, 0.0)
    is_shared = jnp.where(n_u > 1.0, 1.0, 0.0)
    contrast = jnp.sum(non_sink_lex * is_unique, axis=0) - jnp.sum(
        non_sink_lex * is_shared, axis=0
    )
    prior = softmax_prior(
        params["w_contrast"] * contrast + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
