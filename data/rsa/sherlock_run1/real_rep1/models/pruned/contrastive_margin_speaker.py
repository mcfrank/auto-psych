"""RSA pragmatic listener with a contrastive margin speaker.

Speakers choose descriptions not merely to maximize the absolute probability
of the intended referent, but to maximize the discriminative contrast margin
between the intended referent and the nearest competing referent. Pragmatic
listeners invert this competitive speaker, favoring referents for which the
heard word creates the sharpest discriminatory advantage against alternative candidates.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., margin: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * at(margin, u, r))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., margin: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * at(margin, u, r))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def calc_margin(probs):
    n_obj = probs.shape[-1]
    eye = jnp.eye(n_obj, dtype=bool)[None, :, :]
    masked = jnp.where(eye, -1.0, probs[:, None, :])
    max_comp = jnp.max(masked, axis=-1)
    return probs - max_comp


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    l0_probs = L0(ctx.lex, prior)
    margin1 = calc_margin(l0_probs)
    l1_probs = L1(params["alpha"], ctx.lex, prior, margin1)
    margin2 = calc_margin(l1_probs)
    heard = L2(params["alpha"], ctx.lex, prior, margin2)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
