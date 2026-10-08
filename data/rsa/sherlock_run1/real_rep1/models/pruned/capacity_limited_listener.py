"""RSA pragmatic listener with capacity-limited attention across display set size.

Listeners have bounded attentional resources when processing visual displays.
In accordance with continuous-resource models of visual working memory, the
precision of referential reasoning (speaker rationality alpha) decays as a power
law of the number of objects in the display: alpha = alpha_base * (2 / N_OBJ)^gamma.
Listeners perform sharp pragmatic inferences on minimal two-object displays, but
become more stochastic and bounded as the visual scene contains more candidate objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha_base": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Beta(1.0, 1.0),
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
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    n_obj = ctx.lex.shape[1]
    alpha = params["alpha_base"] * (2.0 / n_obj) ** params["gamma"]
    heard = L1(alpha, ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
