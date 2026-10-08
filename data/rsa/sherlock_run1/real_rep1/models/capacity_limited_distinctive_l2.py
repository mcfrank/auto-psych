"""RSA depth-2 pragmatic listener with distinctiveness, multimodal salience, graded semantics, utterance costs, and capacity-limited attention across display set size.

Refining distinctive_multimodal_l2_listener by incorporating capacity limitation from
capacity_limited_listener: listeners have bounded cognitive resources when processing
visual displays, causing referential reasoning precision (speaker rationality alpha) to
decay as a power law of display set size alpha = alpha_base * (2 / N_OBJ)^gamma.
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
    "w_valence": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_extension": dist.Normal(0.0, 1.0),
    "beta_graded": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., weights: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) + vec(weights, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., weights: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior, weights) + {EPS}) + vec(weights, u)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feature_freq = jnp.sum(real_lex, axis=1)
    specificity = jnp.where(feature_freq > 0, 1.0 / jnp.maximum(feature_freq, 1.0), 0.0) * (1.0 - ctx.is_sink)
    distinctiveness = jnp.sum(real_lex * specificity[:, None], axis=0)

    prior = softmax_prior(
        (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
        + params["w_familiar"] * ctx.familiarization
        + params["w_distinct"] * distinctiveness
    )
    dilution = jnp.exp(-params["beta_graded"] * jnp.maximum(ctx.feature_count - 1.0, 0.0))
    is_real = (1.0 - ctx.is_sink)[:, None]
    graded_lex = jnp.where(is_real > 0, ctx.lex * dilution[None, :], ctx.lex)
    weights = params["w_extension"] * (1.0 - ctx.is_sink) * (jnp.sum(ctx.lex, axis=1) - 1.0)
    n_obj = ctx.lex.shape[1]
    alpha = params["alpha_base"] * (2.0 / n_obj) ** params["gamma"]
    heard = L2(alpha, graded_lex, prior, weights)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
