"""Framed alignment listener: inverting a framed, accessibility-sensitive speaker."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "gamma_acc": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](beta, lex: ..., prior: ..., utt_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * vec(utt_weight, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_color = 1.0 - ctx.grayscale
    prior_logits = (
        (params["w_features"] + params["w_valence"] * ctx.valence) * ctx.feature_count
        + params["w_color"] * is_color
        + params["w_familiar"] * ctx.familiarization
    )
    prior = softmax_prior(prior_logits)

    is_real = 1.0 - ctx.is_sink
    ext = jnp.sum(ctx.lex * is_real[:, None], axis=1)
    log_ext = jnp.log(jnp.maximum(1.0, ext))
    utt_weight = jnp.exp(params["gamma_acc"] * log_ext)

    heard = L1(params["beta"], ctx.lex, prior, utt_weight)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
