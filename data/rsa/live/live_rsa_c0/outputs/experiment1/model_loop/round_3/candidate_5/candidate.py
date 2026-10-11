"""Expected success limited attention listener.

Refining limited_attention_listener by incorporating the expected communicative success component
from expected_success_speaker: speakers choose referring expressions to maximize expected communicative
success—the bounded linear likelihood of referent recovery—rather than logarithmic surprisal.
Pragmatic listeners invert this success-maximizing speaker under capacity-limited visual attention.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

MODES = jnp.arange(2)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "logit_p_base": dist.Normal(2.0, 1.0),
    "w_load": dist.Normal(0.5, 0.5),
    "w_popout": dist.Normal(1.0, 0.5),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., attention: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L0[u, r](lex)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_attention[u: UTT, r: OBJ](alpha, p_attend, lex: ..., attention: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(attention, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L1[u, r](alpha, lex, attention)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(mode in MODES, wpp=p_attend if mode == 1 else (1.0 - p_attend))
    listener: chooses(
        r in OBJ,
        wpp=Pr[speaker.r == r] if mode == 1 else vec(attention, r),
    )
    return Pr[listener.r == r]


def compute_salience(ctx, params):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    copy_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(copy_count == 1, 1.0, 0.0)
    distinct = jnp.sum(diff, axis=(0, 2))

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )
    return softmax_prior(salience_logits)


def choice_probs(params, ctx):
    attention = compute_salience(ctx, params)

    real_matches = ctx.lex[ctx.utterance] * (1.0 - ctx.is_sink[ctx.utterance])
    matching_count = jnp.sum(real_matches)
    is_popout = jnp.where(matching_count == 1.0, 1.0, 0.0)

    n_obj = ctx.lex.shape[1]
    logit_p_attend = (
        params["logit_p_base"]
        + params["w_popout"] * is_popout
        - params["w_load"] * (n_obj - 2.0)
    )
    p_attend = jax.nn.sigmoid(logit_p_attend)

    heard = L_attention(params["alpha"], p_attend, ctx.lex, attention)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, attention, heard)
    return with_lapse(probs, params["lapse"])
