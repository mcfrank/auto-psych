import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import at, vec, EPS, softmax_prior, with_lapse

BOOL = (0, 1)

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_silence": dist.Normal(0.0, 1.0),
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
def L1[u: UTT, r: OBJ](alpha, p_speak: ..., lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            speaks in BOOL,
            wpp=(1.0 - vec(p_speak, r)) if speaks == 0 else vec(p_speak, r),
        ),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes_that [speaker.speaks == 1]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    prior = softmax_prior(
        params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    l0 = L0(ctx.lex)
    real_weights = real_lex * jnp.exp(params["alpha"] * jnp.log(l0 + EPS))
    mass = jnp.sum(real_weights, axis=0)
    raw_p = mass / (jnp.exp(params["c_silence"]) + mass + EPS)
    p_speak = jnp.clip(raw_p, 1e-4, 1.0 - 1e-4)

    heard = L1(params["alpha"], p_speak, ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
