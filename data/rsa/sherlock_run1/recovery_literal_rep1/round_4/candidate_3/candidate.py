import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def S0[u: UTT, r: OBJ](lex: ...):
    speaker: knows(r)
    speaker: chooses(u in UTT, wpp=at(lex, u, r))
    return Pr[speaker.u == u]


@memo
def ResonantListener[u: UTT, r: OBJ](beta, lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r)
        * exp(
            beta
            * (
                log(L0[u, r](lex) + {EPS})
                + log(S0[u, r](lex) + {EPS})
            )
        ),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = ResonantListener(params["beta"], ctx.lex, prior)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
