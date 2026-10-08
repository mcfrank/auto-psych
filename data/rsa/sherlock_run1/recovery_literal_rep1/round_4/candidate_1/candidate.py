import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](lex: ..., aspect_weights: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * at(aspect_weights, u, g) + {EPS}),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    ext = jnp.sum(ctx.lex, axis=1, keepdims=True)
    l0 = ctx.lex / (ext + EPS)
    aspect_info = l0 @ ctx.lex.T
    aspect_weights = jnp.exp(params["alpha"] * jnp.log(aspect_info + EPS))
    heard = L1(ctx.lex, aspect_weights, prior)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
