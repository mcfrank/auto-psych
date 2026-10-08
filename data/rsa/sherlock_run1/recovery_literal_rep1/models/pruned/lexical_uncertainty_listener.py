import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "sigma": dist.Beta(1.0, 19.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def L1[u: UTT, r: OBJ](alpha, soft_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(soft_lex, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(alpha * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    real_mask = (1.0 - ctx.is_sink)[:, None]
    # Interference: objects with more features have higher baseline semantic noise
    noise = params["sigma"] * real_mask * ctx.feature_count[None, :]
    soft_lex = ctx.lex + noise
    heard = L1(params["alpha"], soft_lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
