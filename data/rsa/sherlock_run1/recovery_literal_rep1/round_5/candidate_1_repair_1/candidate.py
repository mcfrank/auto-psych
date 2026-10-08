import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def LexicalContrastListener[u: UTT, r: OBJ](lex: ..., weights: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * at(weights, u, r))
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    ext = jnp.sum(ctx.lex * (1.0 - ctx.is_sink[:, None]), axis=1)
    spec = (1.0 - ctx.is_sink) / (ext + EPS)
    total_spec = jnp.sum(ctx.lex * spec[:, None], axis=0)
    alt_spec = jnp.maximum(0.0, total_spec[None, :] - ctx.lex * spec[:, None])
    weights = jnp.exp(
        -params["beta"] * alt_spec + params["w_familiar"] * ctx.familiarization[None, :]
    )
    heard = LexicalContrastListener(ctx.lex, weights)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
