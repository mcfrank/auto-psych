import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_referable": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](beta, lex: ..., prior: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * exp(beta * Pr[speaker.r == r]))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_words = 1.0 - ctx.is_sink
    ext = jnp.sum(ctx.lex * real_words[:, None], axis=1)
    spec = jnp.where(real_words > 0, 1.0 / (ext + EPS), 0.0)
    word_spec = ctx.lex * real_words[:, None] * spec[:, None]
    referability = jnp.max(word_spec, axis=0)

    prior = softmax_prior(
        params["w_referable"] * referability + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["beta"], ctx.lex, prior)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * prior
    return jnp.where(ctx.is_prior > 0, prior, choice)
