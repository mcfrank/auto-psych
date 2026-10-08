import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "goal_weight": dist.HalfNormal(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_qud[u: UTT, r: OBJ](alpha, aspect_prior: ..., aspect_info: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            g in UTT,
            u in UTT,
            wpp=at(aspect_prior, g, r) * at(lex, u, r) * exp(alpha * log(at(aspect_info, u, g) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    count = jnp.sum(ctx.lex, axis=1)
    distinctiveness = (1.0 / count) ** params["goal_weight"]
    aspect_prior = ctx.lex * distinctiveness[:, None]

    l0_mat = ctx.lex / jnp.sum(ctx.lex, axis=1, keepdims=True)
    aspect_info = jnp.matmul(l0_mat, ctx.lex.T)

    heard = L_qud(params["alpha"], aspect_prior, aspect_info, ctx.lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
