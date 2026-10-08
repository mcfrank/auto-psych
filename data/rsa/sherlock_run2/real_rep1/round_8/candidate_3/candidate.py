"""Perceptual confusability speaker model.

When choosing an utterance, speakers actively anticipate perceptual confusability:
rather than assuming all competitor referents that share a word are equally distracting,
speakers penalize descriptions in proportion to their visual feature similarity with
the matching competitors in the scene. Pragmatic listeners invert this confusability-sensitive
speaker to resolve referential ambiguity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_confusability": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost_confusability, lex: ..., conf_risk: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - cost_confusability * at(conf_risk, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist_mat = jnp.sum(diff, axis=0)

    sim = jnp.exp(-dist_mat)
    eye = jnp.eye(dist_mat.shape[0])
    competitor_sim = sim * (1.0 - eye)
    conf_risk = jnp.matmul(real_lex, competitor_sim.T) * (real_lex > 0)

    heard = L1(
        params["alpha"], params["cost_confusability"], ctx.lex, conf_risk
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
