"""Pragmatic listener reasoning about a speaker sensitive to competitor collision risk.

Speakers incur a communicative penalty when producing an utterance that creates collision
with competitor objects in the visual scene. The collision penalty scales with how dependent
the competitor objects are on that specific word: competitors with fewer alternative features
rely more heavily on the shared word, increasing the communicative risk of using it.
Pragmatic listeners invert this collision-sensitive speaker via Bayes' rule. On uninformative
prior trials, listeners guess uniformly among candidate referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_collision": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost_collision, lex: ..., risk: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - cost_collision * at(risk, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    feat_counts = jnp.sum(real_lex, axis=0, keepdims=True)
    safe_counts = jnp.where(feat_counts > 0, feat_counts, 1.0)
    competitor_claim = real_lex / safe_counts
    total_claim = jnp.sum(competitor_claim, axis=1, keepdims=True)
    risk = jnp.maximum(0.0, total_claim - competitor_claim) * (real_lex > 0)

    heard = L1(
        params["alpha"], params["cost_collision"], ctx.lex, risk
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
