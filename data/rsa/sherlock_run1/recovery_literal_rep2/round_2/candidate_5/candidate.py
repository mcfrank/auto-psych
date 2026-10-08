"""Pragmatic listener with contextual feature rarity and softmax decision rationality.

Refines softmax_listener_shared_prior by replacing the raw feature-count
prior with a contextual feature rarity prior: an object is salient to the
extent that its features are rare and distinctive across competitors in the
visual scene. This prior is shared common knowledge between the speaker and
literal listener, while the pragmatic listener applies an independent softmax
decision rationality parameter beta when selecting referents.

Differences from softmax_listener_shared_prior:
- Refined model: softmax_listener_shared_prior
- Recursion depth: depth 1 (choice_probs calls L1, unchanged).
- Parameters added: w_rarity ~ Normal(0.0, 1.0) governing the weight of contextual rarity.
- Parameters removed: w_features.
- Prior calculation in choice_probs: replaces raw feature count with contextual feature rarity
  (features weighted inversely by their contextual frequency across objects in the display).
- All other memo agents, distributions, and prior terms are unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_rarity": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_utt = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_referents = jnp.sum(real_utt, axis=-1)
    rarity_weight = jnp.where(ctx.is_sink > 0, 0.0, 1.0 / (n_referents + EPS))
    rarity_score = jnp.sum(real_utt * rarity_weight[:, None], axis=0)

    prior = softmax_prior(
        params["w_rarity"] * rarity_score + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
