"""Pragmatic listener reasoning about an aspect-oriented speaker goal.

Rather than assuming communication is strictly about identifying a referent's complete
identity, the listener models a speaker whose goal is to convey a specific visual
aspect (feature) of the target object. Given a referent, the speaker selects an aspect
goal and chooses an utterance that communicates that aspect to the listener.
Pragmatic listeners perform joint inference over the speaker's latent communicative
goal and intended referent through recursive pragmatic reasoning, marginalizing over
the latent aspect goal. On uninformative prior trials, listeners guess uniformly.
A lapse parameter accounts for random choices.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., aspect_prior: ..., aspect_info: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(aspect_prior, g, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(at(aspect_info, u, g) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., aspect_prior: ..., aspect_info: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(g in UTT, wpp=at(aspect_prior, g, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(at(aspect_info, u, g) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_aspect_matrix(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute the normalized aspect distribution P(aspect g | referent r)."""
    features = lex * (1.0 - is_sink)[:, None]
    fc = jnp.sum(features, axis=0)
    is_plain = (fc == 0).astype(jnp.float32)
    safe_fc = jnp.maximum(fc, 1.0)
    T_features = features / safe_fc[None, :]
    T_sink = is_sink[:, None] * is_plain[None, :]
    return T_features + T_sink


def compute_l0(lex: jnp.ndarray) -> jnp.ndarray:
    """Literal listener probabilities P_L0(r | u)."""
    count = jnp.sum(lex, axis=1, keepdims=True)
    return lex / jnp.maximum(count, 1.0)


def choice_probs(params, ctx):
    T = compute_aspect_matrix(ctx.lex, ctx.is_sink)
    l0 = compute_l0(ctx.lex)
    aspect_info_0 = jnp.maximum(jnp.matmul(l0, T.T), 0.01)
    l1 = L1(params["alpha"], ctx.lex, T, aspect_info_0)
    aspect_info_1 = jnp.maximum(jnp.matmul(l1, T.T), 0.01)
    heard = L2(params["alpha"], ctx.lex, T, aspect_info_1)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
