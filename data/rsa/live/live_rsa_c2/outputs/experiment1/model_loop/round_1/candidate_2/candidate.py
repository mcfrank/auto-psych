"""RSA pragmatic listener at depth 2 with redundancy-discounted alternative utterances.

Listeners model a speaker who discounts the base availability of alternative utterances
according to their contextual feature overlap (Jaccard similarity across display objects).
When candidate referents have correlated or redundant features, those features share
communicative mass rather than independently multiplying the Gricean preemption penalty
against shared utterances.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_redundancy(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute contextual redundancy for each utterance via pairwise Jaccard overlap."""
    real_lex = lex * (1.0 - is_sink)[:, None]
    intersection = jnp.dot(real_lex, real_lex.T)
    counts = jnp.sum(real_lex, axis=1)
    union = counts[:, None] + counts[None, :] - intersection
    jaccard = jnp.where(union > 0, intersection / (union + 1e-10), 0.0)
    # Self-overlap is 1 for all utterances
    jaccard = jnp.where(jnp.eye(lex.shape[0]) > 0, 1.0, jaccard)
    # Sink utterance is orthogonal to all real utterances
    sink_mask = (
        is_sink[:, None] * (1.0 - is_sink[None, :])
        + (1.0 - is_sink[:, None]) * is_sink[None, :]
    )
    jaccard = jnp.where(sink_mask > 0, 0.0, jaccard)
    return jnp.sum(jaccard, axis=1)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., alt_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(alt_weight, u)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., alt_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * vec(alt_weight, u)
            * exp(alpha * log(L1[u, r](alpha, lex, alt_weight) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    redundancy = compute_redundancy(ctx.lex, ctx.is_sink)
    alt_weight = jnp.exp(-params["beta"] * jnp.log(jnp.maximum(redundancy, 1.0)))

    heard = L2(params["alpha"], ctx.lex, alt_weight)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
