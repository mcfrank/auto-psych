
"""Pragmatic listener inverting a Pointwise Mutual Information speaker.

Rather than evaluating an utterance by its literal truth or simulated recursive
listener belief, the speaker chooses words that maximize the Pointwise Mutual
Information (PMI) between the feature and the target referent. An utterance's
communicative value reflects how strongly it is associated with the target
relative to its baseline prevalence across the display context. Listeners
invert this mutual-information speaker via Bayes' rule. On uninformative
prior trials, choice is uniform guessing. A lapse parameter mixes in random choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_pmi(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Pointwise Mutual Information between each utterance and referent."""
    features = lex * (1.0 - is_sink)[:, None]
    fc = jnp.maximum(jnp.sum(features, axis=0, keepdims=True), 1.0)
    p_u_given_r = features / fc
    n_obj = lex.shape[1]
    p_u = jnp.sum(p_u_given_r, axis=1, keepdims=True) / n_obj
    pmi = jnp.log((p_u_given_r + EPS) / (p_u + EPS))
    return jnp.where(is_sink[:, None] > 0, 0.0, pmi)


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., pmi: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(pmi, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    pmi = compute_pmi(ctx.lex, ctx.is_sink)
    heard = L1(params["alpha"], ctx.lex, pmi)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
