"""Graded truth listener.

Listeners interpret referring expressions using graded semantic truth values
rather than binary all-or-nothing semantics. When an uttered word applies to
multiple candidate referents, its semantic applicability degrades with the
presence of additional extraneous features on the referent, making simpler
referents with fewer extraneous features better, more prototypical exemplars.
A communicative speaker balances this graded truth against listener recovery,
while listeners prioritize referents by feature complexity when no informative
word is uttered. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
    "w_prior": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_graded_lex(
    lex: jnp.ndarray,
    feature_count: jnp.ndarray,
    is_sink: jnp.ndarray,
    gamma: jnp.ndarray,
) -> jnp.ndarray:
    """Degrade semantic truth value of real features with additional extraneous features."""
    fc = jnp.maximum(feature_count, 1.0)
    precision = jnp.exp(-gamma * (fc - 1.0))
    real_truth = lex * precision[None, :]
    return jnp.where(is_sink[:, None] > 0, lex, real_truth)


def choice_probs(params, ctx):
    graded_lex = compute_graded_lex(
        ctx.lex, ctx.feature_count, ctx.is_sink, params["gamma"]
    )
    prior = softmax_prior(params["w_prior"] * ctx.feature_count)
    heard = L1(params["alpha"], graded_lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
