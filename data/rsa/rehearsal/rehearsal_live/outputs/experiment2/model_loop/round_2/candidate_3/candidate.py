"""Feature uncertainty listener.

Listeners interpret referring expressions under lexical feature uncertainty,
recognizing that an uttered word might not deterministically pick out a single
intended visual dimension. When an expression is heard, the listener assumes the
speaker primarily intended the conventional feature but maintains uncertainty
that the word refers to another contextual feature present in the visual scene,
weighting candidate referents by their joint possession of the named and
alternative contextual features. On uninformative prior trials where no
informative word is uttered, listener expectations reflect that speakers prefer
to communicate about referents with greater descriptive feature richness.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "theta": dist.Beta(1.0, 9.0),
    "w_prior": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    l1 = L1(params["alpha"], ctx.lex)

    real_words = 1.0 - ctx.is_sink
    n_real = jnp.sum(real_words)
    k_alt = jnp.maximum(n_real - 1.0, 1.0)
    has_multiple = (n_real > 1.5).astype(jnp.float32)
    alt_weight = has_multiple * (params["theta"] / k_alt)

    sum_real_l1 = jnp.sum(l1 * real_words[:, None], axis=0)
    u = ctx.utterance
    alt_l1 = jnp.maximum(sum_real_l1 - l1[u], 0.0)
    heard_u = (1.0 - alt_weight * k_alt) * l1[u] + alt_weight * alt_l1
    heard = jnp.where(ctx.is_sink[u] > 0, l1[u], heard_u)

    prior = softmax_prior(params["w_prior"] * ctx.feature_count)
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
