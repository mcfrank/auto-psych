"""Contrastive feature speaker.

Speakers select features that provide contrastive differentiation, preferring
descriptors that distinguish the intended referent from competitor objects sharing
its other features (minimal contrast pairs, Sedivy et al. 1999). Pragmatic listeners
invert this contrast-sensitive speaker during recursive reference resolution.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "contrast_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, contrast_weight, contrast: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + contrast_weight * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    overlap = jnp.dot(real_lex.T, real_lex)
    overlap = overlap - jnp.diag(jnp.diag(overlap))
    contrast = real_lex * jnp.matmul(1.0 - real_lex, overlap)
    heard = L1(
        params["alpha"], params["contrast_weight"], contrast, ctx.lex
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
