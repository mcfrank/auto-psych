"""Contrastive inference listener.

Listeners interpret referring expressions by assuming that speakers produce
descriptive features specifically to resolve contextual contrast pairs—identifying
a target that contrasts with an otherwise similar competitor lacking that feature.
When hearing an expression, candidate referents are weighted by their minimal
feature distance to an object lacking the asserted feature (minimal contrast
partner), favoring referents whose extraneous features align with a contextual
contrast rather than isolated exemplars. On uninformative prior trials without
an informative word, choices default to uniform baseline expectations.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](beta, lex: ..., contrast_support: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * at(contrast_support, u, r)),
    )
    return Pr[listener.r == r]


def compute_contrast_support(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    n_utt = ctx.lex.shape[0]
    eye_utt = jnp.eye(n_utt)[:, :, None, None]
    bg_dist = jnp.sum(diff[None, :, :, :] * (1.0 - eye_utt), axis=1)
    not_u = (ctx.lex == 0.0)[:, None, :]
    safe_bg_dist = jnp.where(not_u, bg_dist, 1e5)
    min_bg_dist = jnp.min(safe_bg_dist, axis=-1)
    contrast_support = jnp.where(min_bg_dist < 1e4, jnp.exp(-min_bg_dist), 0.0)
    return contrast_support


def choice_probs(params, ctx):
    contrast_support = compute_contrast_support(ctx)
    heard = L_contrast(params["beta"], ctx.lex, contrast_support)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
