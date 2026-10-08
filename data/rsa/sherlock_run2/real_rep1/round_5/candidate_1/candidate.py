"""Contrast set listener model.

Listeners interpret referring expressions through referential contrast sets:
when hearing a descriptive word, candidate referents that form a minimal contrastive
pair with an object lacking that feature are favored, because the utterance serves
a contrastive function to distinguish that referent from its contextual counterpart.
On uninformative prior trials, choice is uniform. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "lambda_contrast": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](lambda_contrast, lex: ..., contrast_sim: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(lambda_contrast * at(contrast_sim, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist_mat = jnp.sum(diff, axis=0)

    is_contrast = 1.0 - ctx.lex
    sim = jnp.exp(-dist_mat)
    contrast_sim = jnp.matmul(is_contrast, sim)

    heard = L_contrast(
        params["lambda_contrast"], ctx.lex, contrast_sim
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
