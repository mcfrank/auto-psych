"""Distractor divergence heuristic listener model.

When interpreting referring expressions, listeners rely on a non-Bayesian
distractor divergence penalty heuristic rather than recursive Theory-of-Mind
mental simulation: after filtering candidate objects to those literally
matching the uttered word, the listener penalizes referents in proportion
to their total visual feature distance from the non-matching distractors in
the scene, favoring referents that exhibit minimal feature divergence from
the ruled-out distractors. On uninformative trials with no descriptive word,
choice is uniform. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "lambda_divergence": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_divergence[u: UTT, r: OBJ](lambda_divergence, lex: ..., foil_dist: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-lambda_divergence * at(foil_dist, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist_mat = jnp.sum(diff, axis=0)

    is_distractor = 1.0 - ctx.lex
    foil_dist = jnp.matmul(is_distractor, dist_mat)

    heard = L_divergence(
        params["lambda_divergence"], ctx.lex, foil_dist
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
