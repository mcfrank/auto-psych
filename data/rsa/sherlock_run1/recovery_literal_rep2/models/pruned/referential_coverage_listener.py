"""Referential coverage listener evaluating how sufficiently the utterance covers the referent.

When interpreting a referring expression, listeners evaluate candidate referents
based on referential coverage: the proportion of the object's total contextual
information that is captured by the uttered word. If a referent has other unmentioned
features that are highly distinctive and informative, the uttered word leaves most
of the referent's identity unexplained (low coverage). Conversely, if an object has
no other features or only ubiquitous, uninformative features, the uttered word accounts
for nearly all of its communicative identity (high coverage). Listeners prefer referents
whose communicative identity is most comprehensively covered by the speaker's word.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_coverage[u: UTT, r: OBJ](beta, lex: ..., coverage: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * vec(coverage, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    # Contextual informativeness of each feature is inverse object count
    col_sums = jnp.sum(ctx.lex, axis=1)
    is_real = 1.0 - ctx.is_sink
    info = jnp.where(is_real > 0, 1.0 / jnp.maximum(1.0, col_sums), 0.0)

    # Total contextual informativeness of each object across all its features
    total_info = jnp.sum(ctx.lex * info[:, None], axis=0)
    safe_total = jnp.maximum(1e-4, total_info)

    # Referential coverage of the heard utterance for each object
    u_info = info[ctx.utterance]
    coverage = (ctx.lex[ctx.utterance] * u_info) / safe_total

    # On prior trials, best single-word coverage of each object
    best_feat_info = jnp.max(ctx.lex * info[:, None], axis=0)
    prior_coverage = best_feat_info / safe_total
    prior = softmax_prior(params["beta"] * prior_coverage)

    heard = L_coverage(params["beta"], ctx.lex, coverage)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
