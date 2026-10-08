"""Ambiguity-modulated specificity listener: referent choice driven by ambiguity-gated graded truth values."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "lambda_spec": dist.Beta(1.0, 5.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](truth: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(truth, u, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["w_familiar"] * ctx.familiarization)

    is_real = 1.0 - ctx.is_sink
    real_lex = ctx.lex * is_real[:, None]
    ext_size = jnp.sum(real_lex, axis=1)

    is_unique = (ext_size == 1.0)[:, None] * real_lex
    unmentioned_unique = jnp.sum(is_unique, axis=0)[None, :] - is_unique

    counts = jnp.maximum(1.0, ctx.feature_count)
    lam = params["lambda_spec"]

    # Ambiguity of utterance u: excess extension beyond 1
    ambig = jnp.maximum(0.0, ext_size[:, None] - 1.0)
    effective_penalty = ambig * (counts[None, :] - 1.0 + unmentioned_unique)

    truth_scale = (1.0 - lam) + lam / (1.0 + effective_penalty)
    truth = is_real[:, None] * (ctx.lex * truth_scale) + ctx.is_sink[:, None] * ctx.lex

    heard = L0(truth, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
