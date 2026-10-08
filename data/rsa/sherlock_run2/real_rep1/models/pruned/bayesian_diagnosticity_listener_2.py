"""Bayesian diagnosticity listener model.

When interpreting referring expressions, listeners evaluate candidate referents
through Bayesian diagnosticity—the likelihood ratio of the evidence for the target
against the alternative scene competitors—rather than standard posterior
marginalization. The heard word provides evidence in favor of a referent to the
extent that the word would be produced for that referent compared to how likely
it would be produced across all competing objects in the context. Consequently,
referents for which the heard word is uniquely diagnostic are favored over candidates
that share the description with viable competitors. On uninformative trials
without a descriptive word, listeners choose uniformly among the objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "lambda_diag": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_diag[u: UTT, r: OBJ](lambda_diag, lex: ..., woe: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(lambda_diag * at(woe, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    fc = jnp.maximum(1.0, jnp.sum(real_lex, axis=0, keepdims=True))
    p_u_r = real_lex / fc

    sum_p = jnp.sum(p_u_r, axis=1, keepdims=True)
    comp_p = jnp.maximum(0.0, sum_p - p_u_r)

    num_matching = jnp.sum(real_lex, axis=1, keepdims=True)
    ambiguous = (num_matching > 1) & (real_lex > 0) & (comp_p > 0)

    safe_comp = jnp.where(ambiguous, comp_p, 1.0)
    safe_target = jnp.where(ambiguous, p_u_r, 1.0)
    woe = jnp.where(ambiguous, jnp.log(safe_target / safe_comp), 0.0)
    safe_woe = jnp.where(ctx.is_sink[:, None] > 0, 0.0, woe)

    heard = L_diag(params["lambda_diag"], ctx.lex, safe_woe)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
