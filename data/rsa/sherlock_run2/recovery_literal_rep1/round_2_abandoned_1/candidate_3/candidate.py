"""Co-occurrence feature uncertainty listener: listener unsure which feature a word picks out."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "theta": dist.Beta(1.0, 5.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_unc[u: UTT, r: OBJ](alignment: ..., lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(f in UTT, wpp=at(alignment, u, f))
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, f, r))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    is_real = 1.0 - ctx.is_sink
    real_lex = ctx.lex * is_real[:, None]
    co_occur = jnp.dot(real_lex, real_lex.T)

    n_utt = ctx.lex.shape[0]
    eye = jnp.eye(n_utt)
    theta = params["theta"]

    diag_weight = eye * (1.0 - theta)
    offdiag_weight = (1.0 - eye) * co_occur * theta
    raw_align = diag_weight + offdiag_weight
    row_sums = jnp.maximum(1e-6, jnp.sum(raw_align, axis=1, keepdims=True))
    norm_align = raw_align / row_sums

    alignment = is_real[:, None] * norm_align + ctx.is_sink[:, None] * eye

    heard = L_unc(alignment, ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
