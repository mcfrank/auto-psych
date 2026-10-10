"""Diagnostic evidence listener.

Listeners interpret referring expressions by evaluating the weight of evidence each
word provides in favor of the target over the scene distractors. Rather than engaging
in nested recursive Theory of Mind, listeners invert a speaker who chooses descriptions
proportional to their diagnostic log-odds—the degree to which the named feature singles
out the intended referent rather than alternative objects in the context. On
uninformative trials without a distinguishing word, choices default to visual singleton
salience. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lambda_smooth": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[
    u: UTT,
    r: OBJ,
](
    alpha,
    lex: ...,
    prior: ...,
    diag: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=(at(lex, u, r) + {EPS}) * exp(alpha * at(diag, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_diagnosticity(ctx, lambda_smooth):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    n_obj = ctx.lex.shape[1]

    # k[u, r] is the count of other objects satisfying word u (excluding object r)
    n_matching = jnp.sum(real_lex, axis=1, keepdims=True)
    k = jnp.maximum(n_matching - real_lex, 0.0)
    ruled_out = (n_obj - 1.0) - k

    # Diagnostic log-odds with Bayesian smoothing
    odds = (ruled_out + lambda_smooth) / (k + lambda_smooth)
    log_odds = jnp.log(odds)

    # For the sink utterance or false features, keep diag at 0
    diag = log_odds * real_words[:, None] * real_lex
    return diag


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    diag = compute_diagnosticity(ctx, params["lambda_smooth"])

    heard = L1(
        params["alpha"],
        ctx.lex,
        prior,
        diag,
    )[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
