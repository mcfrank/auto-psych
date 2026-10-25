"""Grammatical exhaustification listener with contextual distinctiveness.

Refining grammatical_exhaustification by incorporating continuous contextual
distinctiveness into the visual attention prior. Listeners interpret referring
expressions using grammatical exhaustification over unasserted alternative
descriptions, ruling out referents proportionally to the informativeness of any
more specific unmentioned features. Crucially, the visual attention prior
combines discrete singleton salience (favoring items that lack identical clones)
with continuous contextual distinctiveness (favoring items with greater total
feature contrast across the visual display).
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


@memo
def L_exh[u: UTT, r: OBJ](gamma, lex: ..., prior: ..., exh_penalty: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r)
        * at(lex, u, r)
        * exp(-gamma * at(exh_penalty, u, r)),
    )
    return Pr[listener.r == r]


def compute_singleton_and_distinct(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))
    return is_singleton, distinct


def compute_exh_penalty(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    ext = jnp.maximum(jnp.sum(real_lex, axis=1), 1.0)
    info = jnp.log(n_obj / ext) * (1.0 - ctx.is_sink)
    total_object_info = jnp.sum(real_lex * info[:, None], axis=0)
    penalty = jnp.maximum(
        0.0, total_object_info[None, :] - real_lex * info[:, None]
    )
    return penalty


def listener(params, ctx):
    is_singleton, distinct = compute_singleton_and_distinct(ctx)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )
    exh_penalty = compute_exh_penalty(ctx)
    heard = L_exh(params["gamma"], ctx.lex, prior, exh_penalty)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
