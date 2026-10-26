"""Aspect Question Under Discussion (QUD) listener.

Speakers choose referring expressions to address a latent question under discussion
regarding a specific visual aspect of the target object, rather than seeking global
unique identification of the referent. Listeners interpret heard words by jointly
inferring the intended referent and the speaker's latent aspect goal, recovering belief
in distinctive competitors that share the highlighted feature. On uninformative displays
without a referring expression, visual singleton salience and contextual distinctiveness
guide referent expectations.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}


@memo
def L_aspect_qud[u: UTT, r: OBJ](speaker_w: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(speaker_w, u, r),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diff, axis=0)

    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    # Contextual contrast of aspect q for referent r against distractors lacking q
    contrast = real_lex * ((1.0 - real_lex) @ sim)

    # Aspect informativeness: P_L0(q | u)
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    overlap = real_lex @ real_lex.T
    aspect_prob = overlap / jnp.where(ext > 0, ext, 1.0)
    is_sink = ctx.is_sink[:, None]
    aspect_info = aspect_prob * (1.0 - is_sink) + jnp.outer(ctx.is_sink, ctx.is_sink)

    alpha = params["alpha"]
    aspect_weight = jnp.exp(alpha * jnp.log(aspect_info + EPS))

    # Speaker's two-stage generative process:
    # 1. P(q | r): select aspect q among features of r
    q_weights = ctx.lex * jnp.exp(alpha * contrast) + EPS
    p_q_given_r = q_weights / jnp.sum(q_weights, axis=0, keepdims=True)

    # 2. P(u | r, q): select utterance u to communicate aspect q
    u_weights = ctx.lex[:, None, :] * aspect_weight[:, :, None] + EPS
    p_u_given_rq = u_weights / jnp.sum(u_weights, axis=0, keepdims=True)

    # Marginalize over latent aspect goal q: P(u | r) = sum_q P(u | r, q) * P(q | r)
    speaker_w = jnp.sum(p_u_given_rq * p_q_given_r[None, :, :], axis=1)

    # Perceptual prior: discrete singleton salience and continuous contextual distinctiveness
    duplicate_count = jnp.sum(dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    distinct = jnp.sum(dist, axis=1)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
    )

    heard = L_aspect_qud(
        speaker_w,
        prior,
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
