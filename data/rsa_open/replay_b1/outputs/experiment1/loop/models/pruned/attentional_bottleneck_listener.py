"""Attentional bottleneck listener.

Speakers with limited visual attention choose referring expressions to differentiate the target
from its nearest confusable context competitor lacking the feature, rather than exhaustively summing
contrast across all distractors. Listeners interpret utterances by inverting this attention-limited
speaker, while allocating referent expectations on uninformative displays based on perceptual
isolation from surrounding items.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_isolation": dist.Normal(0.0, 1.0),
}


@memo
def L_attention[u: UTT, r: OBJ](alpha, lex: ..., contrast: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(contrast, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diff, axis=0)
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    # Competitors lacking utterance u
    not_lex = 1.0 - real_lex
    comp_sim = not_lex[:, :, None] * sim[None, :, :]

    # Nearest-competitor contrast (attentional bottleneck onto the most confusable competitor)
    contrast = real_lex * jnp.max(comp_sim, axis=1)

    # Perceptual isolation prior (minimum distance to any other context object)
    dist_other = dist + eye * 999.0
    isolation = jnp.min(dist_other, axis=1)
    prior = softmax_prior(params["w_isolation"] * isolation)

    heard = L_attention(params["alpha"], ctx.lex, contrast, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
