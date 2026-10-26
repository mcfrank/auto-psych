"""Type-contrastive listener.

Speakers choose referring expressions to maximize contextual contrast against distinct competitor
types in the immediate display, weighting each distractor's contribution inversely by its duplicate
multiplicity rather than accumulating contrast independently across identical clones. Listeners
interpret utterances by inverting this type-contrastive speaker, expecting heard words to discriminate
the target from each competing distractor category. On uninformative displays without a referring
expression, visual singleton salience guides referent expectations.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L_contrast[u: UTT, r: OBJ](alpha, lex: ..., contrast: ..., prior: ...):
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

    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    dist = jnp.sum(diff, axis=0)

    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    duplicate_count = jnp.sum(dist == 0, axis=1)
    type_sim = sim / duplicate_count[:, None]

    # Contextual contrast of utterance u for referent r against distractor types lacking u
    contrast = real_lex * ((1.0 - real_lex) @ type_sim)

    # Perceptual singleton prior on uninformative displays
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)
    prior = softmax_prior(params["w_singleton"] * is_singleton)

    heard = L_contrast(params["alpha"], ctx.lex, contrast, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
