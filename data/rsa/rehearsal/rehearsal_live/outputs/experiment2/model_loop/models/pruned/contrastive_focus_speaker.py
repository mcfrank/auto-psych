"""Pragmatic listener reasoning about a contrastive focus speaker.

Speakers in visual reference games produce referring expressions with contrastive
focus, selecting descriptive features specifically to distinguish the intended
referent from a minimal contrast partner in the visual scene—a competitor that
shares the referent's other features but lacks the named one. Pragmatic listeners
invert this contrastive focus speaker via Bayes' rule: when multiple candidate
referents satisfy the uttered word, listeners favor the referent that forms a minimal
contrast pair with a competitor lacking the word. On uninformative trials where no
informative word is uttered, listener expectations reflect baseline communicative
contrast potential, favoring referents with distinctive visual contrast.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 2.0),
    "w_contrast_prior": dist.Normal(0.0, 2.0),
    "w_valence": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "w_base": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., contrast_focus: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_contrast * at(contrast_focus, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast_focus(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    total_dist = jnp.sum(diff, axis=0)
    dist_other = total_dist[None, :, :] - diff

    lacks_u = (real_lex == 0.0)
    masked_dist = jnp.where(lacks_u[:, None, :], dist_other, 1e5)
    min_dist = jnp.min(masked_dist, axis=-1)

    contrast_focus = jnp.where(
        (min_dist < 1e4) & (real_words[:, None] > 0),
        jnp.exp(-min_dist),
        0.0,
    ) * real_lex
    return contrast_focus


def choice_probs(params, ctx):
    contrast_focus = compute_contrast_focus(ctx)
    prior_contrast = jnp.max(contrast_focus, axis=0)

    val_bias = params["w_valence"] * ctx.valence * ctx.feature_count
    color_bias = params["w_color"] * (1.0 - ctx.grayscale)
    base_bias = params["w_base"] * ctx.familiarization

    prior = softmax_prior(
        params["w_contrast_prior"] * prior_contrast
        + val_bias
        + color_bias
        + base_bias
    )

    heard = L1(
        params["alpha"],
        params["w_contrast"],
        ctx.lex,
        contrast_focus,
        prior,
    )[ctx.utterance]

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
