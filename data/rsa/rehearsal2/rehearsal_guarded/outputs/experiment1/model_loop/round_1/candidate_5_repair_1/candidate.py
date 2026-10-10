"""RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, literal L0, and contrastive speaker utility.

Refining rsa_l2_singleton_feat_color_valence_l0 by incorporating the contrast-maximizing referring expression component from contrastive_speaker:
speakers and listeners reason at depth 2 (inverting a speaker who simulates a depth-1 pragmatic listener) with common knowledge of an object prior combining discrete visual singleton salience, continuous contextual distinctiveness, visual feature complexity modulated by evaluative valence framing, and visual color contrast.
In addition, speakers at depths 1 and 2 choose referring expressions to establish visual contrast against context competitors that lack the named feature: candidate words receive a communicative utility bonus proportional to their contrast resolution with minimally different competitors in the scene.
Pragmatic listeners invert this contrastive speaker at depth 2.
Literal listener L0 operates purely on truth-conditional semantics without perceptual bias.
On prior trials with no informative word, choices are governed directly by the combined perceptual prior.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., prior: ..., contrast: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., prior: ..., contrast: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_contrast, lex, prior, contrast) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    pair_dist = jnp.sum(jnp.abs(diff), axis=0)

    n_obj = pair_dist.shape[0]
    eye = jnp.eye(n_obj)

    lacks_u = (1.0 - ctx.lex)[:, None, :]
    sim_other = jnp.exp(1.0 - pair_dist)[None, :, :] * (1.0 - eye)[None, :, :]
    contrast = jnp.sum(lacks_u * sim_other, axis=-1) * real_lex
    return contrast


def choice_probs(params, ctx):
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

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )

    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    contrast = compute_contrast(ctx)

    heard = L2(
        params["alpha"],
        params["w_contrast"],
        ctx.lex,
        prior,
        contrast,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
