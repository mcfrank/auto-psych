"""RSA pragmatic listener at depth 2 with singleton salience, utterance-extension distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, and literal L0.

Refines rsa_l2_singleton_feat_color_valence_l0 by evaluating contextual feature distinctiveness
on utterance trials within the extension of each candidate utterance (the objects of which the
word is literally true) rather than across false distractors ruled out by the word, while
retaining scene-wide distinctiveness on uninformative prior trials. Recursion depth remains L2
(calling L1, which calls literal L0) with the exact same six parameters.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(prior, u, r))
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(prior, u, r))
    return Pr[listener.r == r]


def listener(params, ctx):
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
    distinct = jnp.sum(pair_dist, axis=1)
    ext_distinct = real_lex @ pair_dist

    effective_w_features = (
        params["w_features"] + ctx.valence * params["w_valence"]
    )
    base_logits = (
        params["w_singleton"] * is_singleton
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    scene_prior = softmax_prior(base_logits + params["w_distinct"] * distinct)
    utt_prior = softmax_prior(
        base_logits[None, :] + params["w_distinct"] * ext_distinct
    )

    heard = L2(params["alpha"], ctx.lex, utt_prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, scene_prior, heard)
