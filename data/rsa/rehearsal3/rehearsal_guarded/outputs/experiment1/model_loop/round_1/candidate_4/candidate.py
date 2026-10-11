"""RSA pragmatic listener at depth 2 with singleton salience, contextual distinctiveness, visual feature complexity, evaluative valence framing, visual color contrast, literal L0, and competitor-confusion aversion.

Refining rsa_l2_singleton_feat_color_valence_l0 by incorporating the competitor confusion component from competitor_confusion_speaker:
speakers actively penalize utterances that are shared with visually confusable competitors in the context.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., prior: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, lex, prior, confusion) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


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

    n_obj = total_dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-total_dist) * (1.0 - eye)
    confusion = real_lex @ sim

    heard = L2(params["alpha"], params["w_confusion"], ctx.lex, prior, confusion)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
