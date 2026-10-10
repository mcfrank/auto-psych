"""RSA pragmatic listener at depth 2 with exponential perceptual decay gradient, mean competitor confusion, perceptual salience, familiarization base rates, and softmax decision rationality.

Refining rsa_l2_mean_confusion_softmax_l0 by estimating the exponential perceptual decay rate (generalization gradient) governing pairwise visual similarity between display objects:
Communicative speakers evaluate candidate referring expressions according to a psychological similarity gradient where competitor confusability decays exponentially with multidimensional perceptual distance, while listeners convert their resulting pragmatic beliefs into referent selections via a decisive softmax decision rule. Rather than assuming visual confusion decays at an arbitrary fixed rate, communicative speakers discount competitors along a learned perceptual generalization gradient, avoiding expressions shared with near-identical distractors while normalizing confusion risk by display competitor density. Pragmatic listeners invert this gradient-decay confusion-averse speaker at depth two while integrating perceptual salience, empirical base rates, and a decisive choice rule with internal decision rationality.

Differences from source (rsa_l2_mean_confusion_softmax_l0):
- Recursion depth: Unchanged; choice_probs calls depth-2 listener L2.
- Parameters added: gamma ~ LogNormal(0.0, 1.0) governing the exponential perceptual decay gradient in visual similarity.
- Parameters removed: None.
- Terms changed: Exactly one term changed in pairwise similarity computation: sim = jnp.exp(-params["gamma"] * total_dist) * (1.0 - eye) (scaling total perceptual distance by generalization gradient parameter gamma, rather than assuming a fixed unit decay rate). The prior computation, competitor confusion normalization by n_obj - 1, literal L0 semantics, S1 and S2 speaker choice utilities, depth-2 recursive reasoning, softmax decision rule with beta, and lapse process remain identical to the source.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 4.0),
    "w_familiar": dist.Normal(0.0, 2.0),
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
        + params["w_familiar"] * ctx.familiarization
    )

    n_obj = total_dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-params["gamma"] * total_dist) * (1.0 - eye)
    confusion = (real_lex @ sim) / jnp.maximum(n_obj - 1.0, 1.0)

    heard = L2(
        params["alpha"],
        params["w_confusion"],
        ctx.lex,
        prior,
        confusion,
    )[ctx.utterance]
    belief = jnp.where(ctx.is_prior > 0, prior, heard)
    decision = jax.nn.softmax(params["beta"] * jnp.log(belief + EPS))
    return with_lapse(decision, params["lapse"])
