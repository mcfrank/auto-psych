"""Competitor foil differentiation listener.

Speakers formulate referring expressions by identifying a specific confusable visual competitor
foil in the context and choosing a word that differentiates the intended referent from that foil.
Speakers prioritize foils that share many visual features with the target, selecting true
descriptions that are false of the focal competitor. Pragmatic listeners invert this two-stage
communicative process, jointly inferring the intended referent and the competitor foil the speaker
sought to eliminate. On uninformative prior trials with no distinguishing word, choices follow visual
singleton salience. A lapse parameter captures random guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., foil_weight: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(c in OBJ, wpp=at(foil_weight, r, c)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * (1.0 - at(lex, u, c))),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_pair_dist(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    feat_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    return feat_dist + gray_diff + fam_diff


def compute_singleton_indicator(dist):
    duplicate_count = jnp.sum(dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_foil_weight(dist, beta):
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    return (1.0 - eye) * jnp.exp(-beta * dist)


def choice_probs(params, ctx):
    dist = compute_pair_dist(ctx)
    is_singleton = compute_singleton_indicator(dist)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    foil_weight = compute_foil_weight(dist, params["beta"])

    heard = L1(
        params["alpha"],
        ctx.lex,
        prior,
        foil_weight,
    )[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
