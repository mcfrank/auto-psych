"""Divided attention listener: pragmatic listener under capacity-limited visual attention
divided among competing candidate referents.

Listeners interpret referring expressions under capacity-limited visual attention that is divided
among competing candidate referents in the visual scene. When an utterance is referentially ambiguous
and applies to multiple matching candidates, the attentional capacity available for counterfactual
pragmatic mentalizing is divided across the candidate set, causing pragmatic scalar exclusion to
attenuate with candidate load while choices become increasingly guided by visual salience and feature
complexity. On unambiguous pop-out displays with a single matching candidate, undivided attention
ensures full pragmatic verification, whereas multi-match displays dilute attention and soften
excessive scalar inferences.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.5, 1.0),
    "logit_gamma_base": dist.Normal(2.0, 1.0),
    "w_candidate": dist.Normal(0.5, 0.5),
    "w_load": dist.Normal(0.5, 0.5),
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
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L0[u, r](lex)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_divided[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., gamma: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * L1[u, r](alpha, lex)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r)
        * at(lex, u, r)
        * exp(vec(gamma, u) * log(Pr[speaker.r == r] + {EPS})),
    )
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

    salience_logits = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + effective_w_features * ctx.feature_count
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    prior = softmax_prior(salience_logits)

    real_matches_per_u = ctx.lex * (1.0 - ctx.is_sink[:, None])
    matching_counts = jnp.sum(real_matches_per_u, axis=1)
    candidate_loads = jnp.maximum(0.0, matching_counts - 1.0)

    n_obj = ctx.lex.shape[1]
    display_load = n_obj - 2.0

    logit_gamma = (
        params["logit_gamma_base"]
        - params["w_candidate"] * candidate_loads
        - params["w_load"] * display_load
    )
    gamma = jax.nn.sigmoid(logit_gamma)

    heard = L_divided(
        params["alpha"],
        ctx.lex,
        prior,
        gamma,
    )[ctx.utterance]

    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
