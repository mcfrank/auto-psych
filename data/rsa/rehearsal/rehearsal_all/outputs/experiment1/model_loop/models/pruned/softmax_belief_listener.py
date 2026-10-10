"""Softmax decision rule over pragmatic and prior beliefs.

Listeners in reference games infer beliefs about candidate referents through
probabilistic pragmatic reasoning and perceptual prior expectations, but convert
those continuous beliefs into a discrete referent click via a softmax decision rule
with decision rationality beta. Rather than passively probability-matching their
beliefs (the standard RSA assumption of beta = 1), decisive comprehenders with
beta > 1 amplify the highest-probability referent, while beta < 1 compresses
choices toward indifference. A lapse parameter captures uniform motor noise.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    effective_w_features = params["w_features"] + ctx.valence * params["w_valence"]
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + effective_w_features * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
        + params["w_color"] * (1.0 - ctx.grayscale)
    )

    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    belief = jnp.where(ctx.is_prior > 0, prior, heard)
    decision = jax.nn.softmax(params["beta"] * jnp.log(belief + EPS))
    return with_lapse(decision, params["lapse"])
