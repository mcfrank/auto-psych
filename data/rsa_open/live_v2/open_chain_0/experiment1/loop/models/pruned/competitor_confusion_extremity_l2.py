"""Pragmatic listener at depth 2 reasoning about a competitor-confusion-averse speaker under a shared confusion and extremity prior.

Refines competitor_confusion_l2 by incorporating the relative feature-complexity extremity
term from extension_confusion_cost_l2 into the shared object prior
(`softmax_prior(-w_confusion * baseline_confusion + w_extremity * extremity)`),
adding one parameter (`w_extremity`) while keeping recursion depth at 2 (`L2` -> `L1` -> `L0`),
`alpha`, `w_confusion`, and all other terms unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
    "w_extremity": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., confusion: ..., prior: ...):
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
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., confusion: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, lex, confusion, prior) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_confusion_and_prior(ctx, w_confusion, w_extremity):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    confusion = real_lex @ sim
    baseline_confusion = jnp.sum(sim, axis=1)

    fc = ctx.feature_count
    mean_fc = jnp.mean(fc)
    extremity = (fc - mean_fc) * (fc - mean_fc) / (fc + 1.0)
    prior = softmax_prior(-w_confusion * baseline_confusion + w_extremity * extremity)
    return confusion, prior


def listener(params, ctx):
    confusion, prior = compute_confusion_and_prior(
        ctx, params["w_confusion"], params["w_extremity"]
    )
    heard = L2(
        params["alpha"], params["w_confusion"], ctx.lex, confusion, prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
