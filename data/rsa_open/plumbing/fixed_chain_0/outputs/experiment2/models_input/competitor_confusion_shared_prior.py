"""Pragmatic listener at depth 1 reasoning about a competitor-confusion-averse speaker under a shared visual confusability prior.

Refines competitor_confusion_speaker with one change: the baseline visual confusability prior over
objects is passed into L1 as common knowledge between speaker and listener on heard-word trials
(speaker: given(r in OBJ, wpp=vec(prior, r)) instead of wpp=1), rather than being used only on
uninformative prior trials. Recursion depth remains depth 1 (listener calls L1, which simulates a
speaker reasoning about literal listener L0), no parameters are added or removed (keeping only
alpha and w_confusion), and all other terms remain identical.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
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


def compute_confusion_and_prior(ctx, w_confusion):
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
    prior = softmax_prior(-w_confusion * baseline_confusion)
    return confusion, prior


def listener(params, ctx):
    confusion, prior = compute_confusion_and_prior(ctx, params["w_confusion"])
    heard = L1(params["alpha"], params["w_confusion"], ctx.lex, confusion, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
