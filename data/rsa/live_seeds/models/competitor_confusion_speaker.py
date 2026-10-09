"""Pragmatic listener reasoning about a competitor-confusion-averse speaker.

Speakers in reference games actively avoid referring expressions that are shared
with visually similar or confusable distractors in the context. Candidate words
are penalized proportionally to their feature overlap with alternative referents
that also satisfy the word. Pragmatic listeners invert this confusion-averse
speaker, expecting words to target referents whose competitors are visually
distinct rather than near-identical clones.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
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


def choice_probs(params, ctx):
    confusion, prior = compute_confusion_and_prior(ctx, params["w_confusion"])
    heard = L1(params["alpha"], params["w_confusion"], ctx.lex, confusion)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
