"""Pragmatic listener reasoning about a competitor-confusion-averse speaker with singleton salience.

Refining competitor_confusion_speaker by incorporating discrete perceptual singleton
salience from oddity_heuristic_listener into the shared referent prior: speakers and
listeners not only penalize referring expressions that overlap with confusable
competitors, but also share a perceptual expectation that unique objects lacking
identical duplicates in the display are more salient and more likely referents a
priori. This shared singleton preference biases spontaneous object choices on
uninformative prior trials and guides pragmatic listener interpretation when
resolving referring expressions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
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


def compute_confusion_and_prior(ctx, w_confusion, w_singleton):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)

    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    confusion = real_lex @ sim
    baseline_confusion = jnp.sum(sim, axis=1)
    prior = softmax_prior(-w_confusion * baseline_confusion + w_singleton * is_singleton)
    return confusion, prior


def choice_probs(params, ctx):
    confusion, prior = compute_confusion_and_prior(
        ctx, params["w_confusion"], params["w_singleton"]
    )
    heard = L1(
        params["alpha"], params["w_confusion"], ctx.lex, confusion, prior
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
