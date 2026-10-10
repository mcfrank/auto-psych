"""Attenuated solitary perceptual oddity listener with mutual exclusivity.

Refining solitary_oddity_exclusivity_listener by decoupling ungrounded prior solitary
oddity pop-out from linguistic utterance comprehension solitary oddity bias. Listeners
interpret referring expressions using a solitary perceptual oddity heuristic combined
with the mutual exclusivity principle. On ungrounded prior trials without an informative
word, listeners spontaneously target the solitary singleton with strong bottom-up visual
pop-out (beta_prior). When an informative referring expression is heard, top-down semantic
constraints focus selective attention on truth-conditional applicability, which competes
with and attenuates bottom-up perceptual oddity pop-out (beta_utt) among matching referents;
simultaneously, listeners resolve ambiguity among distinct referents through the mutual
exclusivity principle, penalizing any candidate referent possessing an exclusive contextually
unique feature that the speaker could have used instead (beta_me). When multiple singletons
or no duplicates exist, solitary oddity pop-out is absent. A lapse parameter captures
random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "beta_me": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](
    beta_utt, beta_me, lex: ..., is_singleton: ..., exclusive_count: ...
):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r)
        * exp(
            beta_utt * vec(is_singleton, r)
            - beta_me * vec(exclusive_count, r)
        ),
    )
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    feats = jnp.vstack([
        ctx.lex,
        ctx.grayscale[None, :],
        ctx.familiarization[None, :],
    ])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_item_singleton = (copy_count <= 1.0).astype(jnp.float32)
    singleton_count = jnp.sum(is_item_singleton)
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0)
        & (singleton_count > 0.5)
        & (singleton_count < 1.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton


def compute_exclusive_count(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    is_exclusive_feat = (ext == 1.0).astype(jnp.float32) * real_words
    return jnp.sum(real_lex * is_exclusive_feat, axis=0)


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    exclusive_count = compute_exclusive_count(ctx)
    prior = softmax_prior(params["beta_prior"] * is_singleton)
    heard = L_heur(
        params["beta_utt"],
        params["beta_me"],
        ctx.lex,
        is_singleton,
        exclusive_count,
    )[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
