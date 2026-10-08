"""Pragmatic listener inverting an expected perceptual error-minimizing speaker.

Speakers evaluate candidate referring expressions by minimizing expected perceptual error
(the expected Hamming feature distance between the intended target and the referent chosen by
a literal listener). An utterance that risks confusing the target with a distant, visually
discordant distractor incurs severe communicative loss, whereas an utterance whose potential
misidentifications are visually near the target incurs minimal loss. Pragmatic listeners
invert this error-minimizing speaker to resolve ambiguous referring expressions. On prior trials
without an informative word, choice is uniform guessing. A lapse parameter accounts for random choices.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_expected_error(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Expected perceptual feature distance under the literal listener distribution.

    For utterance u and target r, expected error is sum_{r'} L0(r' | u) * dist(r, r').
    """
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)  # (N_OBJ, N_OBJ)
    counts = jnp.sum(lex, axis=1, keepdims=True)
    l0 = lex / jnp.maximum(counts, 1.0)  # (N_UTT, N_OBJ)
    expected_loss = jnp.matmul(l0, pair_dist)  # (N_UTT, N_OBJ)
    return jnp.where(is_sink[:, None] > 0, 0.0, expected_loss)


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., err: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(-alpha * at(err, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    err = compute_expected_error(ctx.lex, ctx.is_sink)
    heard = L1(params["alpha"], ctx.lex, err)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
