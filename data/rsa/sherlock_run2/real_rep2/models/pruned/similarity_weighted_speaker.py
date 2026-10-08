"""Pragmatic listener inverting a similarity-weighted contrastive speaker.

Rather than treating all distractors as equally threatening, the speaker evaluates
candidate utterances by their power to eliminate distractors in proportion to each
distractor's perceptual similarity to the intended target. Under the context model
of categorization, similarity between items decays with feature distance as s^d.
Ruling out highly similar, confusable distractors provides greater communicative
value than ruling out distractors that already differ across multiple dimensions.
Pragmatic listeners invert this similarity-weighted contrastive speaker via Bayes'
rule. On uninformative prior trials without an informative word, choice is uniform
guessing. A lapse parameter mixes in random choice.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "similarity": dist.Beta(2.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_contrast_util(lex: jnp.ndarray, is_sink: jnp.ndarray, similarity) -> jnp.ndarray:
    """Communicative utility of each utterance for each target object.

    Utility is the fraction of distractor similarity ruled out by the utterance,
    where each distractor is weighted by its perceptual similarity (s^d) to the
    target object.
    """
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    dist = jnp.sum(diff, axis=0)  # (N_OBJ, N_OBJ) pairwise Hamming distance
    sim = similarity ** dist  # (N_OBJ, N_OBJ) similarity

    n_utt, n_obj = lex.shape
    mask = 1.0 - jnp.eye(n_obj)  # mask out self-target
    ruled_out = 1.0 - lex  # (N_UTT, N_OBJ): 1 where utterance u is false of object r'

    # For utterance u and target r, sum over distractors r': sim[r, r'] * ruled_out[u, r']
    weighted_ruled_out = jnp.sum(
        (ruled_out[:, None, :] * mask[None, :, :]) * sim[None, :, :], axis=-1
    )
    denom = jnp.sum(mask[None, :, :] * sim[None, :, :], axis=-1)
    util = weighted_ruled_out / jnp.maximum(denom, EPS)
    return jnp.where(is_sink[:, None] > 0, 0.0, util)


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., util: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(util, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    util = compute_contrast_util(ctx.lex, ctx.is_sink, params["similarity"])
    heard = L1(params["alpha"], ctx.lex, util)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
