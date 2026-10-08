"""Pragmatic listener inverting a discriminative, distractor-eliminating speaker.

Rather than simulating a literal listener's posterior probability, the speaker
evaluates each utterance by its contrastive discriminative power: the proportion
of distractor objects in the visual context that the feature rules out (Dale &
Reiter, 1995; Gatt & Krahmer, 2018). The pragmatic listener inverts this
discriminative speaker via Bayes' rule. With no informative word, choice is
uniform. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., disc: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * at(disc, u, r))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_utt, n_obj = ctx.lex.shape
    distractors_false = (n_obj - jnp.sum(ctx.lex, axis=1, keepdims=True)) - (1.0 - ctx.lex)
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    disc = distractors_false / denom
    disc = jnp.where(ctx.is_sink[:, None] > 0, 0.0, disc)
    heard = L1(params["alpha"], ctx.lex, disc)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
