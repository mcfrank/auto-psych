"""Pragmatic listener with similarity-gated visual attention over display distractors.

Hearing a referring expression directs the listener's attentional spotlight to
candidate referents that match the uttered word. Distractor objects that lack
the uttered word enter the listener's communicative model with attenuated
attention that decays exponentially with their perceptual Hamming distance to
the nearest candidate. Distractors sharing visual features with candidates
demand communicative differentiation, while visually distant distractors are
preattentively filtered out of the display. On prior trials without an informative
word, visual attention is uniformly distributed. Reasoning operates at depth 2.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "att_decay": dist.Exponential(1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., att: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(att, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, att) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., att: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, att) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    n_obj = ctx.lex.shape[1]
    is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance]) > 0
    cand_mask = jnp.where(is_cand, 1.0, 0.0)

    # Feature matrix excluding the unobserved sink utterance
    real_features = ctx.lex * (1.0 - ctx.is_sink[:, None])

    # Pairwise Hamming distance between context objects
    diff = jnp.abs(real_features[:, :, None] - real_features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)

    # Minimum perceptual distance from each object to any candidate referent
    dist_to_cand = jnp.min(pair_dist + (1.0 - cand_mask[None, :]) * 1e5, axis=1)

    # Attentional weight decays exponentially with perceptual distance from candidates
    att_raw = jnp.exp(-params["att_decay"] * dist_to_cand)
    att = jnp.where(ctx.is_prior > 0, 1.0, jnp.where(is_cand, 1.0, att_raw))

    heard = L2(params["alpha"], ctx.lex, att)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / n_obj)
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
