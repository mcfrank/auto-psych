"""Pragmatic listener at depth 2 with evaluative stance and visual prominence.

Speakers select referents according to evaluative prominence: the alignment
between an object's sensory visual prominence (chromatic vibrancy and visual
feature elaboration) and the speaker's evaluative stance. Under positive
framing ('favorite'), speakers seek objects of high visual prominence; under
negative framing ('least favorite'), this evaluative stance inverts toward
minimal, unadorned objects; and under neutral framing, choices follow a
baseline positive orientation toward visual features and chromatic pop-out.
Listeners reason at depth 2, inverting a speaker who anticipates a pragmatic
listener. On uninformative prior trials, choices directly follow the
evaluative prominence distribution. A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_base": dist.Normal(0.0, 1.0),
    "w_val": dist.Normal(0.0, 1.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_referent_prior(ctx, w_base, w_val, w_color):
    """Compute referent prior from evaluative stance and visual prominence."""
    stance = w_base + w_val * ctx.valence
    score = stance * ctx.feature_count + w_color * (1.0 - ctx.grayscale)
    return softmax_prior(score)


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
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = compute_referent_prior(
        ctx, params["w_base"], params["w_val"], params["w_color"]
    )
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
