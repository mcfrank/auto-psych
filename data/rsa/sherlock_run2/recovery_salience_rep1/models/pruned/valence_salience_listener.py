"""Valence-salience listener: evaluative preference in pragmatic reference.

The listener models a speaker whose choice of referent reflects their evaluative
attitude (valence): under positive framing ("favorite"), the speaker favors
feature-rich objects, whereas under negative framing ("least favorite"), the
speaker disfavors them. Under neutral framing, the speaker's prior is uniform.
The listener inverts this evaluative speaker alongside literal semantics. On
prior trials (no word spoken), choice is driven by the evaluative prior alone.
A lapse parameter mixes in uniform guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "valence_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(params["valence_weight"] * ctx.valence * ctx.feature_count)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    probs = jnp.where(ctx.is_prior > 0, prior, heard)
    return with_lapse(probs, params["lapse"])
