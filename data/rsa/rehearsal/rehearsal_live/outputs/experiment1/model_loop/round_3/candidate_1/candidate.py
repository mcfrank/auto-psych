"""Evaluative attribute listener.

Listeners interpret referring expressions by reasoning about a communicative
speaker who selects referents according to their evaluative desirability,
treating chromatic coloration as an intrinsic visual attribute alongside
semantic features. Evaluative framing shifts the speaker's referential goal along
this unified attribute dimension: positive framing ("favorite") favors attribute-rich
referents, negative framing ("least favorite") inverts the preference to favor
sparse, featureless referents, and neutral framing retains a baseline preference
for visually rich referents. During comprehension, listeners invert this speaker,
naturally integrating communicative informativeness with evaluative desirability
across both framed and neutral contexts.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_base": dist.Normal(0.0, 1.0),
    "w_val": dist.Normal(0.0, 1.0),
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
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    attr = ctx.feature_count + (1.0 - ctx.grayscale)
    w_eff = params["w_base"] + ctx.valence * params["w_val"]
    prior = softmax_prior(w_eff * attr)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
