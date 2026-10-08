"""RSA pragmatic listener at depth 1 with decision rationality, feature salience, and base rates.

Refinement of decision_salience_listener: listeners apply softmax decision rationality
when selecting referents, while integrating empirical familiarization base rates into
their referent prior alongside visual feature complexity.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "salience": dist.Normal(0.0, 1.0),
    "baserate": dist.LogNormal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=exp(beta * log(Pr[speaker.r == r] + {EPS})))
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    fam_logits = jnp.where(ctx.has_familiarization > 0, jnp.log(ctx.familiarization + EPS), 0.0)
    prior = softmax_prior(params["salience"] * ctx.feature_count + params["baserate"] * fam_logits)
    heard = L1(params["alpha"], params["beta"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
