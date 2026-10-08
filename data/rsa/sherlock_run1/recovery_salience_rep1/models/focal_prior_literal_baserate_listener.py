"""RSA pragmatic listener with prior-informed literal interpretation and focal attention.

Refines prior_literal_baserate_listener: listeners combine Bayesian literal
interpretation with focal visual attention, so that when simulating communicative
success, the speaker anticipates a listener whose candidate evaluations incorporate
experiential base rates, feature complexity priors, and focal attentional
discounting of display distractors.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience": dist.Normal(0.0, 1.0),
    "baserate": dist.LogNormal(0.0, 1.0),
    "distractor_weight": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ..., atten: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * (vec(prior, r) + {EPS}) * (vec(atten, r) + {EPS}),
    )
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., atten: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior, atten) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance])
    atten = jnp.where(is_cand > 0, 1.0, params["distractor_weight"])
    fam_logits = jnp.where(ctx.has_familiarization > 0, jnp.log(ctx.familiarization + EPS), 0.0)
    prior = softmax_prior(params["salience"] * ctx.feature_count + params["baserate"] * fam_logits)
    heard = L1(params["alpha"], ctx.lex, prior, atten)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
