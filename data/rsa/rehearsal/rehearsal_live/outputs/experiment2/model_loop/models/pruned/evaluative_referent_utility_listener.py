"""Pragmatic listener reasoning about an evaluative referent utility speaker.

Listeners interpret referring expressions by modeling a communicative speaker
whose prior choice of an intended referent is driven by the object's subjective
visual endowment (its descriptive features and chromatic coloration). The speaker
evaluates this endowment according to their stated evaluative attitude—seeking
maximal endowment under positive framing ('favorite'), minimal endowment under
negative framing ('least favorite'), and retaining a baseline default preference
for endowed referents under neutral framing. Pragmatic listeners invert this
evaluative speaker via Bayes' rule, integrating the referent utility prior with
communicative informativeness.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta_val": dist.Normal(0.0, 2.0),
    "beta_0": dist.Normal(0.0, 1.0),
    "beta_color": dist.Normal(0.0, 2.0),
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
    eval_weight = params["beta_val"] * ctx.valence + params["beta_0"]
    color_bias = params["beta_color"] * (1.0 - ctx.grayscale)
    prior = softmax_prior(eval_weight * ctx.feature_count + color_bias)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
