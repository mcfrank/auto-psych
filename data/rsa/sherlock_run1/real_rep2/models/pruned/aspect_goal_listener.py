"""Aspect goal listener: pragmatic inference over latent feature-level communicative goals.

Listeners interpret referring expressions by modeling the speaker as aiming to inform
them about a specific aspect of the referent rather than its overall identity, jointly
inferring both the intended object and the speaker's communicative goal. When choosing
an utterance, the speaker selects a true feature of the object to convey and evaluates
candidate words by how effectively they communicate that target feature to a literal
listener. The pragmatic listener reasons counterfactually about which latent feature-level
goal prompted the speaker's word, favoring referents for which the observed utterance
was the most informative way to communicate an aspect.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0_feat[u: UTT, g: UTT](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r) + {EPS})
    return Pr[at(lex, g, listener.r) > 0]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(g in UTT, wpp=at(lex, g, r) + {EPS}),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0_feat[u, g](lex, prior) + {EPS})) + {EPS},
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
