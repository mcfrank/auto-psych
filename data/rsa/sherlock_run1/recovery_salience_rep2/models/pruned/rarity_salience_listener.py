"""Feature-rarity pragmatic listener.

Listeners resolve reference under a perceptual salience prior governed by
feature rarity (Shannon information content / surprisal) across the context.
Objects possessing infrequent or unique features capture bottom-up visual
attention and stand out as communicative referents.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "rarity_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, prior: ..., lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r) + {EPS}),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_obj = ctx.feature_count.shape[0]
    feat_freq = jnp.sum(real_lex, axis=1, keepdims=True)
    feat_rarity = jnp.where(feat_freq > 0, jnp.log((n_obj + EPS) / (feat_freq + EPS)), 0.0)
    obj_rarity = jnp.sum(real_lex * feat_rarity, axis=0)

    prior = softmax_prior(params["rarity_weight"] * obj_rarity)
    heard = L1(params["alpha"], prior, ctx.lex)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
