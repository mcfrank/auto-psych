"""Visually grounded pragmatic listener.

Listeners interpret referring expressions by modeling a communicative speaker who
reasons about a perceptually grounded literal listener, rather than an abstract
semantics-only agent. Bottom-up visual distinctiveness biases literal attentional
selection toward prominent referents; communicative speakers anticipate this
perceptual bias when selecting referring expressions; and pragmatic listeners
invert this visually grounded speaker. On uninformative prior trials without an
informative word, listener choices track continuous visual distinctiveness directly.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_sal": dist.Normal(0.0, 1.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](w_sal, lex: ..., salience: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(w_sal * vec(salience, r)),
    )
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_sal, lex: ..., salience: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](w_sal, lex, salience) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_salience(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = pair_dist.shape[0]
    n_comp = jnp.maximum(n_obj - 1.0, 1.0)
    return jnp.sum(pair_dist, axis=1) / n_comp


def choice_probs(params, ctx):
    salience = compute_salience(ctx)
    prior = softmax_prior(params["w_prior"] * salience)
    heard = L1(params["alpha"], params["w_sal"], ctx.lex, salience)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
