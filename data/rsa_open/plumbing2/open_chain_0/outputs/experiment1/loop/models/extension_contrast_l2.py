"""Extension-relative figure-ground contrast and crowding RSA at depth 2."""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., ext_salience: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * at(ext_salience, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., ext_salience: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, ext_salience) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(ext_salience, u, r))
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., ext_salience: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, ext_salience) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def listener(params, ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    pair_dist = jnp.sum(jnp.abs(feats[:, :, None] - feats[:, None, :]), axis=0)
    pair_dist_norm = pair_dist / jnp.maximum(jnp.mean(pair_dist), 1e-5)

    is_clone = (pair_dist < 1e-5).astype(jnp.float32)
    near_sim = jnp.exp(-pair_dist) * (1.0 - is_clone)
    contrast_mat = pair_dist_norm - near_sim

    # Within-extension figure-ground contrast for each utterance u:
    ext_sing = jnp.where((ctx.lex @ is_clone) <= 1.0, 1.0, 0.0)
    ext_dist = ctx.lex @ contrast_mat
    ext_salience = jax.nn.softmax(
        params["w_singleton"] * ext_sing + params["w_contrast"] * ext_dist,
        axis=-1,
    )

    # Full-scene figure-ground contrast when no word is heard (and at L2):
    scene_sing = jnp.where(jnp.sum(is_clone, axis=1) <= 1.0, 1.0, 0.0)
    scene_dist = jnp.sum(contrast_mat, axis=1)
    prior = softmax_prior(
        params["w_singleton"] * scene_sing + params["w_contrast"] * scene_dist
    )

    heard = L2(params["alpha"], ctx.lex, ext_salience, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
