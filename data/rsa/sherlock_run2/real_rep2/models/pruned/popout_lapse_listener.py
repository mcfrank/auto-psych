"""Pragmatic listener with a visual pop-out structured lapse.

Rather than assuming that lapses are uniformly distributed across the visual
display (the standard RSA lapse assumption), this model posits that inattentive
clicks are captured by bottom-up visual pop-out. When deliberate communicative
reasoning lapses, the listener's response is drawn toward perceptually isolated
objects that exhibit nearest-neighbor visual contrast under exponential distance
decay. On prior trials without an informative word, top-down linguistic guidance
is absent, so choices directly reflect this bottom-up visual pop-out distribution.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_popout": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def shepard_isolation(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Perceptual isolation under Shepard exponential distance decay."""
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = lex.shape[1]
    mask = 1.0 - jnp.eye(n_obj)
    sim = jnp.sum(jnp.exp(-pair_dist) * mask, axis=1)
    return -jnp.log(jnp.maximum(sim, 1e-6))


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    dist_vec = shepard_isolation(ctx.lex, ctx.is_sink)
    popout_dist = softmax_prior(params["w_popout"] * dist_vec)
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    choice = (1.0 - params["lapse"]) * heard + params["lapse"] * popout_dist
    return jnp.where(ctx.is_prior > 0, popout_dist, choice)
