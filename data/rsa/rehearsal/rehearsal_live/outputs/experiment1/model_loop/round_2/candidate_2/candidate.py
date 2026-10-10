"""Visual crowding listener.

Listeners interpret referring expressions under limited visual attention across the display,
where candidate referents that are perceptually crowded by identical or similar neighboring
items suffer from attentional suppression. When choosing among referents consistent with
the speaker's utterance, the listener weights items by their perceptual isolation in the
visual display, favoring uncrowded referents that avoid visual crowding. On uninformative
prior trials where no word directs attention, choice is governed directly by visual display
attention, spontaneously selecting uncrowded items over crowded duplicates.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., isolation: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * exp(beta * vec(isolation, r)))
    return Pr[listener.r == r]


def compute_isolation(ctx):
    features = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    feats = jnp.vstack([features, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = pair_dist.shape[0]
    eye = jnp.eye(n_obj) * 1e5
    return jnp.min(pair_dist + eye, axis=1)


def choice_probs(params, ctx):
    isolation = compute_isolation(ctx)
    prior = softmax_prior(params["beta"] * isolation)
    heard = L1(params["alpha"], params["beta"], ctx.lex, isolation)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
