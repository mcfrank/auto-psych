"""RSA pragmatic listener at depth 2 with attentional crowding, singleton isolation, chromatic prominence, and discriminative alternative cost.

Refines crowding_singleton_chromatic_l2 by incorporating a communicative penalty
on ambiguous words when distinguishing alternatives exist (taking the discriminative
alternative cost component from discriminative_alternative_speaker) into the simulated
speakers' utility. Speakers incur a production cost when choosing an ambiguous word for
their intended referent if a uniquely distinguishing word was available in the visual
context. Pragmatic listeners invert this cost-sensitive speaker across recursive reasoning
levels, recognizing that using an ambiguous word signals that the speaker lacked an
informative distinguishing alternative, combining Gricean quantity reasoning with shared
visual crowding avoidance, singleton isolation, and chromatic prominence.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 2.0),
    "w_crowd": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., prior: ..., cost_mat: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex, prior) + {EPS})
                - cost * at(cost_mat, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, cost, lex: ..., prior: ..., cost_mat: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, cost, lex, prior, cost_mat) + {EPS})
                - cost * at(cost_mat, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_singleton, w_crowd, w_color):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)

    n_obj = dist.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist) * (1.0 - eye)
    crowding = jnp.sum(sim, axis=1)

    color_prominence = 1.0 - ctx.grayscale

    return softmax_prior(
        w_singleton * is_singleton - w_crowd * crowding + w_color * color_prominence
    )


def compute_cost_mat(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    extension = jnp.sum(real_lex, axis=1)
    is_unique = jnp.where(extension == 1.0, 1.0, 0.0)
    has_unique = jnp.where(jnp.sum(real_lex * is_unique[:, None], axis=0) > 0.0, 1.0, 0.0)
    is_ambiguous = jnp.where(extension > 1.0, 1.0, 0.0)
    return is_ambiguous[:, None] * has_unique[None, :]


def choice_probs(params, ctx):
    prior = compute_prior(
        ctx,
        params["w_singleton"],
        params["w_crowd"],
        params["w_color"],
    )
    cost_mat = compute_cost_mat(ctx)
    heard = L2(params["alpha"], params["cost"], ctx.lex, prior, cost_mat)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
