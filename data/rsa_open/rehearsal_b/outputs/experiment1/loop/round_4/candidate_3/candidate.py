"""Communicative regret speaker model.

Speakers choose referring expressions by balancing communicative recovery against the
regret of omitting a more informative description available for the target. Listeners
invert this regret-averse speaker, attributing underspecified descriptions to minimal
referents that lack distinguishing alternatives rather than multi-feature singletons.
On displays where a heard word is shared between duplicate minimal targets and
multi-feature singletons, this mechanism predicts listeners will strongly favor the
minimal targets because the speaker would regret omitting the singletons' distinguishing
features, disagreeing sharply with models whose visual singleton biases inflate belief
in the complex competitor.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.LogNormal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., regret: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - beta * at(regret, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def compute_regret(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_obj = ctx.lex.shape[1]
    ext = jnp.maximum(jnp.sum(real_lex, axis=1), 1.0)
    info = jnp.log(n_obj / ext) * (1.0 - ctx.is_sink)
    obj_max_info = jnp.max(real_lex * info[:, None], axis=0)
    regret = jnp.maximum(0.0, obj_max_info[None, :] - info[:, None]) * real_lex
    return regret


def listener(params, ctx):
    is_singleton = compute_singleton(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)
    regret = compute_regret(ctx)
    heard = L1(
        params["alpha"], params["beta"], ctx.lex, regret, prior
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
