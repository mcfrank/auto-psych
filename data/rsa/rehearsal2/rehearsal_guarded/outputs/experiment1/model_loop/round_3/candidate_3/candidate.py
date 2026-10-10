import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, beta, lex: ..., prior: ..., goal_weight: ..., comm: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            g in UTT,
            wpp=(at(lex, g, r) + {EPS}) * exp(beta * vec(goal_weight, g)),
        ),
        speaker: chooses(
            u in UTT,
            wpp=(at(lex, u, r) + {EPS}) * exp(alpha * log(at(comm, u, g) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(
        ctx.familiarization[:, None] - ctx.familiarization[None, :]
    )
    total_dist = pair_dist + gray_diff + fam_diff
    duplicate_count = jnp.sum(total_dist == 0, axis=1)
    return jnp.where(duplicate_count == 1, 1.0, 0.0)


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["w_singleton"] * is_singleton)

    n_obj = ctx.lex.shape[1]
    counts = jnp.maximum(jnp.sum(ctx.lex, axis=1), 1.0)
    goal_weight = -jnp.log(counts / n_obj)

    dots = ctx.lex @ ctx.lex.T
    comm = dots / counts[:, None]

    heard = L1(
        params["alpha"],
        params["beta"],
        ctx.lex,
        prior,
        goal_weight,
        comm,
    )[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
