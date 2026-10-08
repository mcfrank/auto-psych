"""Mixture of recursive pragmatic and perceptual distinctiveness heuristic listeners.

Listeners in reference games differ in their cognitive strategy: the population
comprises a mixture of communicative pragmatic listeners, who reason recursively
at depth 2 about speaker informativeness, and perceptual heuristic listeners, who
choose among matching referents in proportion to their contextual visual feature
distinctiveness relative to the display. On uninformative prior trials, pragmatic
listeners choose uniformly while perceptual heuristic listeners choose based on
visual distinctiveness. Population choices reflect this latent mixture, with the
pragmatic proportion as a free parameter alongside speaker rationality,
distinctiveness weight, and lapse rate.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "p_pragmatic": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L_distinct[u: UTT, r: OBJ](w_distinct, lex: ..., distinct: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(w_distinct * vec(distinct, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    l2_heard = L2(params["alpha"], ctx.lex)[ctx.utterance]
    dist_heard = L_distinct(params["w_distinct"], ctx.lex, distinct)[ctx.utterance]

    p_prag = params["p_pragmatic"]
    heard = p_prag * l2_heard + (1.0 - p_prag) * dist_heard

    l2_prior = jnp.full_like(heard, 1.0 / heard.shape[0])
    dist_prior = softmax_prior(params["w_distinct"] * distinct)
    prior = p_prag * l2_prior + (1.0 - p_prag) * dist_prior

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
