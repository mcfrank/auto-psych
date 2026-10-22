"""Question Under Discussion (QUD) goal uncertainty pragmatic listener.

Speakers choose utterances to address one of two discourse goals (Questions Under Discussion):
either uniquely identifying the target referent, or truthfully attributing a property to it.
Listeners perform joint Bayesian inference over the speaker's target referent and communicative
goal, expecting uniquely identifying descriptions when speakers aim to identify, but retaining
graded credibility for competitors that share the named property when speakers merely describe
an attribute. On uninformative prior trials, visual pop-out favors contextually unique singletons
and feature-distinctive items.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_qud": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
}

QUD = jnp.arange(2)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L_qud[u: UTT, r: OBJ](alpha, w_qud, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(q in QUD, wpp=exp(w_qud) if q == 0 else 1.0),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * (exp(alpha * log(L0[u, r](lex) + {EPS})) if q == 0 else 1.0),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_singleton, w_distinct):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    duplicate_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(duplicate_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    return softmax_prior(w_singleton * is_singleton + w_distinct * distinct)


def listener(params, ctx):
    prior = compute_prior(ctx, params["w_singleton"], params["w_distinct"])
    heard = L_qud(params["alpha"], params["w_qud"], ctx.lex, prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
