"""Salience-biased softmax decision listener.

Listeners interpret referring expressions by combining pure communicative inference with a
salience-biased softmax decision rule that trades communicative evidence against visual feature
complexity. Rather than attributing perceptual preferences to the speaker as a common-knowledge
prior, the listener evaluates candidate referents at the decision stage by combining the log-evidence
of referent recovery with an intrinsic preference for distinctive, feature-rich objects. This decision
rule naturally moderates extreme scalar implicatures on ambiguous and ubiquitous displays while
capturing strong preferences for salient referents on uninformative trials.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta_comm": dist.LogNormal(0.0, 0.5),
    "w_singleton": dist.Normal(1.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
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


def choice_probs(params, ctx):
    diffs = jnp.abs(ctx.lex[:, :, None] - ctx.lex[:, None, :])
    pair_dist = jnp.sum(diffs * (1.0 - ctx.is_sink[:, None, None]), axis=0)
    dup_count = jnp.sum(pair_dist == 0, axis=1)
    is_singleton = jnp.where(dup_count == 1, 1.0, 0.0)

    real_lex = ctx.lex * (1.0 - ctx.is_sink[:, None])
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    distinct = jnp.sum(diff, axis=(0, 2))

    salience = (
        params["w_singleton"] * is_singleton
        + params["w_distinct"] * distinct
        + params["w_features"] * ctx.feature_count
    )

    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]

    u_comm = jnp.where(
        heard > 0,
        params["beta_comm"] * jnp.log(heard + EPS),
        -1e4,
    )
    u_dec = jnp.where(ctx.is_prior > 0, salience, u_comm + salience)

    choice = jax.nn.softmax(u_dec)
    return with_lapse(choice, params["lapse"])
