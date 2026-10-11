"""Pragmatic listener reasoning over uncertain communicative goals (discriminative vs descriptive QUD).

Listeners interpret referring expressions by reasoning about a speaker with uncertain
communicative goals, balancing discriminative identification against descriptive feature
attribution. When hearing an ambiguous or ubiquitous word that fails to isolate a unique
object, listeners infer that the speaker was operating under a descriptive rather than
discriminative goal, evaluating referents by visual salience and feature complexity rather
than drawing an exhaustive scalar exclusion inference.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "log_alpha": dist.Normal(1.0, 0.5),
    "logit_p_ident": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

GOALS = jnp.arange(2)


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L_qud[u: UTT, r: OBJ](alpha, p_ident, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(goal in GOALS, wpp=p_ident if goal == 1 else (1.0 - p_ident)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * goal * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_singleton_indicator(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    real_features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(real_features[:, :, None] - real_features[:, None, :])
    dist_matrix = jnp.sum(diff, axis=0)
    copy_count = jnp.sum(dist_matrix == 0, axis=1)
    return jnp.where(copy_count == 1, 1.0, 0.0)


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx.lex, ctx.is_sink)
    prior = softmax_prior(
        params["w_singleton"] * is_singleton
        + params["w_features"] * ctx.feature_count
    )
    alpha = jnp.exp(params["log_alpha"])
    p_ident = jax.nn.sigmoid(params["logit_p_ident"])
    heard = L_qud(alpha, p_ident, ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
