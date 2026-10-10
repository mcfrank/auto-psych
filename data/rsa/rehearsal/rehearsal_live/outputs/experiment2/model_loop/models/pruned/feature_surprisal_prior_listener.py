"""Feature surprisal prior pragmatic listener.

Listeners interpret referring expressions by modeling a communicative speaker
who chooses intended referents according to their information-theoretic feature
surprisal, expecting speakers to target objects whose visual features are rare
within the context display. On uninformative prior trials, listener choices
track this feature surprisal prior directly; when an informative expression
is produced, pragmatic listeners invert the speaker via Bayes' rule, integrating
the surprisal prior with semantic applicability.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_surprisal(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    n_obj = ctx.lex.shape[1]

    extension = jnp.sum(real_lex, axis=1)
    prevalence = jnp.maximum(extension / n_obj, 1.0 / n_obj)
    feat_surprisal = -jnp.log(prevalence) * real_words

    obj_surprisal = jnp.sum(real_lex * feat_surprisal[:, None], axis=0)
    return obj_surprisal


def choice_probs(params, ctx):
    surprisal = compute_surprisal(ctx)
    prior = softmax_prior(params["w_prior"] * surprisal)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
