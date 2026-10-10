"""Incremental alternative elimination speaker and pragmatic listener.

Speakers formulate referring expressions by evaluating candidate descriptors against
an active communicative alternative feature of the intended referent, choosing words
that eliminate visual competitors the alternative feature left confusable. Pragmatic
listeners invert this two-stage communicative process, jointly inferring the intended
referent and the alternative descriptor the speaker sought to improve upon. On
uninformative trials with no descriptive word, choices follow visual singleton
salience. A lapse parameter captures random clicking.
"""

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
def L1[
    u: UTT,
    r: OBJ,
](
    alpha,
    beta,
    lex: ...,
    prior: ...,
    salience: ...,
    elim: ...,
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            a in UTT,
            wpp=(at(lex, a, r) + {EPS}) * exp(beta * vec(salience, a)),
        ),
        speaker: chooses(
            u in UTT,
            wpp=(at(lex, u, r) + {EPS}) * exp(alpha * at(elim, u, a)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_salience_and_elim(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    n_obj = ctx.lex.shape[1]
    n_true = jnp.sum(real_lex, axis=1)

    # Contextual feature surprisal for alternative weighting
    freq = jnp.where(n_true > 0, n_true / n_obj, 1.0)
    salience = real_words * (-jnp.log(freq))

    # Incremental competitor elimination:
    # How many objects satisfying alternative 'a' are eliminated by utterance 'u'
    # elim[u, a] = sum_r (1 - real_lex[u, r]) * real_lex[a, r]
    elim = (1.0 - real_lex) @ real_lex.T
    return salience, elim


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
    salience, elim = compute_salience_and_elim(ctx)

    heard = L1(
        params["alpha"],
        params["beta"],
        ctx.lex,
        prior,
        salience,
        elim,
    )[ctx.utterance]

    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
