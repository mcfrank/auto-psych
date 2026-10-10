"""Feature surprisal prior pragmatic listener.

Listeners expect speakers to communicate about referents possessing greater
contextual information content, assuming a prior over objects proportional to
total feature surprisal (Shannon self-information). Rarer features across the
visual scene carry greater communicative relevance. Pragmatic listeners invert
an informative speaker using this feature surprisal prior via Bayesian inference
(RSA depth 1). On uninformative prior trials without an informative word, choices
follow the feature surprisal prior directly. A lapse parameter captures random
guessing.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_surprisal": dist.Normal(0.0, 1.0),
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
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_surprisal_prior(w_surprisal, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    counts = jnp.sum(real_lex, axis=1)
    safe_counts = jnp.maximum(counts, 1.0)
    n_obj = ctx.lex.shape[1]
    feat_surprisal = (
        jnp.where(counts > 0, jnp.log(n_obj / safe_counts), 0.0)
        * (1.0 - ctx.is_sink)
    )
    total_surprisal = jnp.sum(real_lex * feat_surprisal[:, None], axis=0)
    return softmax_prior(w_surprisal * total_surprisal)


def choice_probs(params, ctx):
    prior = compute_surprisal_prior(params["w_surprisal"], ctx)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
