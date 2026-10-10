"""Pragmatic listener reasoning about a bounded communicative payoff speaker.

Speakers in reference games evaluate utterances under a bounded linear payoff
representing the expected probability of communicative success, rather than
unbounded logarithmic information surprisal. Pragmatic listeners invert this
speaker: when an ambiguous referring expression is produced, candidate referents
possessing unique distinguishing features are penalized, because a rational
speaker intending them would have secured unambiguous communication with their
unique label. On uninformative prior trials without an informative word, listener
expectations directly track expected communicative success, favoring communicable
singletons over incommunicable duplicates.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(1.0, 1.0),
    "w_comm": dist.Normal(0.0, 2.0),
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
            wpp=at(lex, u, r) * exp(alpha * L0[u, r](lex)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_communicative_prior(w_comm, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.maximum(jnp.sum(real_lex, axis=1, keepdims=True), 1.0)
    l0_table = real_lex / ext
    success = jnp.max(l0_table * real_lex, axis=0)
    return softmax_prior(w_comm * success)


def choice_probs(params, ctx):
    prior = compute_communicative_prior(params["w_comm"], ctx)
    heard = L1(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard),
        params["lapse"],
    )
