"""Pragmatic listener reasoning about a speaker with graded truth semantics.

Word meanings have graded truth values: an utterance is literally true of an object
to the degree that it comprehensively describes that object. Candidate referents with
more extraneous features receive lower semantic truth values for any single feature word.
The literal listener uses graded truth, the speaker optimizes communicative informativeness,
and the pragmatic listener inverts the speaker.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.Normal(0.0, 1.0),
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
    # Graded truth: feature truth value scales with feature specificity
    # Extraneous features dilute the completeness of a single feature description
    degree = jnp.exp(-params["gamma"] * jnp.maximum(ctx.feature_count - 1.0, 0.0))
    # Real utterances scale by degree; the sink utterance stays crisp at 1.0
    graded_lex = ctx.lex * jnp.where(ctx.is_sink[:, None] > 0, 1.0, degree[None, :])
    heard = L1(params["alpha"], graded_lex)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
