"""RSA depth-1 pragmatic listener with graded truth values.

Rather than evaluating utterances with binary all-or-nothing semantics, the
literal listener evaluates referents according to graded truth values: an
object's semantic compatibility with a feature term decays exponentially with
each additional unmentioned feature it possesses, reflecting graded
prototypicality and exemplar specificity. A rational speaker chooses utterances
to maximize communicative informativeness under these graded semantics, and
the pragmatic listener inverts this speaker. Prior-elicitation trials reflect
uniform guessing. A lapse parameter accounts for random decision noise.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](truth: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(truth, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, truth: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(truth, u, r) * exp(alpha * log(L0[u, r](truth, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    unmentioned = jnp.maximum(0.0, ctx.feature_count[None, :] - ctx.lex)
    truth = ctx.lex * jnp.exp(-params["gamma"] * unmentioned)
    heard = L1(params["alpha"], truth, prior)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
