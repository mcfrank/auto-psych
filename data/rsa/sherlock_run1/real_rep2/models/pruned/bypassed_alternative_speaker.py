"""Bypassed-alternative speaker: referent-grounded opportunity cost of unchosen words.

When choosing a referring expression for an intended referent, speakers evaluate
candidate words against the opportunity cost of bypassed alternatives: choosing an
unspecific word when a more specific alternative word applies to that referent
incurs a production penalty proportional to the bypassed informativeness. In
contrast, when an intended referent has no better naming alternatives (a speaker
who cannot name it more specifically), the speaker uses an unspecific word without
penalty. Pragmatic listeners invert this cost-sensitive speaker, allowing the
availability of referent-specific alternatives to guide pragmatic implicatures.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost_bypassed": dist.Normal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex, prior) + {EPS}) - at(cost, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    non_sink_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_referents = jnp.sum(non_sink_lex, axis=-1, keepdims=True)
    specificity = jnp.where(n_referents > 0.0, 1.0 / n_referents, 0.0)
    word_spec = non_sink_lex * specificity
    eye_u = jnp.eye(ctx.lex.shape[0])[:, :, None]
    other_words_spec = (1.0 - eye_u) * word_spec[None, :, :]
    preemption = jnp.max(other_words_spec, axis=1)
    cost = params["cost_bypassed"] * preemption

    prior = softmax_prior(
        params["w_features"] * ctx.feature_count
        + params["w_familiar"] * ctx.familiarization
    )
    heard = L1(params["alpha"], ctx.lex, prior, cost)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
