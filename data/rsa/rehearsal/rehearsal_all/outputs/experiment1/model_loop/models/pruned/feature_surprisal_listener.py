"""Pragmatic listener with Shannon feature-surprisal object prior.

People expect communicative referents to be visually salient according to their
informational surprisal, assigning higher prior probability to objects possessing
rare features or unexpected feature absences. Observers evaluate the total Shannon
information content of each candidate referent relative to the display, anticipating
that speakers are more likely to select items with rare, distinctive visual
characteristics. Pragmatic listeners integrate this feature-surprisal prior with
depth-two recursive communicative reasoning to interpret referring expressions and
make spontaneous referent choices on uninformative trials.
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
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_surprisal):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    counts = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]

    p = counts / n_obj
    eps = 1e-6
    surp_pres = -jnp.log(jnp.clip(p, eps, 1.0))[:, None] * real_lex
    surp_abs = (
        -jnp.log(jnp.clip(1.0 - p, eps, 1.0))[:, None]
        * (1.0 - real_lex)
        * real_words[:, None]
    )
    surprisal = jnp.sum(surp_pres + surp_abs, axis=0)
    return softmax_prior(w_surprisal * surprisal)


def choice_probs(params, ctx):
    prior = compute_prior(ctx, params["w_surprisal"])
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
