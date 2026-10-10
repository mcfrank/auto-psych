"""Pragmatic listener reasoning about an alternative-inhibition speaker.

Speakers in reference games experience lateral lexical inhibition among candidate
referring expressions, where the presence of a dominant, unambiguous alternative
description for an object actively inhibits competing ambiguous words. Pragmatic
listeners invert this competitive production process, inferring that hearing an
ambiguous word indicates the speaker intended a referent that lacked any dominant
alternative label. On uninformative trials where no informative word is uttered,
listener expectations reflect baseline communicative accessibility, favoring
referents that possess uninhibited identifying descriptions.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_inhibit": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, c_inhibit, lex: ..., inhibition: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - c_inhibit * at(inhibition, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_inhibition(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    extension = jnp.maximum(jnp.sum(real_lex, axis=1, keepdims=True), 1.0)
    l0_table = real_lex / extension  # (N_UTT, N_OBJ)

    n_utt = ctx.lex.shape[0]
    diag_mask = (1.0 - jnp.eye(n_utt))[:, :, None]
    l0_others = jnp.where(
        (diag_mask > 0.5) & (real_lex[:, None, :] > 0),
        l0_table[:, None, :],
        0.0,
    )
    inhibition = jnp.max(l0_others, axis=0) * real_lex  # (N_UTT, N_OBJ)

    net_info = jnp.where(real_lex > 0, l0_table - inhibition, -1.0)
    accessibility = jnp.maximum(jnp.max(net_info, axis=0), 0.0)

    return inhibition, accessibility


def choice_probs(params, ctx):
    inhibition, accessibility = compute_inhibition(ctx)
    prior = softmax_prior(params["w_prior"] * accessibility)
    heard = L1(params["alpha"], params["c_inhibit"], ctx.lex, inhibition)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
