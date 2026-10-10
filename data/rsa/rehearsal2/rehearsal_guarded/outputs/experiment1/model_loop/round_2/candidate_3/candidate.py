"""Pragmatic listener reasoning under lexical feature uncertainty.

Listeners interpret referring expressions under uncertainty about which visual
feature of the display an uttered word picks out. Rather than assuming words
map deterministically to features, listeners recognize that words may pick out
alternative features due to lexical ambiguity or communication noise. Simulated
speakers choose an intended feature to maximize target recovery, and pragmatic
listeners jointly infer the target object by integrating over their uncertainty
about the word-to-feature mapping.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "theta": dist.Beta(1.0, 9.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[f: UTT, r: OBJ](lex: ...):
    listener: knows(f)
    listener: chooses(r in OBJ, wpp=at(lex, f, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., trans: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r) * exp(alpha * log(L0[f, r](lex) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(trans, f, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., trans: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r)
            * exp(alpha * log(L1[f, r](alpha, lex, trans) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(trans, f, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_trans(is_sink: jnp.ndarray, theta) -> jnp.ndarray:
    n_utt = is_sink.shape[0]
    real_mask = (1.0 - is_sink)[:, None] * (1.0 - is_sink)[None, :]
    n_real = jnp.sum(1.0 - is_sink)
    denom = jnp.maximum(n_real - 1.0, 1.0)
    theta_eff = jnp.where(n_real > 1.0, theta, 0.0)

    eye = jnp.eye(n_utt)
    real_trans = (
        (1.0 - theta_eff) * eye + (theta_eff / denom) * (1.0 - eye)
    ) * real_mask
    sink_trans = jnp.outer(is_sink, is_sink)
    return real_trans + sink_trans


def choice_probs(params, ctx):
    trans = compute_trans(ctx.is_sink, params["theta"])
    heard = L2(params["alpha"], ctx.lex, trans)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
