"""Limited display attention listener (focal attention with distractor attenuation).

When interpreting referring expressions, listeners focus visual attention on the
candidate objects matching the heard description, while non-matching distractor
objects in the display receive attenuated attention. In evaluating how effectively
alternative expressions would describe candidate objects, the listener's mental
model of the speaker evaluates communicative informativeness over this
attention-weighted display rather than the full display.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "gamma_distractor": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., attn: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(attn, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., attn: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, attn) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    matches = ctx.lex[ctx.utterance]
    attn = jnp.where(matches > 0, 1.0, params["gamma_distractor"])
    heard = L1(params["alpha"], ctx.lex, attn)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
