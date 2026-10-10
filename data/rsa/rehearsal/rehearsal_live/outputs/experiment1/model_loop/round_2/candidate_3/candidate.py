"""Feature-grounding uncertainty listener.

Listeners interpret referring expressions under feature-grounding uncertainty,
reasoning that a speaker chooses an intended descriptive feature to maximize
communicative informativeness, but that transmission across the communicative
channel is subject to semantic feature confusion between visual attributes
present in the context. On uninformative trials where no informative word is
given, listeners choose uniformly among available objects. A lapse parameter
captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "noise": dist.Beta(1.0, 9.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., channel: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            f in UTT,
            wpp=at(lex, f, r) * exp(alpha * log(L0[f, r](lex) + {EPS})),
        ),
        speaker: chooses(u in UTT, wpp=at(channel, f, u)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_channel(is_sink: jnp.ndarray, noise) -> jnp.ndarray:
    """Construct channel confusion matrix between utterances."""
    n_utt = is_sink.shape[0]
    eye = jnp.eye(n_utt)
    is_real = (1.0 - is_sink)[:, None] * (1.0 - is_sink)[None, :]
    # For real features: 1.0 on diagonal, noise on off-diagonal
    real_channel = (eye + noise * (1.0 - eye)) * is_real
    # For sink utterance: identity only
    sink_channel = eye * (is_sink[:, None] * is_sink[None, :])
    return real_channel + sink_channel


def choice_probs(params, ctx):
    channel = compute_channel(ctx.is_sink, params["noise"])
    heard = L1(params["alpha"], ctx.lex, channel)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard),
        params["lapse"],
    )
