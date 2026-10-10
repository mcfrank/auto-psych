"""Pragmatic listener reasoning about a markedness-congruence-sensitive speaker.

Speakers in reference games adhere to Horn's division of pragmatic labor, matching
the contextual markedness of a referring expression to the visual complexity of the
intended referent. Unmarked expressions that apply broadly across the context are
preferred for simple, feature-sparse objects, whereas marked expressions that are
rare in the display are reserved for complex, feature-rich objects. Pragmatic listeners
invert this markedness-sensitive speaker to resolve referential ambiguity, expecting
broad descriptions to single out simple items and narrow descriptions to identify
complex items. On uninformative prior trials, listeners default to uniform choice.
A lapse parameter accounts for random clicking noise.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_marked": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_marked, lex: ..., congruence: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_marked * at(congruence, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_congruence(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words

    utt_counts = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]
    frac = utt_counts / jnp.maximum(n_obj, 1.0)
    utt_markedness = -jnp.log(jnp.maximum(frac, 1e-4)) * (1.0 - ctx.is_sink)
    n_real_words = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    mean_utt_markedness = jnp.sum(utt_markedness) / n_real_words
    centered_u = (utt_markedness - mean_utt_markedness)[:, None]

    obj_complexity = ctx.feature_count
    mean_complexity = jnp.mean(obj_complexity)
    centered_r = (obj_complexity - mean_complexity)[None, :]

    congruence = (centered_u * centered_r) * real_words
    return congruence


def choice_probs(params, ctx):
    congruence = compute_congruence(ctx)
    heard = L1(
        params["alpha"], params["w_marked"], ctx.lex, congruence
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
