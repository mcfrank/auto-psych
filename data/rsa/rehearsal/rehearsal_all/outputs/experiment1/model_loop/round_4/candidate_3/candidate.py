"""Pragmatic listener reasoning about a contrast-criterion-sensitive speaker.

Speakers in reference games adhere to a contrast criterion, selecting referring
expressions that establish an explicit contrast against a visually similar competitor
in the display that lacks the named feature. The communicative utility of a descriptive
word is enhanced proportionally to how closely a contrast partner in the context shares
the referent's other features while lacking the target feature. Pragmatic listeners invert
this contrast-sensitive speaker to resolve referential ambiguity, expecting ambiguous
descriptions to identify referents that possess a minimal contrast partner rather than
isolated or contrast-poor alternatives. On uninformative prior trials, listeners default
to uniform choice. A lapse parameter accounts for random clicking noise.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_contrast, lex: ..., contrast: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                + w_contrast * at(contrast, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_contrast(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    diff = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    total_dist = jnp.sum(diff, axis=0)
    dist_non_u = total_dist[None, :, :] - diff

    n_obj = ctx.lex.shape[1]
    eye = jnp.eye(n_obj)[None, :, :]

    lacks_u = (1.0 - real_lex)[:, None, :]
    valid_partner = lacks_u * (1.0 - eye)

    sim = jnp.exp(-dist_non_u) * valid_partner
    contrast = jnp.max(sim, axis=-1) * real_lex
    return contrast


def choice_probs(params, ctx):
    contrast = compute_contrast(ctx)
    heard = L1(
        params["alpha"], params["w_contrast"], ctx.lex, contrast
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
