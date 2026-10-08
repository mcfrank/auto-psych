"""Pragmatic listener reasoning about a speaker who avoids competitor confusion.

Speakers in reference games actively avoid communicative confusion by penalizing
utterances that are shared with contextually similar competitors. The communicative
cost of using an utterance for an intended referent increases with the visual feature
similarity between that referent and other display items that also possess the named
feature. Pragmatic listeners invert this confusion-averse speaker, expecting referring
expressions to target referents whose competitors are visually distinct rather than
confusingly similar.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confuse": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confuse, lex: ..., confusion: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confuse * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    eye = jnp.eye(dist_mat.shape[0])
    sim = jnp.exp(-dist_mat) * (1.0 - eye)
    confusion = jnp.sum(real_lex[:, None, :] * sim[None, :, :], axis=2)

    heard = L1(params["alpha"], params["w_confuse"], ctx.lex, confusion)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
