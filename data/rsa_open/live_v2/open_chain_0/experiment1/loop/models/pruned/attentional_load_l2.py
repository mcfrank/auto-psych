"""Depth-2 resource-limited listener with capacity-bounded attentional allocation across display features.

Refines attentional_load_l1 by increasing recursion depth from L1 to L2 (L2 -> L1 -> L0),
passing the shared salience prior and speaker confusion penalty into both L1 and L2 while
keeping the two parameters (alpha, w_confusion), compute_attention_and_prior, and the
attentional overload mixture in listener unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_confusion": dist.Normal(0.0, 2.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., confusion: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, w_confusion, lex: ..., confusion: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, w_confusion, lex, confusion, prior) + {EPS})
                - w_confusion * at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_attention_and_prior(ctx, w_confusion):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feats = jnp.vstack([real_lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    n_real = jnp.maximum(jnp.sum(1.0 - ctx.is_sink), 1.0)
    fc_norm = ctx.feature_count / n_real
    fc_dev = fc_norm - jnp.mean(fc_norm)
    asymmetric_extremity = fc_dev * fc_dev / (fc_norm + 1.0 / n_real)

    confusion = real_lex @ sim
    baseline_confusion = jnp.sum(sim, axis=1)

    prior = softmax_prior(w_confusion * (asymmetric_extremity - baseline_confusion))

    ext_size = jnp.sum(real_lex, axis=1, keepdims=True)
    false_size = n_obj - ext_size
    overload = jnp.maximum(ext_size - false_size, 0.0) * jnp.minimum(false_size, 1.0) / n_obj
    return overload, confusion, prior


def listener(params, ctx):
    overload, confusion, prior = compute_attention_and_prior(
        ctx, params["w_confusion"]
    )
    prag = L2(
        params["alpha"], params["w_confusion"], ctx.lex, confusion, prior
    )[ctx.utterance]
    load = overload[ctx.utterance, 0] * overload[ctx.utterance, 0]
    heard = (1.0 - load) * prag + load * prior
    return jnp.where(ctx.is_prior > 0, prior, heard)
