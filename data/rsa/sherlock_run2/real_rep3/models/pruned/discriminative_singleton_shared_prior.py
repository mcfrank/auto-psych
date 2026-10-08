"""RSA pragmatic listener (depth 1) with discriminative alternative cost and shared singleton prior.

Refines rsa_l1_singleton_shared_prior by incorporating a communicative penalty
on ambiguous words when distinguishing alternatives exist (taking the discriminative
alternative cost component from discriminative_alternative_speaker). A speaker
who uses an ambiguous word when a uniquely distinguishing word for their intended
referent was available incurs a production cost. Pragmatic listeners invert this
speaker, integrating mutual perceptual salience (singleton isolation, feature
parsimony, and familiarization base rates) with the expectation that speakers
actively avoid ambiguity when informative alternatives are present.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "cost": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 2.0),
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
def L1[u: UTT, r: OBJ](alpha, cost, lex: ..., prior: ..., cost_mat: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(alpha * log(L0[u, r](lex, prior) + {EPS}) - cost * at(cost_mat, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_prior(ctx, w_singleton, w_features, w_familiar):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_val = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist_val < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)
    return softmax_prior(
        w_singleton * is_singleton
        + w_features * ctx.feature_count
        + w_familiar * ctx.familiarization
    )


def choice_probs(params, ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    extension = jnp.sum(real_lex, axis=1)
    is_unique = jnp.where(extension == 1.0, 1.0, 0.0)
    has_unique = jnp.where(jnp.sum(real_lex * is_unique[:, None], axis=0) > 0.0, 1.0, 0.0)
    is_ambiguous = jnp.where(extension > 1.0, 1.0, 0.0)
    cost_mat = is_ambiguous[:, None] * has_unique[None, :]

    prior = compute_prior(
        ctx, params["w_singleton"], params["w_features"], params["w_familiar"]
    )
    heard = L1(params["alpha"], params["cost"], ctx.lex, prior, cost_mat)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
