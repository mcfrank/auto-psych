"""Two-stage perceptual oddity listener with attentional attenuation.

Refining oddity_heuristic_listener by decoupling the ungrounded prior oddity
pop-out effect from the grounded linguistic interpretation oddity bias.
When no informative utterance is heard, listeners exhibit strong visual oddity
pop-out favoring unique singletons (beta_prior). When an informative referring
expression is heard, linguistic semantic constraints compete with visual
pop-out, resulting in an attenuated oddity bias (beta_utt) among semantically
satisfying referents. A lapse parameter captures random clicking.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_prior": dist.Normal(0.0, 2.0),
    "beta_utt": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heur[u: UTT, r: OBJ](beta_utt, lex: ..., is_singleton: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta_utt * vec(is_singleton, r)),
    )
    return Pr[listener.r == r]


def compute_singleton_indicator(ctx):
    feats = jnp.vstack([
        ctx.lex,
        ctx.grayscale[None, :],
        ctx.familiarization[None, :],
    ])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_singleton = (copy_count <= 1.0).astype(jnp.float32)
    return is_singleton


def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    prior = softmax_prior(params["beta_prior"] * is_singleton)
    heard = L_heur(params["beta_utt"], ctx.lex, is_singleton)[ctx.utterance]
    return with_lapse(
        jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"]
    )
