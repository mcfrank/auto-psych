
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "beta_salience": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]

@memo
def L1[u: UTT, r: OBJ](alpha, lex: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS}))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def compute_singleton_indicator(ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist = jnp.sum(jnp.abs(diff), axis=0)
    is_identical = (dist < 1e-5).astype(jnp.float32)
    copy_count = jnp.sum(is_identical, axis=1)
    is_item_singleton = (copy_count <= 1.0).astype(jnp.float32)
    singleton_count = jnp.sum(is_item_singleton)
    is_solitary_singleton = jnp.where(
        (is_item_singleton > 0.0) & (singleton_count > 0.5) & (singleton_count < 1.5),
        1.0,
        0.0,
    )
    return is_solitary_singleton

def choice_probs(params, ctx):
    is_singleton = compute_singleton_indicator(ctx)
    p_salience = softmax_prior(params["beta_salience"] * is_singleton)
    heard = L1(params["alpha"], ctx.lex)[ctx.utterance]
    # Decision rule: structured lapse towards bottom-up visual salience
    choice_heard = (1.0 - params["lapse"]) * heard + params["lapse"] * p_salience
    choice_prior = with_lapse(p_salience, params["lapse"])
    return jnp.where(ctx.is_prior > 0, choice_prior, choice_heard)
