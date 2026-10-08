
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "p_perceptual": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

def object_isolation(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    features = lex * (1.0 - is_sink)[:, None]
    diff = jnp.abs(features[:, :, None] - features[:, None, :])
    pair_dist = jnp.sum(diff, axis=0)
    n_obj = lex.shape[1]
    eye = jnp.eye(n_obj) * 1e5
    return jnp.min(pair_dist + eye, axis=1)

@memo
def L_perceptual[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r) * vec(prior, r))
    return Pr[listener.r == r]

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

def choice_probs(params, ctx):
    dist_vec = object_isolation(ctx.lex, ctx.is_sink)
    prior = softmax_prior(params["w_distinct"] * dist_vec)
    
    p_perc = L_perceptual(ctx.lex, prior)[ctx.utterance]
    p_prag = L1(params["alpha"], ctx.lex)[ctx.utterance]
    
    heard = params["p_perceptual"] * p_perc + (1.0 - params["p_perceptual"]) * p_prag
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
