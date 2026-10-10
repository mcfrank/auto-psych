import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta_utt": dist.Normal(0.0, 2.0),
    "beta_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def L_diag[u: UTT, r: OBJ](beta_utt, lex: ..., woe: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta_utt * at(woe, u, r)),
    )
    return Pr[listener.r == r]

def compute_woe_and_prior(ctx, eps=0.05):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    feat_count = jnp.maximum(jnp.sum(real_lex, axis=0, keepdims=True), 1.0)
    p_u_r = real_lex / feat_count
    n_obj = ctx.lex.shape[1]
    sum_p = jnp.sum(p_u_r, axis=1, keepdims=True)
    p_u_not_r = (sum_p - p_u_r) / jnp.maximum(n_obj - 1.0, 1.0)
    woe = jnp.log(p_u_r + eps) - jnp.log(p_u_not_r + eps)
    exp_woe = jnp.sum(real_lex * woe, axis=0) / feat_count[0]
    return woe, exp_woe

def choice_probs(params, ctx):
    woe, exp_woe = compute_woe_and_prior(ctx)
    prior = softmax_prior(params["beta_prior"] * exp_woe)
    heard = L_diag(params["beta_utt"], ctx.lex, woe)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
