import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, with_lapse

PARAMS = {
    "fam_weight": dist.Normal(0.0, 1.0),
    "gray_weight": dist.Normal(0.0, 1.0),
    "complexity_weight": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0)
}

@memo
def L1_inverts_S0[u: UTT, r: OBJ](fam_wt, gray_wt, comp_wt, lex: ..., fam: ..., gray: ..., fcount: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=exp(
            fam_wt * vec(fam, r) + 
            gray_wt * vec(gray, r) + 
            comp_wt * vec(fcount, r)
        )),
        speaker: chooses(u in UTT, wpp=at(lex, u, r))
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    heard = L1_inverts_S0(
        params["fam_weight"],
        params["gray_weight"],
        params["complexity_weight"],
        ctx.lex,
        ctx.familiarization,
        ctx.grayscale,
        ctx.feature_count
    )[ctx.utterance]
    
    prior_score = jnp.exp(
        params["fam_weight"] * ctx.familiarization +
        params["gray_weight"] * ctx.grayscale +
        params["complexity_weight"] * ctx.feature_count
    )
    prior_p = prior_score / jnp.sum(prior_score)
    
    return with_lapse(jnp.where(ctx.is_prior > 0, prior_p, heard), params["lapse"])
