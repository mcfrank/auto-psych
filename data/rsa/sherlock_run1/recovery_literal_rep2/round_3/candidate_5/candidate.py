
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.Gamma(2.0, 1.0),
    "capacity": dist.Beta(2.0, 2.0),
    "w_rarity": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]

@memo
def S1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, prior) + {EPS})),
    )
    return Pr[speaker.u == u]

@memo
def L_bounded[u: UTT, r: OBJ](alpha, capacity, lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=L0[u, r](lex, prior)
        * exp(capacity * log(S1[u, r](alpha, lex, prior) + {EPS})),
    )
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    real_utt = (1.0 - ctx.is_sink)[:, None] * ctx.lex
    n_referents = jnp.sum(real_utt, axis=-1)
    rarity_weight = jnp.where(ctx.is_sink > 0, 0.0, 1.0 / (n_referents + EPS))
    rarity_score = jnp.sum(real_utt * rarity_weight[:, None], axis=0)

    prior = softmax_prior(
        params["w_rarity"] * rarity_score + params["w_familiar"] * ctx.familiarization
    )
    heard = L_bounded(params["alpha"], params["capacity"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
