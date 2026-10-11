"""Feature informativeness listener: pragmatic listener with an information-theoretic referent prior.

Listeners interpret referring expressions by reasoning about an informative speaker,
grounding their prior expectations in visual feature informativeness (Shannon surprisal).
Ubiquitous features shared by all objects carry zero surprisal, whereas rare and unique
features carry high information content. Pragmatic listeners invert this speaker with
the feature informativeness prior, resolving ambiguous and ubiquitous expressions while
moderating scalar implicatures in multi-attribute displays.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_info": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L1[u, r](alpha, lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_feature_informativeness(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    k_u = jnp.sum(real_lex, axis=1)
    n_obj = ctx.lex.shape[1]
    surprisal = jnp.log(n_obj / jnp.maximum(k_u, 1.0)) * (1.0 - ctx.is_sink)
    return jnp.sum(real_lex * surprisal[:, None], axis=0)


def choice_probs(params, ctx):
    info = compute_feature_informativeness(ctx)
    prior = softmax_prior(params["w_info"] * info)
    heard = L2(params["alpha"], ctx.lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
