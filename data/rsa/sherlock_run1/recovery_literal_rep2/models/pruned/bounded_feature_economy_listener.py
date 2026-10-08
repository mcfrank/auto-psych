"""Resource-bounded pragmatic listener with feature economy literal baseline.

Refines bounded_capacity_listener by incorporating the feature economy heuristic
from fewest_features_listener into the default literal interpretation: matching
referents with extraneous unmentioned features are penalized in proportion to
their total feature count. The pragmatic listener maintains bounded processing
capacity, anchoring on this feature-economical literal interpretation and
incorporating the speaker's communicative likelihood only to the extent
permitted by their cognitive resources.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.Gamma(2.0, 1.0),
    "capacity": dist.Beta(2.0, 2.0),
    "beta_features": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](beta_features, lex: ..., prior: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-beta_features * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


@memo
def S1[u: UTT, r: OBJ](alpha, beta_features, lex: ..., prior: ..., feature_count: ...):
    speaker: given(r in OBJ, wpp=1)
    speaker: chooses(
        u in UTT,
        wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](beta_features, lex, prior, feature_count) + {EPS})),
    )
    return Pr[speaker.u == u]


@memo
def L_bounded[u: UTT, r: OBJ](alpha, capacity, beta_features, lex: ..., prior: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=L0[u, r](beta_features, lex, prior, feature_count)
        * exp(capacity * log(S1[u, r](alpha, beta_features, lex, prior, feature_count) + {EPS})),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    heard = L_bounded(
        params["alpha"],
        params["capacity"],
        params["beta_features"],
        ctx.lex,
        prior,
        ctx.feature_count,
    )[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
