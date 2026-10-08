"""Non-Bayesian heuristic listener with evaluative valence modulation of feature economy.

Refines fewest_features_listener by incorporating the speaker's evaluative stance
into the feature economy heuristic: listeners penalize candidate referents that
possess extraneous unmentioned features, but modulate this penalty according to
the speaker's evaluative framing. Under neutral framing, listeners apply the
default penalty favoring minimally specified referents; when the speaker describes
a 'favorite' object, extraneous features are penalized more heavily or expected to
align with prominence, whereas under negative framing ('least favorite'), the
evaluative polarity shifts, softening or inverting the penalty on extraneous features.

Differences from fewest_features_listener:
- Refined model: fewest_features_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: w_valence ~ Normal(0.0, 1.0) governing the modulation of
  feature economy by speaker valence framing.
- Parameters removed: none.
- Feature economy penalty in choice_probs: the effective feature penalty is
  beta_eff = params["beta"] + params["w_valence"] * ctx.valence.
- Heuristic listener L_heuristic: takes beta_eff instead of params["beta"].
- All other memo agents, distributions, and prior terms are unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta_eff, lex: ..., feature_count: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta_eff * vec(feature_count, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    beta_eff = params["beta"] + params["w_valence"] * ctx.valence
    heard = L_heuristic(beta_eff, ctx.lex, ctx.feature_count)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
