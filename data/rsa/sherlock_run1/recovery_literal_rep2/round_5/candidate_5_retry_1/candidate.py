"""Non-Bayesian heuristic listener with evaluative valence and color salience modulation.

Refines valence_feature_economy_listener by incorporating visual color salience from
color_salience_listener into the feature economy heuristic: when candidate referents
literally match the uttered word, listeners penalize objects possessing extraneous
unmentioned features (modulated by the speaker's evaluative valence) and discount
desaturated grayscale objects relative to full-color alternatives. Full-color objects
capture visual attention and enhance perceived referential suitability, whereas
desaturated grayscale objects are dispreferred.

Differences from valence_feature_economy_listener:
- Refined model: valence_feature_economy_listener
- Recursion depth: depth 0 (choice_probs calls L_heuristic, unchanged).
- Parameters added: w_grayscale ~ Normal(0.0, 1.0) governing the visual color salience
  penalty for desaturated grayscale referents.
- Parameters removed: none.
- Heuristic listener L_heuristic: candidate referents are weighted by
  at(lex, u, r) * exp(-beta_eff * vec(feature_count, r) + w_grayscale * vec(grayscale, r)),
  incorporating the visual color salience component from color_salience_listener.
- Arguments to L_heuristic: takes w_grayscale from params["w_grayscale"] and grayscale: ...
  from ctx.grayscale.
- All other memo agents, distributions, and prior terms are unchanged.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
    "w_grayscale": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_heuristic[u: UTT, r: OBJ](beta_eff, w_grayscale, lex: ..., feature_count: ..., grayscale: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta_eff * vec(feature_count, r) + w_grayscale * vec(grayscale, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    beta_eff = params["beta"] + params["w_valence"] * ctx.valence
    heard = L_heuristic(
        beta_eff, params["w_grayscale"], ctx.lex, ctx.feature_count, ctx.grayscale
    )[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
