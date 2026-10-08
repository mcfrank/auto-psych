"""Non-Bayesian heuristic listener with diminishing marginal feature economy.

Refines valence_feature_economy_listener by replacing the linear feature count
penalty with a logarithmic Weber-Fechner functional form: listeners penalize
candidate referents possessing extraneous unmentioned features, but their
sensitivity to complexity exhibits diminishing marginal returns. The first
extraneous feature incurs the sharpest penalty for departing from minimality,
while subsequent extraneous features add diminishing marginal costs. Under
evaluative valence framing, this diminishing penalty is modulated by the
speaker's evaluative stance.

Differences from valence_feature_economy_listener:
- Refined model: valence_feature_economy_listener
- Recursion depth: depth 0 (choice_probs calls L_diminishing, unchanged).
- Parameters added: none.
- Parameters removed: none.
- Functional form in L_diminishing: listeners weight matching referents proportional
  to at(lex, u, r) * exp(-beta_eff * log(1.0 + vec(extraneous, r))) rather than
  at(lex, u, r) * exp(-beta_eff * vec(feature_count, r)), replacing linear feature
  penalization with a logarithmic Weber-Fechner functional form over extraneous features.
- Extraneous features in choice_probs: computed as jnp.maximum(0.0, ctx.feature_count - 1.0)
  and passed to L_diminishing.
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
def L_diminishing[u: UTT, r: OBJ](beta_eff, lex: ..., extraneous: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-beta_eff * log(1.0 + vec(extraneous, r))),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    beta_eff = params["beta"] + params["w_valence"] * ctx.valence
    extraneous = jnp.maximum(0.0, ctx.feature_count - 1.0)
    heard = L_diminishing(beta_eff, ctx.lex, extraneous)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
