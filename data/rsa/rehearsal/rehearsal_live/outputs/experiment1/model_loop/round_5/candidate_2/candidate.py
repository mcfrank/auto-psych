"""Pragmatic listener reasoning about a speaker sensitive to contextual ubiquity costs.

Speakers in reference games actively avoid producing referring expressions that are
contextually ubiquitous across the visual display, incurring a cognitive production
cost proportional to the fraction of display competitors sharing the feature.
Pragmatic listeners invert this ubiquity-averse speaker, inferring that when an ambiguous
expression is produced, the speaker lacked a less ubiquitous alternative for that
referent. On uninformative trials where no informative word is uttered, listener choices
reflect expectations that speakers prefer to communicate about referents with higher
baseline communicative contrast.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "c_ubiq": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, c_ubiq, lex: ..., ubiq_cost: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - c_ubiq * vec(ubiq_cost, u)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_ubiquity_and_prior(ctx):
    real_words = 1.0 - ctx.is_sink
    real_lex = ctx.lex * real_words[:, None]
    n_obj = ctx.lex.shape[1]
    n_comp = jnp.maximum(n_obj - 1.0, 1.0)

    extension = jnp.sum(real_lex, axis=1)
    competitors = jnp.maximum(extension - 1.0, 0.0)
    ubiq_cost = (competitors / n_comp) * real_words

    masked_ubiq = jnp.where(real_lex > 0, ubiq_cost[:, None], 1.0)
    min_ubiq = jnp.min(masked_ubiq, axis=0)

    return ubiq_cost, min_ubiq


def choice_probs(params, ctx):
    ubiq_cost, min_ubiq = compute_ubiquity_and_prior(ctx)
    prior = softmax_prior(-params["w_prior"] * min_ubiq)
    heard = L1(params["alpha"], params["c_ubiq"], ctx.lex, ubiq_cost)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
