"""RSA pragmatic listener with base-rate-informed literal listener, focal attention, and prior-anchored lapse.

Refines prior_lapse_focal_listener: in addition to having the speaker and final choice
distribution reflect baseline prior expectations (visual feature simplicity and
familiarization base rates), the literal listener (L0) directly integrates empirical
familiarization base rates into its literal referent choice rule alongside focal
attentional discounting. The speaker anticipates a literal listener who already expects
frequent referents, strengthening communicative pressure to produce distinctive
utterances for rare referents while allowing concise, shared descriptions for familiar ones.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, vec, softmax_prior

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "salience": dist.Normal(0.0, 1.0),
    "baserate": dist.LogNormal(0.0, 1.0),
    "distractor_weight": dist.Beta(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ..., base_prior: ..., atten: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * (vec(base_prior, r) + {EPS}) * (vec(atten, r) + {EPS}),
    )
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., prior: ..., base_prior: ..., atten: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](lex, base_prior, atten) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    is_cand = jnp.where(ctx.is_prior > 0, 1.0, ctx.lex[ctx.utterance])
    atten = jnp.where(is_cand > 0, 1.0, params["distractor_weight"])
    fam_logits = jnp.where(
        ctx.has_familiarization > 0, jnp.log(ctx.familiarization + EPS), 0.0
    )
    base_prior = softmax_prior(params["baserate"] * fam_logits)
    prior = softmax_prior(
        params["salience"] * ctx.feature_count + params["baserate"] * fam_logits
    )
    heard = L1(params["alpha"], ctx.lex, prior, base_prior, atten)[ctx.utterance]
    p_choice = jnp.where(ctx.is_prior > 0, prior, heard)
    return (1.0 - params["lapse"]) * p_choice + params["lapse"] * prior
