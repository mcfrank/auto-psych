"""Statistical preemption listener: direct cue competition without speaker recursion.

People interpret referring expressions through statistical preemption: when an
unspecific word applies to multiple candidate referents, listeners penalize a
referent in proportion to whether the speaker bypassed a more specific,
unambiguous alternative word that applies to it. Interlocutors evaluate cue
competition directly rather than performing recursive social simulation.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "beta": dist.LogNormal(0.0, 1.0),
    "w_contrast": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_preempt[u: UTT, r: OBJ](beta, lex: ..., preemption: ..., prior: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=vec(prior, r) * at(lex, u, r) * exp(-beta * at(preemption, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    non_sink_lex = ctx.lex * (1.0 - ctx.is_sink)[:, None]
    n_referents = jnp.sum(non_sink_lex, axis=-1, keepdims=True)
    specificity = jnp.where(n_referents > 0.0, 1.0 / n_referents, 0.0)
    word_spec = non_sink_lex * specificity
    eye_u = jnp.eye(ctx.lex.shape[0])[:, :, None]
    other_words_spec = (1.0 - eye_u) * word_spec[None, :, :]
    preemption = jnp.max(other_words_spec, axis=1)

    is_unique = jnp.where(n_referents == 1.0, 1.0, 0.0)
    is_shared = jnp.where(n_referents > 1.0, 1.0, 0.0)
    contrast = jnp.sum(non_sink_lex * is_unique, axis=0) - jnp.sum(
        non_sink_lex * is_shared, axis=0
    )

    prior = softmax_prior(
        params["w_contrast"] * contrast
        + params["w_familiar"] * ctx.familiarization
        + params["w_color"] * (1.0 - ctx.grayscale)
    )
    heard = L_preempt(params["beta"], ctx.lex, preemption, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
