"""Pragmatic listener reasoning about a distractor-elimination speaker.

Speakers in reference games evaluate candidate referring expressions by their
distractor elimination power—the fraction of contextual distractors in the visual
scene that each feature rules out—rather than naive probability matching. Words
that rule out all distractors have maximal utility, whereas words shared by the
entire display rule out zero distractors and are disfavored. Pragmatic listeners
invert this distractor-eliminating speaker. On uninformative prior trials,
choices reflect the contextual distinctiveness of the objects.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_distinct": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., elim: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(elim, u, r)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_elim_and_prior(ctx, w_distinct):
    n_obj = ctx.lex.shape[1]
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    ext = jnp.sum(real_lex, axis=1, keepdims=True)
    
    # Fraction of distractors ruled out by each feature
    denom = jnp.maximum(n_obj - 1.0, 1.0)
    elim_feat = (n_obj - ext) / denom
    elim_feat = jnp.clip(elim_feat, 0.0, 1.0) * real_words
    
    # Broadcast to (N_UTT, N_OBJ)
    elim_matrix = jnp.broadcast_to(elim_feat, ctx.lex.shape)
    
    # Object distinctiveness: sum of diagnosticity of its true features
    obj_diag = jnp.sum(real_lex * elim_matrix, axis=0)
    prior = softmax_prior(w_distinct * obj_diag)
    
    return elim_matrix, prior


def choice_probs(params, ctx):
    elim_matrix, prior = compute_elim_and_prior(ctx, params["w_distinct"])
    heard = L1(params["alpha"], ctx.lex, elim_matrix, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
