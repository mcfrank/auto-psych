"""Distractor discrimination listener penalizing candidate referents that share extraneous features with non-matching distractors.

When interpreting a referring expression, listeners evaluate candidate referents based
on their discriminability from the non-matching distractors in the visual scene.
Extraneous unmentioned features that are also possessed by distractors create perceptual
confusability and are penalized, whereas candidate referents whose extraneous features
do not overlap with distractors stand out as distinctive and are preferred. On a display
where two matching objects both have two features, but one shares its extraneous feature
with a distractor while the other has an unshared feature, the current best model assigns
them equal probability, whereas this model predicts listeners prefer the object that
contrasts cleanly with the distractor.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, vec, with_lapse

PARAMS = {
    "lambda_distractor": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_distractor[u: UTT, r: OBJ](lambda_distractor, lex: ..., overlap: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(-lambda_distractor * vec(overlap, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    # Distractors are objects that do not literally satisfy the heard utterance
    is_distractor = 1.0 - ctx.lex[ctx.utterance]
    n_distractors = jnp.maximum(1.0, jnp.sum(is_distractor))

    # Mask out the uttered feature and the sink utterance
    alt_feats = ctx.lex.at[ctx.utterance].set(0.0) * (1.0 - ctx.is_sink[:, None])

    # For each alternative feature, count how many distractor objects have it
    distractor_feat_count = jnp.sum(alt_feats * is_distractor[None, :], axis=1)

    # For each candidate referent, compute total overlap with distractors, normalized by distractor count
    overlap = jnp.sum(alt_feats * distractor_feat_count[:, None], axis=0) / n_distractors

    heard = L_distractor(params["lambda_distractor"], ctx.lex, overlap)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
