"""Aspect goal question-under-discussion (QUD) pragmatic listener.

Listeners interpret referring expressions by modeling a speaker whose communicative
goal is to convey a specific visual aspect of an object rather than uniquely identify
the object itself. In deciding which aspect to discuss, the speaker prioritizes
distinctive features that set the referent apart from competitors, producing an
utterance that is informative about that selected aspect. A pragmatic listener
inverts this generative process by jointly reasoning about the speaker's communicative
goal and intended referent.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_qud": dist.Normal(0.0, 1.0),
    "w_val": dist.Normal(0.0, 2.0),
    "w_color": dist.Normal(0.0, 2.0),
    "w_base": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L_qud[u: UTT, r: OBJ](
    alpha, lex: ..., goal_weights: ..., aspect_utility: ..., prior: ...
):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(g in UTT, wpp=at(goal_weights, g, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * at(aspect_utility, u, g)),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    ext = jnp.maximum(jnp.sum(ctx.lex, axis=1, keepdims=True), 1.0)
    l0_mat = ctx.lex / ext
    aspect_mat = jnp.matmul(l0_mat, ctx.lex.T)
    aspect_utility = jnp.log(0.02 + 0.98 * aspect_mat)

    n_obj = ctx.lex.shape[1]
    ext_g = jnp.sum(ctx.lex, axis=1)
    dist_g = -jnp.log(ext_g / n_obj)
    goal_weights = ctx.lex * jnp.exp(params["w_qud"] * dist_g)[:, None]

    val_bias = params["w_val"] * ctx.valence * ctx.feature_count
    color_bias = params["w_color"] * (1.0 - ctx.grayscale)
    base_bias = params["w_base"] * ctx.familiarization
    prior = softmax_prior(val_bias + color_bias + base_bias)

    heard = L_qud(
        params["alpha"], ctx.lex, goal_weights, aspect_utility, prior
    )[ctx.utterance]

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
