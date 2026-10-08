
"""RSA pragmatic listener (depth 1) reasoning with graded truth values.

Interlocutors treat word meanings as having graded truth values rather than
crisp boolean applicability: an utterance's semantic truth value dilutes as an
object accumulates unmentioned, extraneous features. A literal listener evaluates
candidate referents using these graded truth values alongside common-knowledge
salience, an informative speaker chooses utterances evaluated under this graded
literal listener, and a pragmatic listener inverts the speaker. Prior salience
over objects is common knowledge, and a lapse parameter mixes in uniform guessing.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo
from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_features": dist.Normal(0.0, 1.0),
    "w_familiar": dist.Normal(0.0, 2.0),
    "gamma_extra": dist.Normal(0.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}

@memo
def L0[u: UTT, r: OBJ](lex: ..., prior: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=vec(prior, r) * at(lex, u, r))
    return Pr[listener.r == r]

@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., graded_lex: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r) * exp(alpha * log(L0[u, r](graded_lex, prior) + {EPS})),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]

def choice_probs(params, ctx):
    prior = softmax_prior(
        params["w_features"] * ctx.feature_count + params["w_familiar"] * ctx.familiarization
    )
    extra = jnp.maximum(0.0, ctx.feature_count[None, :] - 1.0)
    gamma = jax.nn.softplus(params["gamma_extra"])
    graded_weight = 1.0 / (1.0 + gamma * extra)
    
    is_word = (1.0 - ctx.is_sink)[:, None]
    graded_lex = jnp.where(is_word > 0, ctx.lex * graded_weight, ctx.lex)
    
    heard = L1(params["alpha"], ctx.lex, graded_lex, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
