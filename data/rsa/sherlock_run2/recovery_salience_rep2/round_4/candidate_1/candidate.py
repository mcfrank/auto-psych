
"""RSA listener reasoning about the speaker's communicative goal / QUD.

Instead of assuming the speaker aims to specify the referent's complete identity,
the listener assumes the speaker formulates an utterance to address a communicative
goal concerning an aspect (feature) of the referent. The speaker selects an aspect
of the referent to highlight—favoring distinctive visual features that are uncommon
in the visual scene—and chooses words that informatively convey that intended aspect
to a literal listener. The listener inverts this generative process by jointly inferring
the speaker's communicative goal and intended referent.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "goal_penalty": dist.HalfNormal(2.0),
    "base_rate_weight": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](speaker_mat: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(u in UTT, wpp=at(speaker_mat, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_speaker_mat(alpha, goal_penalty, lex, l0, is_sink):
    # p_qud[u, q] = sum_r l0[u, r] * lex[q, r]
    p_qud = jnp.matmul(l0, lex.T)  # (N_UTT, N_UTT) [u, q]

    # Feature extension (excluding sink)
    real_lex = lex * (1.0 - is_sink)[:, None]
    extension = jnp.sum(real_lex, axis=1)  # (N_UTT,)
    ambiguity = jnp.maximum(0.0, extension - 1.0)

    # Goal weight for (q, r): non-zero only when lex[q, r] == 1
    q_weight = lex * jnp.exp(-goal_penalty * ambiguity)[:, None]

    # Safe log(p_qud): when p_qud > 0 compute log, else 0.0 with 0 gradient
    safe_p = jnp.where(p_qud > 0.0, p_qud, 1.0)
    log_p = jnp.where(p_qud > 0.0, jnp.log(safe_p), 0.0)
    u_util = jnp.where(p_qud > 0.0, jnp.exp(alpha * log_p), 0.0)  # (N_UTT, N_UTT) [u, q]

    # Joint weight W(u, q, r) = lex[u, r] * q_weight[q, r] * u_util[u, q]
    w_joint = lex[:, :, None] * q_weight[None, :, :].swapaxes(1, 2) * u_util[:, None, :]  # [u, r, q]

    # Marginalize over goals q:
    w_u_r = jnp.sum(w_joint, axis=2)  # [u, r]

    # Normalize over utterances u:
    s_mat = w_u_r / jnp.maximum(EPS, jnp.sum(w_u_r, axis=0, keepdims=True))
    return s_mat


def choice_probs(params, ctx):
    l0 = L0(ctx.lex)
    speaker_mat = compute_speaker_mat(
        params["alpha"], params["goal_penalty"], ctx.lex, l0, ctx.is_sink
    )
    prior = softmax_prior(params["base_rate_weight"] * ctx.familiarization)
    heard = L1(speaker_mat, prior)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
