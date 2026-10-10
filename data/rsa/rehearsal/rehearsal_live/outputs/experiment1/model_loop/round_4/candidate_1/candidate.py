"""Aspect-goal listener.

Listeners interpret referring expressions by reasoning about a speaker whose
communicative goal addresses an informative aspect of the referent rather than
its exhaustive identity. When hearing an expression, listeners jointly infer what
descriptive aspect the speaker sought to convey and which candidate referent
motivated that goal, expecting speakers to choose distinctive, diagnostic aspects
of an object over non-diagnostic shared properties. On uninformative trials
where no informative word is uttered, listeners expect speakers to target
referents possessing more distinctive communicative aspects.
"""

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, softmax_prior, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(1.0, 1.0),
    "w_goal": dist.Normal(0.0, 2.0),
    "w_prior": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L1[u: UTT, r: OBJ](speaker_p: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(u in UTT, wpp=at(speaker_p, u, r)),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_speaker_probs(ctx, alpha, w_goal):
    # Literal listener L0[u, r]
    ext = jnp.maximum(jnp.sum(ctx.lex, axis=1, keepdims=True), 1.0)
    l0 = ctx.lex / ext  # (N_UTT, N_OBJ)
    aspect_info = l0 @ ctx.lex.T  # (N_UTT, N_UTT): P_L0(has feature q | word u)

    # Specificity of each feature in the context
    spec = 1.0 / ext[:, 0]  # (N_UTT,)

    # Goal probability P(q | r): speaker selects an aspect q of referent r
    # Mask by lex: can only choose an aspect true of r
    goal_logits = jnp.where(ctx.lex > 0, w_goal * spec[:, None], -1e9)
    p_goal_given_r = jax.nn.softmax(goal_logits, axis=0)  # (N_UTT, N_OBJ)

    # Speaker utterance choice P(u | r, q): utility is informativeness about aspect q
    # For each q, utility of utterance u is aspect_info[u, q]
    # Mask by lex[:, r] so speaker only produces true utterances for r
    # aspect_info has shape (N_UTT_u, N_UTT_q)
    # Broadcast to (N_UTT_u, N_UTT_q, N_OBJ_r)
    utt_utility = aspect_info[:, :, None]  # (N_UTT_u, N_UTT_q, 1)
    utt_logits = jnp.where(ctx.lex[:, None, :] > 0, alpha * utt_utility, -1e9)
    p_u_given_r_q = jax.nn.softmax(utt_logits, axis=0)  # (N_UTT_u, N_UTT_q, N_OBJ_r)

    # Marginalize over speaker's aspect goal q:
    # P_S(u | r) = sum_q P(q | r) * P(u | r, q)
    # p_goal_given_r has shape (N_UTT_q, N_OBJ_r)
    speaker_p = jnp.sum(p_u_given_r_q * p_goal_given_r[None, :, :], axis=1)  # (N_UTT, N_OBJ)

    return speaker_p


def compute_prior(ctx, w_prior):
    ext = jnp.maximum(jnp.sum(ctx.lex, axis=1), 1.0)
    spec = 1.0 / ext
    real_words = 1.0 - ctx.is_sink
    distinctiveness = jnp.sum(ctx.lex * (spec * real_words)[:, None], axis=0)
    return softmax_prior(w_prior * distinctiveness)


def choice_probs(params, ctx):
    speaker_p = compute_speaker_probs(ctx, params["alpha"], params["w_goal"])
    prior = compute_prior(ctx, params["w_prior"])
    heard = L1(speaker_p)[ctx.utterance]
    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), params["lapse"])
