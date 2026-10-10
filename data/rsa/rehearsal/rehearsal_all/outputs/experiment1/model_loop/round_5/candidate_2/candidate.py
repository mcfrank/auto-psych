"""Pragmatic listener reasoning about a competitor-nameability-sensitive speaker.

Speakers in reference games evaluate referring expressions by the alternative
nameability of competing display items, avoiding descriptions shared with competitors
that lack distinctive labels of their own. When a distractor cannot be uniquely named,
it relies entirely on shared descriptions and heavily competes for the heard word;
in contrast, distractors that possess uniquely identifying alternatives are preempted
from using the shared expression. Pragmatic listeners invert this competitor-nameability-
aware speaker to resolve referential ambiguity, expecting ambiguous descriptions to
identify referents whose competitors could not be uniquely named.
"""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, with_lapse

PARAMS = {
    "alpha": dist.LogNormal(0.0, 1.0),
    "w_nameability": dist.Normal(0.0, 2.0),
    "lapse": dist.Beta(1.0, 9.0),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, w_nameability, lex: ..., comp_unname: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=1),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - w_nameability * at(comp_unname, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_comp_unnameability(ctx):
    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    utt_counts = jnp.sum(real_lex, axis=1, keepdims=True)
    L0_mat = real_lex / jnp.maximum(utt_counts, 1.0)
    best_L0 = jnp.max(L0_mat, axis=0)
    unnameability = 1.0 - best_L0
    total_unname_u = jnp.sum(
        real_lex * unnameability[None, :], axis=1, keepdims=True
    )
    comp_unname = (
        total_unname_u - real_lex * unnameability[None, :]
    ) * real_words
    n_obj = ctx.lex.shape[1]
    norm_comp_unname = comp_unname / jnp.maximum(n_obj - 1.0, 1.0)
    return norm_comp_unname


def choice_probs(params, ctx):
    comp_unname = compute_comp_unnameability(ctx)
    heard = L1(params["alpha"], params["w_nameability"], ctx.lex, comp_unname)[
        ctx.utterance
    ]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(
        jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"]
    )
