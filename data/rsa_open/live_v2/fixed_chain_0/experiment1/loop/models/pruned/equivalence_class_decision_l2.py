import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_decision": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., confusion: ..., prior: ..., decision: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L0[u, r](lex) + {EPS})
                - at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(decision, u, r))
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., confusion: ..., prior: ..., decision: ...):
    listener: knows(u)
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, confusion, prior, decision) + {EPS})
                - at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r] * at(decision, u, r))
    return Pr[listener.r == r]


def listener(params, ctx):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = jnp.abs(feats[:, :, None] - feats[:, None, :])
    dist_mat = jnp.sum(diff, axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    dup_count = jnp.sum(dist_mat == 0, axis=1)
    is_singleton = jnp.where(dup_count == 1, 1.0, 0.0)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real = jnp.sum(1.0 - ctx.is_sink)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

    is_min_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == min_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    is_inter_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < max_true_fc),
        1.0,
        0.0,
    )
    has_inter_utt = jnp.max(is_inter_utt, axis=1, keepdims=True)
    is_partial_max_utt = jnp.where(
        (real_lex > 0)
        & (fc[None, :] == max_true_fc)
        & (max_true_fc > min_true_fc)
        & (fc[None, :] < n_real),
        1.0,
        0.0,
    )
    is_all_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == n_real) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    has_cloned_min = jnp.max(is_min_utt * (1.0 - is_singleton[None, :]), axis=1, keepdims=True)

    has_false = jnp.where(jnp.sum(real_lex, axis=1, keepdims=True) < n_obj, 1.0, 0.0)
    nonempty_false = (1.0 - real_lex) * jnp.where(fc[None, :] > 0, 1.0, 0.0)
    has_nonempty_false = jnp.max(nonempty_false, axis=1, keepdims=True)

    w = params["w_decision"]

    confusion = w * ((real_lex @ sim) + is_inter_utt)
    baseline_confusion = jnp.sum(sim, axis=1) - ctx.valence * fc - (1.0 - ctx.grayscale) - ctx.familiarization
    word_prior = softmax_prior(-w * baseline_confusion)

    decision_score = (
        is_all_utt * (has_cloned_min + has_false)
        - is_partial_max_utt * has_nonempty_false * (is_singleton[None, :] + has_inter_utt)
    ) / n_obj
    decision = jnp.where(
        ctx.is_sink[:, None] > 0,
        1.0,
        jnp.exp(w * decision_score),
    )

    prior_choice = softmax_prior(
        -w * (baseline_confusion + is_inter_scene - (is_max_fc * is_singleton) / (n_obj - 1.0))
    )

    heard = L2(params["alpha"], ctx.lex, confusion, word_prior, decision)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
