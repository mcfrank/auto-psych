import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "alpha": dist.LogNormal(0.0, 0.5),
    "w_confusion": dist.LogNormal(0.0, 0.5),
}


@memo
def L0[u: UTT, r: OBJ](lex: ...):
    listener: knows(u)
    listener: chooses(r in OBJ, wpp=at(lex, u, r))
    return Pr[listener.r == r]


@memo
def L1[u: UTT, r: OBJ](alpha, lex: ..., confusion: ..., prior: ...):
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
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


@memo
def L2[u: UTT, r: OBJ](alpha, lex: ..., confusion: ..., prior: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(
            u in UTT,
            wpp=at(lex, u, r)
            * exp(
                alpha * log(L1[u, r](alpha, lex, confusion, prior) + {EPS})
                - at(confusion, u, r)
            ),
        ),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def compute_confusion_and_prior(ctx, w_confusion):
    feats = jnp.vstack([ctx.lex, ctx.grayscale[None, :], ctx.familiarization[None, :]])
    diff = feats[:, :, None] - feats[:, None, :]
    dist_mat = jnp.sum(jnp.abs(diff), axis=0)
    n_obj = dist_mat.shape[0]
    eye = jnp.eye(n_obj)
    sim = jnp.exp(-dist_mat) * (1.0 - eye)

    real_words = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_words
    n_real_utt = jnp.sum(1.0 - ctx.is_sink)

    fc = ctx.feature_count
    min_fc = jnp.min(fc)
    max_fc = jnp.max(fc)
    is_inter_scene = jnp.where((fc > min_fc) & (fc < max_fc), 1.0, 0.0)
    is_max_fc = jnp.where((fc == max_fc) & (max_fc > min_fc), 1.0, 0.0)
    max_fc_count = jnp.sum(is_max_fc)
    is_unique_max_fc = jnp.where((is_max_fc > 0) & (max_fc_count == 1.0), 1.0, 0.0)

    masked_min_fc = jnp.where(real_lex > 0, fc[None, :], max_fc + 1.0)
    min_true_fc = jnp.min(masked_min_fc, axis=1, keepdims=True)
    masked_max_fc = jnp.where(real_lex > 0, fc[None, :], 0.0)
    max_true_fc = jnp.max(masked_max_fc, axis=1, keepdims=True)

    true_count = jnp.sum(real_lex, axis=1, keepdims=True)
    has_false_utt = jnp.where(true_count < n_obj, 1.0, 0.0)
    is_min_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] == min_true_fc) & (max_true_fc > min_true_fc),
        1.0,
        0.0,
    )
    min_utt_count = jnp.sum(is_min_utt, axis=1, keepdims=True)
    is_unique_min_utt = jnp.where(
        (is_min_utt > 0) & (min_utt_count == 1.0) & (has_false_utt > 0),
        1.0,
        0.0,
    )

    is_nonmin_nonfull_utt = jnp.where(
        (real_lex > 0) & (fc[None, :] > min_true_fc) & (fc[None, :] < n_real_utt),
        1.0,
        0.0,
    )

    confusion = w_confusion * (
        (real_lex @ sim) + is_nonmin_nonfull_utt - is_unique_min_utt / n_obj
    )
    baseline_confusion = (
        jnp.sum(sim, axis=1) - ctx.valence * fc - is_unique_max_fc / n_obj
    )
    word_prior = softmax_prior(-w_confusion * baseline_confusion)
    prior_choice = softmax_prior(
        -w_confusion * (baseline_confusion + is_inter_scene - is_max_fc / n_obj)
    )
    return confusion, word_prior, prior_choice


def listener(params, ctx):
    confusion, word_prior, prior_choice = compute_confusion_and_prior(
        ctx, params["w_confusion"]
    )
    heard = L2(params["alpha"], ctx.lex, confusion, word_prior)[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior_choice, heard)
