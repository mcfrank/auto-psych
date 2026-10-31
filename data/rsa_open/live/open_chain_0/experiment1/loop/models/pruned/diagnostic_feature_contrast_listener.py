import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import EPS, at, softmax_prior, vec

PARAMS = {
    "beta_truth": dist.LogNormal(1.0, 0.5),
    "w_diag": dist.Normal(0.0, 1.0),
    "w_unmentioned": dist.Normal(0.0, 1.0),
    "w_singleton": dist.Normal(0.0, 1.0),
    "w_valence": dist.Normal(0.0, 1.0),
}


@memo
def L_diag[u: UTT, r: OBJ](beta_truth, w_unmentioned, lex: ..., collat: ..., base_sal: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=exp(
            beta_truth * at(lex, u, r)
            - w_unmentioned * at(collat, u, r)
            + vec(base_sal, r)
        ),
    )
    return Pr[listener.r == r]


def listener(params, ctx):
    real_mask = (1.0 - ctx.is_sink)[:, None]
    real_lex = ctx.lex * real_mask
    n_obj = ctx.lex.shape[1]

    # Step 1: First-order feature selectivity (fraction of objects NOT having the feature)
    feat_ext = jnp.sum(real_lex, axis=1, keepdims=True)
    sel_1 = jnp.maximum(0.0, 1.0 - feat_ext / n_obj) * real_mask

    # Step 2: Second-order feature specificity (a feature is more diagnostic when its bearers
    # lack other selective features that could individuate them)
    obj_sel_1 = jnp.sum(real_lex * sel_1, axis=0, keepdims=True)
    other_obj_sel = obj_sel_1 - real_lex * sel_1
    mean_bearer_other_sel = jnp.sum(real_lex * other_obj_sel, axis=1, keepdims=True) / jnp.maximum(feat_ext, 1.0)
    feat_diag = (sel_1 + 0.5 * sel_1 * (1.0 - mean_bearer_other_sel)) * real_mask

    # Object diagnosticity score
    diag_score = jnp.sum(real_lex * feat_diag, axis=0)

    # Singleton detection
    diffs = jnp.abs(real_lex[:, :, None] - real_lex[:, None, :])
    pair_dist = jnp.sum(diffs, axis=0)
    gray_diff = jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
    fam_diff = jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :])
    total_dist = pair_dist + gray_diff + fam_diff
    is_singleton = jnp.where(jnp.sum(total_dist == 0, axis=1) == 1, 1.0, 0.0)

    valence_shift = ctx.valence * params["w_valence"] * diag_score
    base_sal = params["w_singleton"] * is_singleton + valence_shift
    prior = softmax_prior(params["w_diag"] * jnp.log1p(diag_score) + base_sal)

    # Sublinear unmentioned diagnostic feature penalty: log(1 + unmentioned diagnosticity)
    raw_collat = jnp.maximum(0.0, diag_score[None, :] - real_lex * feat_diag)
    collat = jnp.log1p(raw_collat)

    heard = L_diag(
        params["beta_truth"], params["w_unmentioned"], ctx.lex, collat, base_sal
    )[ctx.utterance]
    return jnp.where(ctx.is_prior > 0, prior, heard)
