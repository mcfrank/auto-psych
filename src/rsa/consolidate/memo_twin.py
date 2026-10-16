# --- the memo twin (src/rsa/consolidate/memo_twin.py) ---
# Appended to superset.py to make a memo variant (`src.rsa.consolidate.variants`,
# engine "memo"); not importable on its own (it uses superset.py's names and the
# loader's OBJ and UTT). The speaker and listener are a memo model: a speaker who
# knows the referent, picks a goal (identify it, or, with probability p_aspect,
# name the sparsest match's feature), then a true word; a listener who inverts
# that speaker under the prior. The utilities are computed in JAX as in
# superset.py and passed in shifted by each object's largest (the softmax
# trick: exact, and memo's exp no longer underflows at depth 2). Every variant
# gives superset.py's probabilities (tests/test_rsa_consolidate.py).
from memo import memo

from src.rsa.memo_kit import at, vec

GOALS = (0, 1)


@memo
def LISTENER[u: UTT, r: OBJ](p_aspect, lex: ..., prior: ..., ident: ..., aspect: ...):
    listener: thinks[
        speaker: given(r in OBJ, wpp=vec(prior, r)),
        speaker: chooses(g in GOALS, wpp=(1.0 - p_aspect) * (1.0 - g) + p_aspect * g),
        speaker: chooses(u in UTT, wpp=at(lex, u, r) * exp((1.0 - g) * at(ident, u, r) + g * at(aspect, u, r))),
    ]
    listener: observes [speaker.u] is u
    listener: chooses(r in OBJ, wpp=Pr[speaker.r == r])
    return Pr[listener.r == r]


def _shifted(lex, exponent):
    e = jnp.where(lex > 0, exponent, NO_WORD)
    return jnp.where(lex > 0, e - jnp.max(e, axis=0, keepdims=True), 0.0)


def _choice(p, ctx, on):  # superset.py's, with the memo speaker and listener
    q = {**OFF, **p}
    lex, sink, fc = ctx.lex, ctx.is_sink, ctx.feature_count
    n_obj = lex.shape[1]
    real = lex * (1.0 - sink)[:, None]

    pair = jnp.sum(jnp.abs(lex[:, :, None] - lex[:, None, :]) * (1.0 - sink)[:, None, None], axis=0)
    total = (pair + jnp.abs(ctx.grayscale[:, None] - ctx.grayscale[None, :])
             + jnp.abs(ctx.familiarization[:, None] - ctx.familiarization[None, :]))
    copies = jnp.sum(total == 0, axis=1)

    lo, hi = jnp.min(fc), jnp.max(fc)
    span = hi - lo
    x = (fc - lo) / jnp.maximum(span, 1.0)
    extremity = jnp.where(span > 0, 1.0 - 4.0 * x * (1.0 - x), 0.0)
    surprisal = jnp.sum(real * -jnp.log(jnp.mean(real, axis=1, keepdims=True) + 1e-4), axis=0)
    prior = softmax_prior(
        -q["w_dup"] * jnp.log(copies)
        + q["w_features"] * fc + q["w_extreme"] * extremity
        + q["w_empty"] * (fc == 0)
        + q["w_surprisal"] * surprisal
    )

    sim = jnp.exp(-q["lambda_sim"] * total)
    l0 = lex / (lex @ sim + EPS)

    alt = -q["w_redundancy"] * jnp.log(equivalence_class(lex, sink))
    cost = q["w_interior"] * interior_penalty(real, fc) + q["c_avoid"] * avoidable_ambiguity(real, sink)
    alpha = q["alpha"] / (1.0 + q["w_competition"] * (jnp.maximum(fc, 1.0) - 1.0))
    sparsest = jnp.min(jnp.where(lex > 0, fc[None, :], 999.0), axis=1, keepdims=True)
    aspect = _shifted(lex, alpha[None, :] * -jnp.log(1.0 + jnp.maximum(fc[None, :] - sparsest, 0.0)))

    def listen(utility):
        ident = _shifted(lex, alt[:, None] + alpha[None, :] * (utility - cost))
        return LISTENER(q["p_aspect"], lex, prior, ident, aspect)

    heard_all = listen(jnp.log(l0 + EPS))
    if "depth2" in on:
        heard_all = listen(jnp.log(heard_all + EPS))
    heard = heard_all[ctx.utterance]

    false_of = (1.0 - sink[ctx.utterance]) * (1.0 - lex[ctx.utterance])
    bind = false_of * (fc / jnp.maximum(hi, 1.0)) * prior
    bind_total = jnp.sum(bind)
    w_bind = q["w_binding"] * (bind_total > 0)
    heard = (1.0 - w_bind) * heard + w_bind * bind / jnp.maximum(bind_total, TINY)

    if "attention" in on:
        attend = jax.nn.sigmoid(q["logit_p_attend"] - q["w_load"] * (n_obj - 2.0))
        capture = prior * real[ctx.utterance] + 1e-6
        heard = attend * heard + (1.0 - attend) * capture / jnp.sum(capture)

    return with_lapse(jnp.where(ctx.is_prior > 0, prior, heard), q["lapse"])
