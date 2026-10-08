"""Non-Bayesian associative cue competition listener based on Rescorla-Wagner learning."""

import jax.numpy as jnp
import numpyro.distributions as dist
from memo import memo

from src.rsa.memo_kit import at, with_lapse

PARAMS = {
    "beta": dist.LogNormal(1.0, 1.0),
    "lapse": dist.Beta(1.0, 9.0),
}


def compute_associative_weights(lex: jnp.ndarray, is_sink: jnp.ndarray) -> jnp.ndarray:
    """Compute asymptotic Rescorla-Wagner associative weights from cues to objects."""
    real_features = lex * (1.0 - is_sink)[:, None]  # (N_UTT, N_OBJ)
    X = real_features.T  # (N_OBJ, N_UTT)
    n_utt = lex.shape[0]
    C = X.T @ X + jnp.eye(n_utt)
    V = jnp.linalg.solve(C, X.T)  # (N_UTT, N_OBJ)
    return jnp.where(is_sink[:, None] > 0, 0.0, V)


@memo
def H[u: UTT, r: OBJ](beta, lex: ..., V: ...):
    listener: knows(u)
    listener: chooses(
        r in OBJ,
        wpp=at(lex, u, r) * exp(beta * at(V, u, r)),
    )
    return Pr[listener.r == r]


def choice_probs(params, ctx):
    V = compute_associative_weights(ctx.lex, ctx.is_sink)
    heard = H(params["beta"], ctx.lex, V)[ctx.utterance]
    uniform = jnp.full_like(heard, 1.0 / heard.shape[0])
    return with_lapse(jnp.where(ctx.is_prior > 0, uniform, heard), params["lapse"])
