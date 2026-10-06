"""Helpers every memo model file may import.

memo parses its own language: an array parameter (declared ``lex: ...``) can
only be read through a function call, never subscripted (``lex[u, r]`` parses
as a query about an agent). These JIT-compiled indexers are that call.

``OBJ`` and ``UTT`` are not here: the loader injects them, sized to the
context shape being evaluated (`src.rsa.model_file`).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

# Added inside log() so a zero probability gives a large finite penalty, not
# -inf and a NaN gradient.
EPS = 1e-10


@jax.jit
def at(matrix, i, j):
    """matrix[i, j] for an array parameter (e.g. the lexicon at [u, r])."""
    return matrix[i, j]


@jax.jit
def vec(vector, i):
    """vector[i] for an array parameter (e.g. a prior over objects at r)."""
    return vector[i]


def with_lapse(p: jnp.ndarray, lapse) -> jnp.ndarray:
    """Mix a choice distribution with a uniform guess over the objects.

    With probability ``lapse`` the participant picks an object at random.
    Without some such noise a model that gives an object probability zero
    (the literal semantics of RSA does, for any object the word is false of)
    makes one participant's choice of it an impossible event.
    """
    return (1.0 - lapse) * p + lapse / p.shape[-1]


def softmax_prior(logits: jnp.ndarray) -> jnp.ndarray:
    """A prior over objects from unnormalised log weights."""
    return jax.nn.softmax(logits)
