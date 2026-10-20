"""The response process every link shares (docs/auto_open/PLAN.md §3, PI 2026-10-21): the
theory's posterior is tempered (a decision temperature), then mixed with a uniform guess
(a lapse), then mapped to the link's response. Kept apart from `src.rsa_open.links` so a
composed model file can import it cheaply."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def temper(p: jnp.ndarray, temperature) -> jnp.ndarray:
    """p ** temperature, renormalized over the last axis (a Luce choice rule).

    Zeros stay zero, with finite gradients; at temperature 1 it is p (to float rounding).
    """
    positive = p > 0
    log_p = jnp.log(jnp.where(positive, p, 1.0))
    return jax.nn.softmax(jnp.where(positive, temperature * log_p, -jnp.inf), axis=-1)


def with_uniform(p: jnp.ndarray, lapse) -> jnp.ndarray:
    """With probability ``lapse`` a uniform guess over the last axis."""
    return (1.0 - lapse) * p + lapse / p.shape[-1]
