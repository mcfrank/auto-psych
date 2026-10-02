"""Short-pattern evenness: a random coin should produce every short pattern equally often.

People expect heads/tails, each of the four flip pairs and each flip triple to occur
about equally often in a random sequence; a sequence looks non-random to the extent
its overlapping one-, two- and three-flip patterns are unevenly spread (entropy short
of the maximum its length allows). How strongly that unevenness counts is each
person's own trait, and people guess on some trials at a personal lapse rate.
"""

import math
from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_PARTICIPANTS = 400


def _unevenness(seq):
    """Mean normalised entropy deficit of overlapping k-grams, k = 1..3."""
    deficits = []
    n = len(seq)
    for k in (1, 2, 3):
        m = n - k + 1  # number of overlapping k-grams
        max_types = min(2**k, m)
        if max_types < 2:
            continue
        counts = Counter(seq[i:i + k] for i in range(m))
        h = -sum((c / m) * math.log(c / m) for c in counts.values())
        # Most even spread achievable with m windows over max_types types.
        q, r = divmod(m, max_types)
        h_max = -sum(
            (c / m) * math.log(c / m)
            for c in [q + 1] * r + [q] * (max_types - r)
            if c > 0
        )
        deficits.append(1.0 - h / h_max)
    if not deficits:
        raise ValueError(f"sequence too short: {seq!r}")
    return float(np.mean(deficits))


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {"uneven_a": _unevenness(a), "uneven_b": _unevenness(b)}


with pm.Model() as model:
    uneven_a = pm.Data("uneven_a", np.zeros(1, dtype="float64"))
    uneven_b = pm.Data("uneven_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal sensitivity to pattern unevenness (non-centred population).
    mu_w = pm.Normal("mu_w", mu=2.0, sigma=2.0)
    sigma_w = pm.HalfNormal("sigma_w", sigma=2.0)
    z_w = pm.Normal("z_w", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    w = mu_w + sigma_w * z_w

    # Personal lapse (guessing) rate.
    mu_lapse = pm.Normal("mu_lapse", mu=-2.0, sigma=1.0)
    sigma_lapse = pm.HalfNormal("sigma_lapse", sigma=1.0)
    z_lapse = pm.Normal("z_lapse", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    lapse = pm.Deterministic("lapse", pm.math.sigmoid(mu_lapse + sigma_lapse * z_lapse))

    wp = w[participant_id]
    p_engaged = pm.math.sigmoid(wp * (uneven_b - uneven_a))
    lam = lapse[participant_id]
    p_left = pm.Deterministic(
        "p_left", pt.clip(0.5 * lam + (1.0 - lam) * p_engaged, 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
