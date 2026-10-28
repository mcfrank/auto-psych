"""Transformational similarity: people judge whether the target can be made
from the examples by a short chain of simple mental operations on numbers
(add or subtract one, add or subtract ten, double, halve, reverse the digits);
the fewer operations it takes to turn the examples into the target, the more
likely they say "yes". No concept space, posterior or size principle.

Each example's transformation distance to the target is the shortest number
of such operations (breadth-first search over 1-100). Similarity to an example
decays exponentially with that distance (rate inferred), the target's support
is its mean similarity to the examples, and "yes" is a logistic function of it.
"""

from collections import deque

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_EXAMPLES = 4


def _neighbours(n):
    out = [n + 1, n - 1, n + 10, n - 10, n * 2]
    if n % 2 == 0:
        out.append(n // 2)
    if n >= 10:
        s = str(n)[::-1].lstrip("0")
        if s:
            out.append(int(s))
    return [m for m in out if 1 <= m <= 100]


def _distances():
    dist = np.zeros((101, 101))
    for start in range(1, 101):
        seen = {start: 0}
        queue = deque([start])
        while queue:
            n = queue.popleft()
            for m in _neighbours(n):
                if m not in seen:
                    seen[m] = seen[n] + 1
                    queue.append(m)
        for m, d in seen.items():
            dist[start, m] = d
    return dist


DIST = _distances()


def compute_features(number_set, target):
    """Transformation distance from each example to the target, with a mask."""
    examples = [int(x) for x in str(number_set).split(",")]
    t = int(target)
    feats = {}
    for i in range(MAX_EXAMPLES):
        if i < len(examples):
            feats[f"tdist_{i}"] = float(DIST[examples[i], t])
            feats[f"tmask_{i}"] = 1.0
        else:
            feats[f"tdist_{i}"] = 0.0
            feats[f"tmask_{i}"] = 0.0
    return feats


with pm.Model() as model:
    dists = pt.stack(
        [pm.Data(f"tdist_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EXAMPLES)], axis=1
    )
    masks = pt.stack(
        [pm.Data(f"tmask_{i}", np.zeros(1, dtype="float64")) for i in range(MAX_EXAMPLES)], axis=1
    )

    # How fast similarity falls with each extra operation needed.
    decay = pm.LogNormal("decay", mu=-0.5, sigma=0.7)
    # Baseline log-odds of "yes" for a target with no similarity, and gain.
    intercept = pm.Normal("intercept", mu=0.0, sigma=1.5)
    gain = pm.HalfNormal("gain", sigma=4.0)

    sim = pt.exp(-decay * dists) * masks
    support = sim.sum(axis=1) / masks.sum(axis=1)
    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(intercept + gain * support), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
