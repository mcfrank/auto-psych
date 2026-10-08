"""People judge randomness by checking whether every short pattern turns up about
equally often: they tally single flips, overlapping pairs and overlapping triples
in a sequence, and a sequence looks random to the extent these tallies are evenly
spread (normalised block entropy), with shorter blocks weighing more than longer
ones by a fitted decay."""
import math
from collections import Counter

import numpy as np
import pymc as pm

K_MAX = 3


def _norm_block_entropy(seq, k):
    n = len(seq)
    m = n - k + 1
    cap = min(2 ** k, m)
    if m <= 0 or cap <= 1:
        return 1.0
    counts = Counter(seq[i:i + k] for i in range(m))
    h = -sum((c / m) * math.log(c / m) for c in counts.values())
    return h / math.log(cap)


def compute_features(sequence_a, sequence_b):
    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    out = {}
    for k in range(1, K_MAX + 1):
        out[f"ent_diff_{k}"] = _norm_block_entropy(a, k) - _norm_block_entropy(b, k)
    return out


with pm.Model() as model:
    d1 = pm.Data("ent_diff_1", np.zeros(1, dtype="float64"))
    d2 = pm.Data("ent_diff_2", np.zeros(1, dtype="float64"))
    d3 = pm.Data("ent_diff_3", np.zeros(1, dtype="float64"))

    # sensitivity to the (weighted) evenness difference
    beta = pm.HalfNormal("beta", sigma=5.0)
    # geometric decay of weight with block size
    decay = pm.Beta("decay", alpha=2.0, beta=2.0)

    w2 = decay
    w3 = decay ** 2
    evidence = (d1 + w2 * d2 + w3 * d3) / (1.0 + w2 + w3)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * evidence))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
