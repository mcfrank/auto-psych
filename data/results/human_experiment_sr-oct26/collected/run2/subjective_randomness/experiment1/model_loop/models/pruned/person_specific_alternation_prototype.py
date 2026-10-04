"""Person-specific alternation prototype in local representativeness.

Refinement of `local_representativeness` (Kahneman & Tversky 1972): randomness
is judged by multiscale local H/T balance plus irregularity (distance from a
preferred alternation rate, and a periodic-template penalty). The one change:
each participant has their own preferred alternation rate, drawn from a
population distribution, instead of a single shared prototype.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

N_SLOTS = 400
LOCAL_WINDOW = 4


def _clean(seq):
    s = seq.strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def periodicity_score(seq):
    s = _clean(seq)
    n = len(s)
    if n <= 2:
        return 0.0
    best_match = 0.5
    for period in range(1, (n // 2) + 1):
        template = s[:period]
        matches = sum(1 for i, c in enumerate(s) if c == template[i % period])
        best_match = max(best_match, matches / n)
    return max(0.0, min(1.0, 2.0 * (best_match - 0.5)))


def multiscale_local_imbalance(seq):
    s = _clean(seq)
    n = len(s)
    heads = sum(1 for c in s if c == "H")
    scale_scores = [2.0 * abs(heads / n - 0.5)]
    for window in range(2, min(LOCAL_WINDOW, n - 1) + 1):
        ws = []
        for start in range(n - window + 1):
            chunk = s[start : start + window]
            ws.append(2.0 * abs(sum(1 for c in chunk if c == "H") / window - 0.5))
        scale_scores.append(sum(ws) / len(ws))
    return sum(scale_scores) / len(scale_scores)


def compute_features(sequence_a, sequence_b):
    features = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        s = _clean(seq)
        n = len(s)
        alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
        features[f"p_alts_{suffix}"] = (alts / (n - 1)) if n > 1 else 0.0
        features[f"periodicity_{suffix}"] = periodicity_score(s)
        features[f"multiscale_imbalance_{suffix}"] = multiscale_local_imbalance(s)
    return features


with pm.Model() as model:
    imb_a = pm.Data("multiscale_imbalance_a", np.zeros(1, dtype="float64"))
    imb_b = pm.Data("multiscale_imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Population distribution of preferred alternation rates (logit scale,
    # mapped to (0.05, 0.95)); non-centred per-person offsets.
    mu_theta = pm.Normal("mu_theta", mu=0.5, sigma=1.0)
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
    z_theta = pm.Normal("z_theta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    theta_person = 0.05 + 0.9 * pm.math.sigmoid(mu_theta + sigma_theta * z_theta)
    theta = theta_person[participant_id]

    alt_weight = pm.Beta("alt_weight", alpha=2.0, beta=2.0)
    periodic_share = pm.Beta("periodic_share", alpha=2.0, beta=2.0)
    beta = pm.HalfNormal("beta", sigma=6.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    irr_a = (1.0 - periodic_share) * pt.abs(p_alts_a - theta) + periodic_share * per_a
    irr_b = (1.0 - periodic_share) * pt.abs(p_alts_b - theta) + periodic_share * per_b
    score_a = -((1.0 - alt_weight) * imb_a + alt_weight * irr_a)
    score_b = -((1.0 - alt_weight) * imb_b + alt_weight * irr_b)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
