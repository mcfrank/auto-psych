"""Streak-penalised local representativeness with a person-specific prototype.

Refinement of `person_specific_alternation_prototype`: randomness is judged by
multiscale local H/T balance plus irregularity (distance from each person's own
preferred alternation rate, and a periodic-template penalty). The one added
component: the longest run in a sequence is a salient cue of non-randomness,
penalised beyond what the alternation rate implies. Distance from the preferred
rate is squared (smooth), as in the incumbent.
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


def longest_run_share(seq):
    """(longest run - 1) / (length - 1): 0 for no repeats, 1 for one long run."""
    s = _clean(seq)
    n = len(s)
    if n <= 1:
        return 0.0
    best = cur = 1
    for i in range(1, n):
        cur = cur + 1 if s[i] == s[i - 1] else 1
        best = max(best, cur)
    return (best - 1) / (n - 1)


def compute_features(sequence_a, sequence_b):
    features = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        s = _clean(seq)
        n = len(s)
        alts = sum(1 for i in range(1, n) if s[i] != s[i - 1])
        features[f"p_alts_{suffix}"] = (alts / (n - 1)) if n > 1 else 0.0
        features[f"periodicity_{suffix}"] = periodicity_score(s)
        features[f"multiscale_imbalance_{suffix}"] = multiscale_local_imbalance(s)
        features[f"longest_run_{suffix}"] = longest_run_share(s)
    return features


with pm.Model() as model:
    imb_a = pm.Data("multiscale_imbalance_a", np.zeros(1, dtype="float64"))
    imb_b = pm.Data("multiscale_imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    run_a = pm.Data("longest_run_a", np.zeros(1, dtype="float64"))
    run_b = pm.Data("longest_run_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Population distribution of preferred alternation rates (non-centred).
    mu_theta = pm.Normal("mu_theta", mu=0.5, sigma=1.0)
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
    z_theta = pm.Normal("z_theta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    theta_person = 0.05 + 0.9 * pm.math.sigmoid(mu_theta + sigma_theta * z_theta)
    theta = theta_person[participant_id]

    # Non-negative weights on the cues: imbalance, alternation distance,
    # periodicity, and the added longest-run (streak) penalty.
    w_imb = pm.HalfNormal("w_imb", sigma=3.0)
    w_alt = pm.HalfNormal("w_alt", sigma=10.0)
    w_per = pm.HalfNormal("w_per", sigma=3.0)
    w_run = pm.HalfNormal("w_run", sigma=3.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    score_a = -(
        w_imb * imb_a + w_alt * (p_alts_a - theta) ** 2 + w_per * per_a + w_run * run_a
    )
    score_b = -(
        w_imb * imb_b + w_alt * (p_alts_b - theta) ** 2 + w_per * per_b + w_run * run_b
    )

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score_a - score_b + side_bias))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
