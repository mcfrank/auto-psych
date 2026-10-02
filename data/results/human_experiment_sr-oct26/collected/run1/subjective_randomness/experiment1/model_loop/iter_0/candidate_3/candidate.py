"""Local representativeness with an individual prototype alternation rate.

Refinement of `local_representativeness` (Kahneman & Tversky 1972): a sequence
looks random when it is locally balanced (multiscale H/T imbalance) and
irregular (close to a prototype alternation rate, not periodic). The single
change: each participant has their own prototype alternation rate, drawn from
a population distribution (non-centred hierarchy), instead of one shared rate.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Upper bound on participant ids (ids are unique across a run's experiments);
# slots of unseen participants stay at the population prior.
MAX_PARTICIPANTS = 512

with pm.Model() as model:
    multiscale_imbalance_a = pm.Data("multiscale_imbalance_a", np.zeros(1, dtype="float64"))
    multiscale_imbalance_b = pm.Data("multiscale_imbalance_b", np.zeros(1, dtype="float64"))
    p_alts_a = pm.Data("p_alts_a", np.zeros(1, dtype="float64"))
    p_alts_b = pm.Data("p_alts_b", np.zeros(1, dtype="float64"))
    periodicity_a = pm.Data("periodicity_a", np.zeros(1, dtype="float64"))
    periodicity_b = pm.Data("periodicity_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Population distribution of the prototype alternation rate (logit scale,
    # mapped into (0.05, 0.95)); participants deviate non-centrally.
    theta_mu = pm.Normal("theta_mu", mu=0.8, sigma=1.0)
    theta_sigma = pm.HalfNormal("theta_sigma", sigma=1.0)
    theta_z = pm.Normal("theta_z", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    theta_person = 0.05 + 0.9 * pm.math.sigmoid(theta_mu + theta_sigma * theta_z)
    theta_trial = theta_person[participant_id]

    alt_weight = pm.Beta("alt_weight", alpha=2.0, beta=2.0)
    periodic_share = pm.Beta("periodic_share", alpha=2.0, beta=2.0)
    beta = pm.LogNormal("beta", mu=np.log(4.0), sigma=0.75)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    balance_weight = 1.0 - alt_weight
    irregularity_a = (1.0 - periodic_share) * pt.abs(p_alts_a - theta_trial) + periodic_share * periodicity_a
    irregularity_b = (1.0 - periodic_share) * pt.abs(p_alts_b - theta_trial) + periodic_share * periodicity_b
    score_a = -(balance_weight * multiscale_imbalance_a + alt_weight * irregularity_a)
    score_b = -(balance_weight * multiscale_imbalance_b + alt_weight * irregularity_b)

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias))
    pm.Bernoulli("response", p=p_left, observed=chose_left)


LOCAL_WINDOW = 4


def clean_sequence(seq):
    s = seq.strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def periodicity_score(seq):
    s = clean_sequence(seq)
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
    s = clean_sequence(seq)
    n = len(s)
    heads = sum(1 for c in s if c == "H")
    scale_scores = [2.0 * abs(heads / n - 0.5)]
    for window in range(2, min(LOCAL_WINDOW, n - 1) + 1):
        window_scores = []
        for start in range(n - window + 1):
            chunk = s[start : start + window]
            chunk_heads = sum(1 for c in chunk if c == "H")
            window_scores.append(2.0 * abs(chunk_heads / window - 0.5))
        scale_scores.append(sum(window_scores) / len(window_scores))
    return sum(scale_scores) / len(scale_scores)


def compute_features(sequence_a, sequence_b):
    features = {}
    for seq, suffix in ((sequence_a, "a"), (sequence_b, "b")):
        cleaned = clean_sequence(seq)
        n = len(cleaned)
        alts = sum(1 for i in range(1, n) if cleaned[i] != cleaned[i - 1])
        features[f"p_alts_{suffix}"] = (alts / (n - 1)) if n > 1 else 0.0
        features[f"periodicity_{suffix}"] = periodicity_score(seq)
        features[f"multiscale_imbalance_{suffix}"] = multiscale_local_imbalance(seq)
    return features
