"""Person-specific prototype local representativeness.

Refinement of ``local_representativeness`` (Kahneman & Tversky 1972): a
sequence looks random when it is locally balanced (multiscale H/T balance) and
irregular (close to a prototype alternation rate, not periodic). The one change:
the prototype alternation rate differs across people (a hierarchical,
per-participant parameter that may lie below or above 0.5), rather than one
shared over-alternating prototype.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

# Capacity of the participant effect vector; ids at or above it raise.
MAX_PARTICIPANTS = 256
LOCAL_WINDOW = 4


def _clean(seq):
    s = str(seq).strip().upper()
    if not s:
        raise ValueError("Sequence must not be empty")
    return s


def _periodicity(seq):
    s = _clean(seq)
    n = len(s)
    if n <= 2:
        return 0.0
    best = 0.5
    for period in range(1, (n // 2) + 1):
        template = s[:period]
        matches = sum(1 for i, c in enumerate(s) if c == template[i % period])
        best = max(best, matches / n)
    return max(0.0, min(1.0, 2.0 * (best - 0.5)))


def _multiscale_imbalance(seq):
    s = _clean(seq)
    n = len(s)
    heads = sum(1 for c in s if c == "H")
    scores = [2.0 * abs(heads / n - 0.5)]
    for window in range(2, min(LOCAL_WINDOW, n - 1) + 1):
        w = []
        for start in range(n - window + 1):
            chunk = s[start : start + window]
            w.append(2.0 * abs(sum(1 for c in chunk if c == "H") / window - 0.5))
        scores.append(sum(w) / len(w))
    return sum(scores) / len(scores)


def _p_alts(seq):
    s = _clean(seq)
    n = len(s)
    return sum(1 for i in range(1, n) if s[i] != s[i - 1]) / (n - 1) if n > 1 else 0.0


def prepare_observed(rows):
    out = {k: [] for k in (
        "imb_a", "imb_b", "alt_a", "alt_b", "per_a", "per_b",
    )}
    pids = []
    chose = []
    for row in rows:
        a, b = row["sequence_a"], row["sequence_b"]
        out["imb_a"].append(_multiscale_imbalance(a))
        out["imb_b"].append(_multiscale_imbalance(b))
        out["alt_a"].append(_p_alts(a))
        out["alt_b"].append(_p_alts(b))
        out["per_a"].append(_periodicity(a))
        out["per_b"].append(_periodicity(b))
        pid = int(row["participant_id"])
        if pid < 0 or pid >= MAX_PARTICIPANTS:
            raise ValueError(
                f"participant_id {pid} outside [0, {MAX_PARTICIPANTS}); raise MAX_PARTICIPANTS"
            )
        pids.append(pid)
        chose.append(int(row.get("chose_left", 0) or 0))
    result = {k: np.asarray(v, dtype="float64") for k, v in out.items()}
    result["participant_id"] = np.asarray(pids, dtype="int64")
    result["chose_left"] = np.asarray(chose, dtype="int64")
    return result


with pm.Model() as model:
    imb_a = pm.Data("imb_a", np.zeros(1, dtype="float64"))
    imb_b = pm.Data("imb_b", np.zeros(1, dtype="float64"))
    alt_a = pm.Data("alt_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_b", np.zeros(1, dtype="float64"))
    per_a = pm.Data("per_a", np.zeros(1, dtype="float64"))
    per_b = pm.Data("per_b", np.zeros(1, dtype="float64"))
    pid = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))

    # Population distribution of prototype alternation rates (logit scale),
    # non-centred per-participant deviations.
    mu_theta = pm.Normal("mu_theta", mu=0.5, sigma=1.0)
    sigma_theta = pm.HalfNormal("sigma_theta", sigma=1.0)
    z_theta = pm.Normal("z_theta", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
    theta_j = 0.05 + 0.9 * pm.math.sigmoid(mu_theta + sigma_theta * z_theta)
    theta = theta_j[pid]

    alt_weight = pm.Beta("alt_weight", alpha=2.0, beta=2.0)
    periodic_share = pm.Beta("periodic_share", alpha=2.0, beta=2.0)
    beta = pm.HalfNormal("beta", sigma=6.0)
    side_bias = pm.Normal("side_bias", mu=0.0, sigma=0.5)

    irr_a = (1.0 - periodic_share) * pt.abs(alt_a - theta) + periodic_share * per_a
    irr_b = (1.0 - periodic_share) * pt.abs(alt_b - theta) + periodic_share * per_b
    score_a = -((1.0 - alt_weight) * imb_a + alt_weight * irr_a)
    score_b = -((1.0 - alt_weight) * imb_b + alt_weight * irr_b)

    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(beta * (score_a - score_b) + side_bias)
    )
    pm.Bernoulli("response", p=p_left, observed=chose_left)
