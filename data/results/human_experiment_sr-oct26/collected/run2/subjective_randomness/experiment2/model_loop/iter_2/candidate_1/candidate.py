"""Run-length profile typicality.

People judge randomness by the mix of streak lengths a sequence breaks into:
each person expects a random coin to produce a characteristic spread of runs
(mostly singles, fewer pairs, rarer triples, occasional longer streaks), and a
sequence looks random to the extent that the proportions of its runs of length
1, 2, 3 and 4+ match that personal expected mix. People differ both in the
mix they expect (their felt chance that a run continues) and in how strongly a
mismatch drives their choice.
"""
import numpy as np
import pymc as pm

N_SLOTS = 400


def _run_profile(seq):
    seq = seq.strip().upper()
    runs = []
    cur = 1
    for x, y in zip(seq, seq[1:]):
        if x == y:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    counts = [0.0, 0.0, 0.0, 0.0]
    for r in runs:
        counts[min(r, 4) - 1] += 1.0
    n = float(len(runs))
    return [c / n for c in counts]


def compute_features(sequence_a, sequence_b):
    pa = _run_profile(sequence_a)
    pb = _run_profile(sequence_b)
    out = {}
    for k in range(4):
        out[f"runprop{k + 1}_a"] = pa[k]
        out[f"runprop{k + 1}_b"] = pb[k]
    return out


with pm.Model() as model:
    props_a = [pm.Data(f"runprop{k + 1}_a", np.zeros(1, dtype="float64")) for k in range(4)]
    props_b = [pm.Data(f"runprop{k + 1}_b", np.zeros(1, dtype="float64")) for k in range(4)]
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Personal felt probability that a run continues (expected run-length mix).
    mu_s = pm.Normal("mu_s", mu=-0.5, sigma=1.0)
    sd_s = pm.HalfNormal("sd_s", sigma=1.0)
    z_s = pm.Normal("z_s", mu=0.0, sigma=1.0, shape=N_SLOTS)
    s = pm.math.sigmoid(mu_s + sd_s * z_s)[participant_id]

    # Personal sensitivity to a mismatch in the run-length profile.
    mu_b = pm.Normal("mu_b", mu=1.5, sigma=1.0)
    sd_b = pm.HalfNormal("sd_b", sigma=0.7)
    z_b = pm.Normal("z_b", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.math.exp(mu_b + sd_b * z_b)[participant_id]

    # Expected proportions of runs of length 1, 2, 3, 4+ under geometric runs.
    expected = [1.0 - s, (1.0 - s) * s, (1.0 - s) * s**2, s**3]

    dist_a = sum((props_a[k] - expected[k]) ** 2 for k in range(4))
    dist_b = sum((props_b[k] - expected[k]) ** 2 for k in range(4))

    p_left = pm.Deterministic("p_left", pm.math.sigmoid(beta * (dist_b - dist_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
