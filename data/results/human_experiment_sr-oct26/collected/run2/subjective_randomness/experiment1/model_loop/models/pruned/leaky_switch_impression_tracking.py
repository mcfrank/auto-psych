"""Leaky running switch-rate impression tracked against a personal ideal.

People read a sequence flip by flip and keep a running impression of how often
the coin switches, updated after each transition by a leaky (exponentially
forgetting) memory that starts at their own ideal switching rate. At every
point of the reading they compare the impression with that ideal; a sequence
looks random to the extent that the impression stays close to the ideal
throughout (mean squared deviation over the reading). Long streaks drag the
impression below the ideal for several flips, so bunched switches look less
random than evenly spread ones. The ideal is person-specific; the forgetting
rate and the sensitivity are shared.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAX_T = 7  # transitions in an 8-flip sequence


def compute_features(sequence_a, sequence_b):
    def switches(seq):
        s = [1.0 if x != y else 0.0 for x, y in zip(seq, seq[1:])]
        mask = [1.0] * len(s)
        pad = MAX_T - len(s)
        return s + [0.0] * pad, mask + [0.0] * pad

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    sa, ma = switches(a)
    sb, mb = switches(b)
    out = {}
    for t in range(MAX_T):
        out[f"sw_a_{t}"] = sa[t]
        out[f"sw_b_{t}"] = sb[t]
        out[f"mk_a_{t}"] = ma[t]
        out[f"mk_b_{t}"] = mb[t]
    return out


N_SLOTS = 400

with pm.Model() as model:
    sw_a = pt.stack([pm.Data(f"sw_a_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    sw_b = pt.stack([pm.Data(f"sw_b_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    mk_a = pt.stack([pm.Data(f"mk_a_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    mk_b = pt.stack([pm.Data(f"mk_b_{t}", np.zeros(1, dtype="float64")) for t in range(MAX_T)], axis=1)
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Population distribution of personal ideal switching rates (logit scale).
    mu_ideal = pm.Normal("mu_ideal", mu=0.5, sigma=1.0)
    sigma_ideal = pm.HalfNormal("sigma_ideal", sigma=1.0)
    z_ideal = pm.Normal("z_ideal", mu=0.0, sigma=1.0, shape=N_SLOTS)
    ideal = pm.Deterministic("ideal", pm.math.sigmoid(mu_ideal + sigma_ideal * z_ideal))

    # Leaky-memory update weight: share of the impression replaced by each new transition.
    leak = pm.Beta("leak", alpha=2.0, beta=2.0)
    # Sensitivity to the mean squared deviation of the running impression from the ideal.
    beta = pm.HalfNormal("beta", sigma=20.0)

    # W[t, k] = leak * (1 - leak)^(t - k) for k <= t: impression deviation after transition t.
    idx = np.arange(MAX_T)
    lag = idx[:, None] - idx[None, :]
    lower = (lag >= 0).astype("float64")
    W = leak * pt.power(1.0 - leak, np.maximum(lag, 0).astype("float64")) * lower

    theta = ideal[participant_id][:, None]

    def tracked_distance(sw, mk):
        dev = (sw - theta) * mk  # padded transitions contribute nothing
        run_dev = pt.dot(dev, W.T)  # running impression minus ideal
        return pt.sum(mk * run_dev ** 2, axis=1) / pt.sum(mk, axis=1)

    score = beta * (tracked_distance(sw_b, mk_b) - tracked_distance(sw_a, mk_a))
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(score))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
