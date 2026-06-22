"""
People judge the randomness of a sequence by computing its likelihood under a Gambler's Fallacy mental model, expecting that the probability of a coin switching states increases the longer it produces the same outcome. They penalize sequences containing long runs because each successive identical flip violates a progressively stronger subjective expectation of an alternation.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Compute counts of switches and stays after runs of length k."""

    def get_run_transitions(seq: str, max_k: int = 10):
        counts = {f"switch_{k}": 0.0 for k in range(1, max_k + 1)}
        counts.update({f"stay_{k}": 0.0 for k in range(1, max_k + 1)})

        if not seq:
            return counts

        seq = seq.strip().upper()

        current_run = 1
        for i in range(1, len(seq)):
            prev = seq[i - 1]
            curr = seq[i]
            k = min(current_run, max_k)
            if curr != prev:
                counts[f"switch_{k}"] += 1.0
                current_run = 1
            else:
                counts[f"stay_{k}"] += 1.0
                current_run += 1
        return counts

    max_k = 10
    trans_a = get_run_transitions(sequence_a, max_k)
    trans_b = get_run_transitions(sequence_b, max_k)

    features = {}
    for k, v in trans_a.items():
        features[f"{k}_a"] = v
    for k, v in trans_b.items():
        features[f"{k}_b"] = v

    return features


with pm.Model() as model:
    max_k = 10

    switches_a = []
    stays_a = []
    switches_b = []
    stays_b = []

    for k in range(1, max_k + 1):
        switches_a.append(pm.Data(f"switch_{k}_a", np.zeros(1, dtype="float64")))
        stays_a.append(pm.Data(f"stay_{k}_a", np.zeros(1, dtype="float64")))
        switches_b.append(pm.Data(f"switch_{k}_b", np.zeros(1, dtype="float64")))
        stays_b.append(pm.Data(f"stay_{k}_b", np.zeros(1, dtype="float64")))

    sw_a = pt.stack(switches_a)
    st_a = pt.stack(stays_a)
    sw_b = pt.stack(switches_b)
    st_b = pt.stack(stays_b)

    # Cognitive parameters
    # alpha: base logit of switching after 1 flip
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    # beta: how much logit(switch) increases per additional flip in the run
    beta = pm.Normal("beta", mu=0.0, sigma=1.0)
    tau = pm.HalfNormal("tau", sigma=2.0)

    # k_vals: [0, 1, 2, ..., max_k - 1]
    k_vals = np.arange(max_k, dtype="float64")
    # reshape for broadcasting: (max_k, 1)
    k_vec = pt.as_tensor_variable(k_vals[:, None])

    logits = alpha + beta * k_vec
    p_switch = pm.math.sigmoid(logits)
    p_switch = pt.clip(p_switch, 1e-5, 1 - 1e-5)

    log_p_switch = pt.log(p_switch)
    log_p_stay = pt.log(1 - p_switch)

    log_p_a = pt.sum(sw_a * log_p_switch + st_a * log_p_stay, axis=0)
    log_p_b = pt.sum(sw_b * log_p_switch + st_b * log_p_stay, axis=0)

    p_raw = pm.math.sigmoid(tau * (log_p_a - log_p_b))
    p_left = pm.Deterministic("p_left", pt.clip(p_raw, 1e-6, 1 - 1e-6))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
