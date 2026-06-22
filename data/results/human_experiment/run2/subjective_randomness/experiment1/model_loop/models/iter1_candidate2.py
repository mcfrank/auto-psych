"""
People judge a sequence's randomness by comparing its local pattern frequencies to a subjective expectation: they compute the Kullback-Leibler divergence between the sequence's empirical bigram distribution (HH, HT, TH, TT) and an idealized prototype distribution that heavily favors alternating pairs, perceiving sequences with lower divergence from this prototype as more random.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

def compute_features(sequence_a: str, sequence_b: str) -> dict:
    def get_bigram_dist(seq: str):
        seq = seq.strip().upper()
        if len(seq) < 2:
            return 0.25, 0.25, 0.25, 0.25  # Uniform fallback
        
        transitions = [seq[i:i+2] for i in range(len(seq)-1)]
        from collections import Counter
        counts = Counter(transitions)
        n = len(transitions)
        
        # Laplace smoothing (add 1 to each count) to avoid log(0) in the PyMC graph
        p_HH = (counts.get('HH', 0) + 1.0) / (n + 4.0)
        p_HT = (counts.get('HT', 0) + 1.0) / (n + 4.0)
        p_TH = (counts.get('TH', 0) + 1.0) / (n + 4.0)
        p_TT = (counts.get('TT', 0) + 1.0) / (n + 4.0)
        
        return p_HH, p_HT, p_TH, p_TT

    p_HH_a, p_HT_a, p_TH_a, p_TT_a = get_bigram_dist(sequence_a)
    p_HH_b, p_HT_b, p_TH_b, p_TT_b = get_bigram_dist(sequence_b)
    
    return {
        "p_HH_a": float(p_HH_a), "p_HT_a": float(p_HT_a), "p_TH_a": float(p_TH_a), "p_TT_a": float(p_TT_a),
        "p_HH_b": float(p_HH_b), "p_HT_b": float(p_HT_b), "p_TH_b": float(p_TH_b), "p_TT_b": float(p_TT_b),
    }

with pm.Model() as model:
    # Empirical bigram probabilities for sequence A
    p_HH_a = pm.Data("p_HH_a", np.zeros(1, dtype="float64"))
    p_HT_a = pm.Data("p_HT_a", np.zeros(1, dtype="float64"))
    p_TH_a = pm.Data("p_TH_a", np.zeros(1, dtype="float64"))
    p_TT_a = pm.Data("p_TT_a", np.zeros(1, dtype="float64"))

    # Empirical bigram probabilities for sequence B
    p_HH_b = pm.Data("p_HH_b", np.zeros(1, dtype="float64"))
    p_HT_b = pm.Data("p_HT_b", np.zeros(1, dtype="float64"))
    p_TH_b = pm.Data("p_TH_b", np.zeros(1, dtype="float64"))
    p_TT_b = pm.Data("p_TT_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    # The subjective ideal alternation rate (prior centered loosely around 0.5-0.6)
    theta = pm.Beta("theta", alpha=2.0, beta=2.0)
    
    # Sensitivity to the difference in KL divergences
    tau = pm.HalfNormal("tau", sigma=10.0)

    # Ideal distribution: P(HT) + P(TH) = theta, P(HH) + P(TT) = 1 - theta
    # Constrain to avoid division by zero or log(0)
    theta_safe = pt.clip(theta, 1e-4, 1.0 - 1e-4)
    ideal_HT = theta_safe / 2.0
    ideal_TH = theta_safe / 2.0
    ideal_HH = (1.0 - theta_safe) / 2.0
    ideal_TT = (1.0 - theta_safe) / 2.0
    
    # KL divergence D_KL(Empirical || Ideal) for sequence A
    kl_a = (
        p_HH_a * pt.log(p_HH_a / ideal_HH) +
        p_HT_a * pt.log(p_HT_a / ideal_HT) +
        p_TH_a * pt.log(p_TH_a / ideal_TH) +
        p_TT_a * pt.log(p_TT_a / ideal_TT)
    )

    # KL divergence D_KL(Empirical || Ideal) for sequence B
    kl_b = (
        p_HH_b * pt.log(p_HH_b / ideal_HH) +
        p_HT_b * pt.log(p_HT_b / ideal_HT) +
        p_TH_b * pt.log(p_TH_b / ideal_TH) +
        p_TT_b * pt.log(p_TT_b / ideal_TT)
    )

    # Probability of choosing sequence A.
    # A smaller KL divergence means it is perceived as more random,
    # so if kl_a < kl_b, p_left should be > 0.5.
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(tau * (kl_b - kl_a)))
    
    # Enforce strict bounds for numerical safety
    p_left_safe = pt.clip(p_left, 1e-6, 1.0 - 1e-6)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
