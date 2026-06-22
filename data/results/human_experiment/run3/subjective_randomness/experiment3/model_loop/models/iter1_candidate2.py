"""
People judge the randomness of a sequence by computing its Bayesian diagnosticity—the log-likelihood ratio of a fair coin versus a subjective "regular" generative process. We model this regular hypothesis as a two-state Markov chain with symmetric Beta priors on its transition probabilities, an alternative that effortlessly assigns high marginal probability to sequences with glaring anomalies like over-long streaks, global imbalance, or artificial perfect alternation, thereby heavily penalizing them as non-random without blending multiple ad-hoc heuristics.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a: str, sequence_b: str) -> dict:
    """Return the counts of HH, HT, TH, TT transitions for each sequence."""

    def get_counts(s):
        s = s.strip().upper()
        counts = {"HH": 0, "HT": 0, "TH": 0, "TT": 0}
        for i in range(len(s) - 1):
            pair = s[i : i + 2]
            if pair in counts:
                counts[pair] += 1
        return counts

    ca = get_counts(sequence_a)
    cb = get_counts(sequence_b)

    return {
        "n_HH_a": ca["HH"],
        "n_HT_a": ca["HT"],
        "n_TH_a": ca["TH"],
        "n_TT_a": ca["TT"],
        "n_HH_b": cb["HH"],
        "n_HT_b": cb["HT"],
        "n_TH_b": cb["TH"],
        "n_TT_b": cb["TT"],
    }


with pm.Model() as model:
    # Stimulus inputs
    n_a = pm.Data("n_a", np.zeros(1, dtype="int64"))
    n_b = pm.Data("n_b", np.zeros(1, dtype="int64"))

    n_HH_a = pm.Data("n_HH_a", np.zeros(1, dtype="int64"))
    n_HT_a = pm.Data("n_HT_a", np.zeros(1, dtype="int64"))
    n_TH_a = pm.Data("n_TH_a", np.zeros(1, dtype="int64"))
    n_TT_a = pm.Data("n_TT_a", np.zeros(1, dtype="int64"))

    n_HH_b = pm.Data("n_HH_b", np.zeros(1, dtype="int64"))
    n_HT_b = pm.Data("n_HT_b", np.zeros(1, dtype="int64"))
    n_TH_b = pm.Data("n_TH_b", np.zeros(1, dtype="int64"))
    n_TT_b = pm.Data("n_TT_b", np.zeros(1, dtype="int64"))

    # Free cognitive parameters
    tau = pm.HalfNormal("tau", sigma=5.0)
    # Adding a small constant to ensure alpha > 0 strictly for numerical safety in gammaln
    alpha = pm.Exponential("alpha", lam=1.0) + 1e-4

    def log_beta(x, y):
        return pt.gammaln(x) + pt.gammaln(y) - pt.gammaln(x + y)

    def log_p_markov(n_HH, n_HT, n_TH, n_TT, alpha):
        # Marginal likelihood of the transitions under Beta(alpha, alpha) priors
        log_p_H_trans = log_beta(n_HH + alpha, n_HT + alpha) - log_beta(alpha, alpha)
        log_p_T_trans = log_beta(n_TT + alpha, n_TH + alpha) - log_beta(alpha, alpha)
        # Assuming probability of the first flip is 0.5 under the regular process too
        return pt.log(0.5) + log_p_H_trans + log_p_T_trans

    log_p_random_a = n_a * pt.log(0.5)
    log_p_markov_a = log_p_markov(n_HH_a, n_HT_a, n_TH_a, n_TT_a, alpha)
    diagnosticity_a = log_p_random_a - log_p_markov_a

    log_p_random_b = n_b * pt.log(0.5)
    log_p_markov_b = log_p_markov(n_HH_b, n_HT_b, n_TH_b, n_TT_b, alpha)
    diagnosticity_b = log_p_random_b - log_p_markov_b

    # Higher diagnosticity means it looks MORE random.
    # We want p_left (prob of choosing A) to increase if diagnosticity_a > diagnosticity_b.
    p_left = pm.Deterministic(
        "p_left", pm.math.sigmoid(tau * (diagnosticity_a - diagnosticity_b))
    )

    # Numerical safety: bound probabilities away from exact 0 or 1
    p_left_safe = pt.clip(p_left, 1e-6, 1.0 - 1e-6)

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left_safe, observed=chose_left)
