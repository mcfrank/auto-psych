"""Final-digit similarity gradient.

People ignore magnitude and judge the target by how much its final digit
resembles the final digits of the examples: same final digit = maximal
similarity, different digit of the same parity = partial similarity, opposite
parity = least similar. "Yes" grows with the target's average final-digit
similarity to all examples shown (graded by the share of examples it
resembles), with no credit for numerical closeness.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in number_set.split(",")]
    t = int(target)
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_parity_other = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    return {"frac_same_digit": same_digit, "frac_same_parity_other": same_parity_other}


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))

    # Similarity of a same-parity, different-digit ending (identical = 1,
    # opposite parity = 0) on the final-digit metric.
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    # Log-odds of yes at zero similarity and its gain per unit similarity.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic("p_yes", pm.math.sigmoid(intercept + gain * mean_sim))

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
