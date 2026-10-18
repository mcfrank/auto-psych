"""Firm criterion with attentional lapses on final-digit similarity.

People judge the target by its average final-digit similarity to the examples
(same final digit most similar, same parity partly, opposite parity least), but
answer by a firm criterion: "yes" almost deterministically when similarity
clears it, "no" when it falls short, except on an occasional fraction of trials
where attention lapses and they answer by a coin flip. This all-or-none rule
with lapses predicts more polarized item-level yes rates than a graded response.
"""
import numpy as np
import pymc as pm


def compute_features(number_set, target):
    xs = [int(x) for x in str(number_set).split(",")]
    t = int(float(target))
    n = float(len(xs))
    same_digit = sum(1 for x in xs if x % 10 == t % 10) / n
    same_par = sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n
    return {"frac_same_digit": same_digit, "frac_same_parity_other": same_par}


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))

    # Final-digit similarity (identical = 1, same parity = s_parity, opposite = 0).
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    # Decision rule: criterion on similarity, steepness of the threshold, lapses.
    criterion = pm.Beta("criterion", alpha=2.0, beta=2.0)
    log_sharp = pm.Normal("log_sharp", mu=2.5, sigma=1.0)
    lapse = pm.Beta("lapse", alpha=1.5, beta=6.0)

    p_follow = pm.math.sigmoid(pm.math.exp(log_sharp) * (mean_sim - criterion))
    p_yes = pm.Deterministic("p_yes", 0.5 * lapse + (1.0 - lapse) * p_follow)

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
