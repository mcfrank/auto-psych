"""Drifting yes-criterion on final-digit similarity.

People judge the target by its average final-digit similarity to the examples
(same final digit most similar, same parity partly, opposite parity least), as
in the simplest current model, but their response criterion drifts steadily
over the session: the bias toward answering "yes" changes linearly with how
far into the session the question is asked, so the same (set, target) pair is
accepted more (or less) often later in the session.
"""
import numpy as np
import pymc as pm

MAX_TRIAL = 449.0


def prepare_observed(rows):
    digit, par, pos, y = [], [], [], []
    for r in rows:
        xs = [int(x) for x in str(r["number_set"]).split(",")]
        t = int(float(r["target"]))
        n = float(len(xs))
        digit.append(sum(1 for x in xs if x % 10 == t % 10) / n)
        par.append(sum(1 for x in xs if x % 2 == t % 2 and x % 10 != t % 10) / n)
        # Session position, centred and scaled to roughly [-0.5, 0.5].
        pos.append(float(r["trial_index"]) / MAX_TRIAL - 0.5)
        y.append(int(float(r["rating"])))
    return {
        "frac_same_digit": np.asarray(digit, dtype="float64"),
        "frac_same_parity_other": np.asarray(par, dtype="float64"),
        "session_pos": np.asarray(pos, dtype="float64"),
        "rating": np.asarray(y, dtype="int64"),
    }


with pm.Model() as model:
    frac_digit = pm.Data("frac_same_digit", np.zeros(1, dtype="float64"))
    frac_par = pm.Data("frac_same_parity_other", np.zeros(1, dtype="float64"))
    session_pos = pm.Data("session_pos", np.zeros(1, dtype="float64"))

    # Final-digit similarity (identical = 1, same parity = s_parity, opposite = 0).
    s_parity = pm.Beta("s_parity", alpha=2.0, beta=2.0)
    mean_sim = frac_digit + s_parity * frac_par

    # Decision rule: criterion (log-odds bias) drifting linearly over the session.
    intercept = pm.Normal("intercept", mu=0.0, sigma=2.0)
    drift = pm.Normal("drift", mu=0.0, sigma=1.0)
    gain = pm.HalfNormal("gain", sigma=5.0)
    p_yes = pm.Deterministic(
        "p_yes", pm.math.sigmoid(intercept + drift * session_pos + gain * mean_sim)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
