"""Best-lag copy-rule detector.

People judge randomness by searching each sequence for a simple copy rule --
"each flip repeats the flip k places back" for some small lag k (lag 1 catches
streaks, lag 2 catches HTHT alternation, lag 4 catches HHTTHHTT) -- and a
sequence looks non-random to the extent that its best such rule predicts its
flips. They choose the sequence whose best copy rule fits worse, with people
differing only in how strongly that detected regularity drives their choice.
"""
import numpy as np
import pymc as pm


def compute_features(sequence_a, sequence_b):
    def best_copy_fit(seq):
        seq = seq.strip().upper()
        n = len(seq)
        best = 0.0
        for lag in range(1, max(1, n // 2) + 1):
            pairs = n - lag
            matches = sum(1 for i in range(lag, n) if seq[i] == seq[i - lag])
            best = max(best, matches / pairs)
        return best

    return {"copy_fit_a": best_copy_fit(sequence_a), "copy_fit_b": best_copy_fit(sequence_b)}


N_SLOTS = 400

with pm.Model() as model:
    fit_a = pm.Data("copy_fit_a", np.zeros(1, dtype="float64"))
    fit_b = pm.Data("copy_fit_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Person-specific sensitivity to detected regularity (log scale, non-centred).
    mu_log_beta = pm.Normal("mu_log_beta", mu=1.0, sigma=1.0)
    sigma_log_beta = pm.HalfNormal("sigma_log_beta", sigma=1.0)
    z_beta = pm.Normal("z_beta", mu=0.0, sigma=1.0, shape=N_SLOTS)
    beta = pm.Deterministic("beta", pm.math.exp(mu_log_beta + sigma_log_beta * z_beta))

    b = beta[participant_id]
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(b * (fit_b - fit_a)))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
