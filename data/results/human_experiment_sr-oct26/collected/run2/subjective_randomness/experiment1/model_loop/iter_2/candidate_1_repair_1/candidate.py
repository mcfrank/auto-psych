"""Most salient regular stretch.

People judge a sequence by its single most striking regular stretch: the
longest streak of identical flips and the longest stretch of perfect
alternation are each scored for salience, and the most salient of the two
(a smooth maximum, not a sum over the whole sequence) decides how non-random
the sequence looks. The less striking sequence is chosen as more random.
Streak salience is shared; alternation-stretch salience is positive and
person-specific (log-normal across people).
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt


def compute_features(sequence_a, sequence_b):
    def longest_run(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return best

    def longest_alt(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x != y else 1
            best = max(best, cur)
        return best

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "run_excess_a": float(longest_run(a) - 1),
        "run_excess_b": float(longest_run(b) - 1),
        "alt_excess_a": float(longest_alt(a) - 2),
        "alt_excess_b": float(longest_alt(b) - 2),
    }


N_SLOTS = 400
TAU = 0.5  # softness of the maximum (fixed, for identifiability)

with pm.Model() as model:
    run_a = pm.Data("run_excess_a", np.zeros(1, dtype="float64"))
    run_b = pm.Data("run_excess_b", np.zeros(1, dtype="float64"))
    alt_a = pm.Data("alt_excess_a", np.zeros(1, dtype="float64"))
    alt_b = pm.Data("alt_excess_b", np.zeros(1, dtype="float64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

    # Shared salience of a streak, per extra repeated flip.
    w_run = pm.HalfNormal("w_run", sigma=1.0)
    # Person-specific salience of a perfectly alternating stretch: positive
    # (a salience, not a preference), log-normal across people, non-centred.
    log_mu_alt = pm.Normal("log_mu_alt", mu=-1.0, sigma=1.0)
    sigma_alt = pm.HalfNormal("sigma_alt", sigma=0.5)
    z_alt = pm.Normal("z_alt", mu=0.0, sigma=1.0, shape=N_SLOTS)
    w_alt = pm.Deterministic("w_alt", pt.exp(log_mu_alt + sigma_alt * z_alt))
    wa = w_alt[participant_id]

    def striking(run, alt):
        return TAU * pt.logaddexp(w_run * run / TAU, wa * alt / TAU)

    s_a = striking(run_a, alt_a)
    s_b = striking(run_b, alt_b)
    p_left = pm.Deterministic("p_left", pm.math.sigmoid(s_b - s_a))

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
