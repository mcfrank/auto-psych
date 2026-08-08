"""
People judge the randomness of a sequence by accumulating a subjective sense of
typicality, but restrict that accumulation to a small recency window — the most
recent few tosses — rather than the entire sequence. Each event in the window
gets a per-event typicality penalty based on its imbalance and alternation
distance from an ideal prototype, accumulated over the window length. This is a
recency-normalization of the accumulated-typicality mechanism: judgments are
driven by what is in working memory, not the full pattern.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

RECENCY_WINDOW = 4


def compute_features(sequence_a, sequence_b):
    """Recency-window statistics (imbalance and alternation over the last few tosses)."""

    def window_stats(seq):
        seq = seq.strip().upper()
        win = seq[-RECENCY_WINDOW:]
        n = len(win)
        if n < 2:
            heads = 1.0 if seq and seq[0] == "H" else 0.0
            alts = 0.0
        else:
            heads = float(win.count("H"))
            alts = float(sum(1 for i in range(1, n) if win[i] != win[i - 1]))
        return float(n), heads, alts

    n_a, h_a, alt_a = window_stats(sequence_a)
    n_b, h_b, alt_b = window_stats(sequence_b)

    return {
        "rec_n_a": n_a,
        "rec_h_a": h_a,
        "rec_alt_a": alt_a,
        "rec_n_b": n_b,
        "rec_h_b": h_b,
        "rec_alt_b": alt_b,
    }


with pm.Model() as model:
    # Recency-window stimulus inputs
    rec_n_a = pm.Data("rec_n_a", np.zeros(1, dtype="float64"))
    rec_h_a = pm.Data("rec_h_a", np.zeros(1, dtype="float64"))
    rec_alt_a = pm.Data("rec_alt_a", np.zeros(1, dtype="float64"))

    rec_n_b = pm.Data("rec_n_b", np.zeros(1, dtype="float64"))
    rec_h_b = pm.Data("rec_h_b", np.zeros(1, dtype="float64"))
    rec_alt_b = pm.Data("rec_alt_b", np.zeros(1, dtype="float64"))

    # Free cognitive parameters
    ideal_p = pm.Beta("ideal_p", alpha=2.0, beta=2.0)
    ideal_alt = pm.Beta("ideal_alt", alpha=2.0, beta=2.0)
    w_p = pm.HalfNormal("w_p", sigma=5.0)
    w_alt = pm.HalfNormal("w_alt", sigma=5.0)
    base_typ = pm.Normal("base_typ", mu=0.0, sigma=5.0)
    penalty_power = pm.LogNormal("penalty_power", mu=0.0, sigma=0.5)

    # Recency-window rates (safeguard against division by zero)
    n_a_f = pt.maximum(rec_n_a, 1.0)
    n_b_f = pt.maximum(rec_n_b, 1.0)

    p_a = rec_h_a / n_a_f
    p_b = rec_h_b / n_b_f

    alt_rate_a = rec_alt_a / pt.maximum(rec_n_a - 1.0, 1.0)
    alt_rate_b = rec_alt_b / pt.maximum(rec_n_b - 1.0, 1.0)

    # Per-event typicality penalties within the window
    dev_p_a = pt.abs(p_a - ideal_p) + 1e-6
    dev_alt_a = pt.abs(alt_rate_a - ideal_alt) + 1e-6
    dev_p_b = pt.abs(p_b - ideal_p) + 1e-6
    dev_alt_b = pt.abs(alt_rate_b - ideal_alt) + 1e-6

    typ_a = base_typ - (w_p * pt.pow(dev_p_a, penalty_power) +
                        w_alt * pt.pow(dev_alt_a, penalty_power))
    typ_b = base_typ - (w_p * pt.pow(dev_p_b, penalty_power) +
                        w_alt * pt.pow(dev_alt_b, penalty_power))

    # Accumulate typicality over the recency window only
    rand_a = n_a_f * typ_a
    rand_b = n_b_f * typ_b

    # Choice probability
    p_left_raw = pm.math.sigmoid(rand_a - rand_b)
    p_left = pm.Deterministic("p_left", pt.clip(p_left_raw, 1e-6, 1.0 - 1e-6))

    # Observed response
    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
