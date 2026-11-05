"""Set-specific criterion graded-parity exemplar model (refines graded_parity_variability_exemplar).

People judge a target by its summed similarity to the remembered examples,
with attention to magnitude and final digit set by how variable the examples
are on each and a fixed parity penalty (same digit > same parity > opposite
parity). The one change: the response criterion is not shared by all sets.
On reading each example set people adopt their own threshold for how much
similarity counts as "belongs" -- some sets suggest a broad, permissive
category, others a strict one -- so sets differ in overall willingness to say
yes beyond what similarity alone predicts. Each set's criterion shift is drawn
from a common population distribution whose spread is inferred.
"""
from collections import Counter

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXK = 4


def prepare_observed(rows):
    sets = sorted({r["number_set"] for r in rows})
    index = {s: i for i, s in enumerate(sets)}
    N = len(rows)
    out = {f"{nm}_{k}": np.zeros(N) for nm in ("dist", "unit_diff", "parity_diff", "present") for k in range(MAXK)}
    out["mag_spread"] = np.zeros(N)
    out["digit_homog"] = np.zeros(N)
    out["set_idx"] = np.zeros(N, dtype="int64")
    out["rating"] = np.zeros(N, dtype="int64")
    for i, r in enumerate(rows):
        ex = [int(x) for x in str(r["number_set"]).split(",")][:MAXK]
        t = int(float(r["target"]))
        n = len(ex)
        for k, e in enumerate(ex):
            out[f"dist_{k}"][i] = abs(t - e) / 10.0
            out[f"unit_diff_{k}"][i] = 0.0 if t % 10 == e % 10 else 1.0
            out[f"parity_diff_{k}"][i] = 0.0 if t % 2 == e % 2 else 1.0
            out[f"present_{k}"][i] = 1.0
        if n > 1:
            out["mag_spread"][i] = (max(ex) - min(ex)) / 100.0
            out["digit_homog"][i] = (max(Counter(e % 10 for e in ex).values()) - 1) / (n - 1)
        else:
            out["mag_spread"][i] = 0.5
            out["digit_homog"][i] = 0.5
        out["set_idx"][i] = index[r["number_set"]]
        out["rating"][i] = int(float(r["rating"]))
    return out


N_SETS_MAX = 200  # upper bound on distinct example sets; unused slots are prior-only

with pm.Model() as model:
    dist = pt.stack([pm.Data(f"dist_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    udiff = pt.stack([pm.Data(f"unit_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    pdiff = pt.stack([pm.Data(f"parity_diff_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    present = pt.stack([pm.Data(f"present_{k}", np.zeros(1, dtype="float64")) for k in range(MAXK)], axis=1)
    spread = pm.Data("mag_spread", np.zeros(1, dtype="float64"))
    homog = pm.Data("digit_homog", np.zeros(1, dtype="float64"))
    set_idx = pm.Data("set_idx", np.zeros(1, dtype="int64"))

    log_c_mag = pm.Normal("log_c_mag", mu=0.0, sigma=1.5)
    log_c_unit = pm.Normal("log_c_unit", mu=1.0, sigma=1.5)
    log_c_parity = pm.Normal("log_c_parity", mu=0.0, sigma=1.5)
    spread_tuning = pm.Normal("spread_tuning", mu=0.0, sigma=4.0)
    homog_tuning = pm.Normal("homog_tuning", mu=0.0, sigma=2.0)
    alpha = pm.Normal("alpha", mu=0.0, sigma=2.0)
    beta = pm.HalfNormal("beta", sigma=2.0)

    # Set-specific criterion shifts (non-centred).
    crit_sd = pm.HalfNormal("crit_sd", sigma=1.0)
    crit_z = pm.Normal("crit_z", mu=0.0, sigma=1.0, shape=N_SETS_MAX)
    crit = crit_sd * crit_z[set_idx]

    c_mag = pt.exp(log_c_mag - spread_tuning * spread)[:, None]
    c_unit = pt.exp(log_c_unit + homog_tuning * homog)[:, None]
    c_parity = pt.exp(log_c_parity)

    log_sim = -c_mag * dist - c_unit * udiff - c_parity * pdiff
    log_sim = pt.switch(present > 0.5, log_sim, -1e3)
    log_summed = pm.math.logsumexp(log_sim, axis=1).flatten()

    p_yes = pm.Deterministic(
        "p_yes", pt.clip(pm.math.sigmoid(alpha + crit + beta * log_summed), 1e-6, 1 - 1e-6)
    )

    rating = pm.Data("rating", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_yes, observed=rating)
