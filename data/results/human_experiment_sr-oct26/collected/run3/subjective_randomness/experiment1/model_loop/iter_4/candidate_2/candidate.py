"""Sequential pattern alarm (survival) model.

People read a sequence flip by flip; at each flip a "this is a pattern" alarm
may fire, with a hazard that rises with the length of the current predictable
stretch — the current run of identical flips, or the current chain of strict
alternation (alternation must persist longer, a separate fitted threshold,
because people expect a fair coin to switch often). A sequence looks random to
the extent it is read to the end without the alarm firing (log survival
probability). People differ in how decisively this drives their choice.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

MAXLEN = 8
LEVELS = np.arange(1, MAXLEN + 1, dtype="float64")


def _stretch_counts(seq):
    """Histogram over flips of current run length and current alternation-chain length."""
    seq = seq.strip().upper()
    rep = np.zeros(MAXLEN)
    alt = np.zeros(MAXLEN)
    r = a = 0
    for i, c in enumerate(seq):
        if i == 0:
            r = a = 1
        elif c == seq[i - 1]:
            r, a = r + 1, 1
        else:
            r, a = 1, a + 1
        rep[r - 1] += 1
        alt[a - 1] += 1
    return rep, alt


def prepare_observed(rows):
    index, rep_tab, alt_tab = {}, [], []
    ia, ib, pid, y = [], [], [], []
    for row in rows:
        ids = []
        for key in ("sequence_a", "sequence_b"):
            s = str(row[key]).strip().upper()
            if s not in index:
                index[s] = len(rep_tab)
                rep, alt = _stretch_counts(s)
                rep_tab.append(rep)
                alt_tab.append(alt)
            ids.append(index[s])
        ia.append(ids[0])
        ib.append(ids[1])
        pid.append(int(row.get("participant_id", 0)))
        y.append(int(row.get("chose_left", 0)))
    return {
        "idx_a": np.asarray(ia, dtype="int64"),
        "idx_b": np.asarray(ib, dtype="int64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "rep_counts": np.asarray(rep_tab, dtype="float64"),
        "alt_counts": np.asarray(alt_tab, dtype="float64"),
    }


with pm.Model() as model:
    idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
    idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
    participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
    rep_counts = pm.Data("rep_counts", np.zeros((1, MAXLEN), dtype="float64"))
    alt_counts = pm.Data("alt_counts", np.zeros((1, MAXLEN), dtype="float64"))

    # Stretch length at which the alarm hazard reaches one half, for runs and alternation chains.
    tau_rep = pm.Normal("tau_rep", mu=4.0, sigma=1.5)
    tau_alt = pm.Normal("tau_alt", mu=6.0, sigma=2.0)
    # How steeply the hazard rises with each further flip of the stretch.
    k = pm.LogNormal("k", mu=0.0, sigma=0.5)

    # -log(1 - hazard) per flip at each stretch length.
    cost_rep = pt.softplus(k * (LEVELS - tau_rep))
    cost_alt = pt.softplus(k * (LEVELS - tau_alt))
    log_survival = -(pt.dot(rep_counts, cost_rep) + pt.dot(alt_counts, cost_alt))

    # Person-specific decisiveness (non-centred log-normal population).
    mu_lb = pm.Normal("mu_log_beta", mu=0.0, sigma=1.0)
    sd_lb = pm.HalfNormal("sigma_log_beta", sigma=0.5)
    z = pm.Normal("z_beta", 0.0, 1.0, shape=400)
    beta = pt.exp(mu_lb + sd_lb * z)

    diff = log_survival[idx_a] - log_survival[idx_b]
    p_left = pm.Deterministic(
        "p_left", pt.clip(pm.math.sigmoid(beta[participant_id] * diff), 1e-6, 1 - 1e-6)
    )

    chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
    pm.Bernoulli("response", p=p_left, observed=chose_left)
