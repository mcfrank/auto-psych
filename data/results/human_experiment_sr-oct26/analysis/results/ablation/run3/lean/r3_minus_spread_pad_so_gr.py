"""Run 3's final, mild_symmetric_trick_coin_streak_weary, with switchable mechanisms.

The core, never removed: a sequence looks random to the extent that a fair
coin explains it better than the regular generators do (their log marginal
likelihood, unknowns integrated out), weighed by a sensitivity; a logistic
choice.

The mechanisms (``MECHANISMS``); "personal" means one value per participant
from a population (non-centred), and its removal gives everyone the
population value. At least one regular generator stays (``KEEP_ONE_OF``):

* the regular generators: ``markov`` (a coin with an unknown switch rate),
  ``trick_coin`` (a coin with an unknown heads rate, Beta(kappa/2, kappa/2);
  ``trick_coin_spread`` fits kappa, off: kappa = 2, uniform), ``motif`` (a
  repeating unit of period 1-4 with slips; ``motif_weight`` its prior weight
  among the generators, off: equal weights; ``motif_weight_personal``).
* ``beta_personal``: a personal (heavy-tailed, signed) sensitivity.
* ``switch_belief`` (``switch_belief_personal``): a believed switch rate of a
  fair coin, a weight on the switch count.
* ``length_scaling``: the evidence weighed per flip by a power of the length
  (off: total evidence); ``length_padding``: c imagined unremarkable flips pad
  the length (off: c = 0).
* ``second_order``: after a switch, chance is believed to switch again more
  or less often.
* ``gamblers_run``: the believed switch odds rise with the current run's length.
* ``side_bias`` (``side_personal``): a left/right lean.
"""

import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

FINAL = "mild_symmetric_trick_coin_streak_weary"
PERIODS = (1, 2, 3, 4)
N_SLOTS = 400
GENERATORS = ("markov", "trick_coin", "motif")
MECHANISMS = (
    *GENERATORS, "trick_coin_spread", "motif_weight", "motif_weight_personal", "beta_personal",
    "switch_belief", "switch_belief_personal", "length_scaling", "length_padding",
    "second_order", "gamblers_run", "side_bias", "side_personal",
)
REQUIRES = {
    "trick_coin_spread": ("trick_coin",),
    "motif_weight": ("motif",),
    "motif_weight_personal": ("motif_weight",),
    "switch_belief_personal": ("switch_belief",),
    "length_padding": ("length_scaling",),
    "side_personal": ("side_bias",),
}
KEEP_ONE_OF = (GENERATORS,)
# For variant names (``variants.variant_name``).
SHORT = {
    "markov": "markov", "trick_coin": "trick", "trick_coin_spread": "spread", "motif": "motif",
    "motif_weight": "mw", "motif_weight_personal": "mwP", "beta_personal": "betaP", "switch_belief": "sb",
    "switch_belief_personal": "sbP", "length_scaling": "len", "length_padding": "pad", "second_order": "so",
    "gamblers_run": "gr", "side_bias": "side", "side_personal": "sideP",
}
NEUTRAL = {
    "markov": None,
    "trick_coin": None,
    "trick_coin_spread": {"log_kappa": math.log(2.0)},
    "motif": None,
    "motif_weight": {"mu_a": 0.0},
    "motif_weight_personal": {"sigma_a": 0.0},
    "beta_personal": {"sigma_b": 0.0},
    "switch_belief": {"mu_w": 0.0},
    "switch_belief_personal": {"sigma_w": 0.0},
    "length_scaling": {"gamma": 0.0},
    "length_padding": {"log_c": -np.inf},
    "second_order": {"d_switch2": 0.0},
    "gamblers_run": {"g_run": 0.0},
    "side_bias": {"mu_side": 0.0},
    "side_personal": {"sigma_side": 0.0},
}


def _log_beta(a, b):
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def _summary(seq):
    seq = seq.strip().upper()
    n = len(seq)
    k = sum(1 for x, y in zip(seq, seq[1:]) if x != y)
    h = seq.count("H")
    log_markov = math.log(0.5) + _log_beta(k + 1, (n - 1 - k) + 1)
    trans = [1 if x != y else 0 for x, y in zip(seq, seq[1:])]
    ss = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 1)
    sr = sum(1 for u, v in zip(trans, trans[1:]) if u == 1 and v == 0)
    gam = 0.0
    run = 1
    for x, y in zip(seq, seq[1:]):
        if run >= 3:
            gam += (run - 2) * (1.0 if x != y else -1.0)
        run = 1 if x != y else run + 1
    out = {"k": float(k), "lm": log_markov, "h": float(h), "alt2": float(ss - sr), "gam": gam}
    for p in PERIODS:
        compared = max(n - p, 0)
        mism = sum(1 for i in range(p, n) if seq[i] != seq[i - p])
        out[f"c{p}"] = float(compared)
        out[f"m{p}"] = float(mism)
    out["n"] = float(n)
    return out


def prepare_observed(rows):
    """Table of distinct sequences (regular-generator statistics) plus per-trial indices."""
    index = {}
    table = []
    idx_a, idx_b, switch_diff, alt2_diff, gam_diff, pid, y = [], [], [], [], [], [], []
    for r in rows:
        ia_ib = []
        for key in ("sequence_a", "sequence_b"):
            seq = str(r[key]).strip().upper()
            if seq not in index:
                index[seq] = len(table)
                table.append(_summary(seq))
            ia_ib.append(index[seq])
        idx_a.append(ia_ib[0])
        idx_b.append(ia_ib[1])
        switch_diff.append(table[ia_ib[0]]["k"] - table[ia_ib[1]]["k"])
        alt2_diff.append(table[ia_ib[0]]["alt2"] - table[ia_ib[1]]["alt2"])
        gam_diff.append(table[ia_ib[0]]["gam"] - table[ia_ib[1]]["gam"])
        pid.append(int(r.get("participant_id", 0)))
        y.append(int(r.get("chose_left", 0)))
    out = {
        "idx_a": np.asarray(idx_a, dtype="int64"),
        "idx_b": np.asarray(idx_b, dtype="int64"),
        "switch_diff": np.asarray(switch_diff, dtype="float64"),
        "alt2_diff": np.asarray(alt2_diff, dtype="float64"),
        "gam_diff": np.asarray(gam_diff, dtype="float64"),
        "participant_id": np.asarray(pid, dtype="int64"),
        "chose_left": np.asarray(y, dtype="int64"),
        "seq_lmark": np.asarray([t["lm"] for t in table], dtype="float64"),
        "seq_heads": np.asarray([t["h"] for t in table], dtype="float64"),
        "seq_len": np.asarray([t["n"] for t in table], dtype="float64"),
    }
    for p in PERIODS:
        out[f"seq_cmp{p}"] = np.asarray([t[f"c{p}"] for t in table], dtype="float64")
        out[f"seq_mis{p}"] = np.asarray([t[f"m{p}"] for t in table], dtype="float64")
    return out


def build(on):
    on = frozenset(on)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")
    if not on & set(GENERATORS):
        raise ValueError(f"at least one regular generator of {GENERATORS} must stay on")

    with pm.Model() as model:
        idx_a = pm.Data("idx_a", np.zeros(1, dtype="int64"))
        idx_b = pm.Data("idx_b", np.zeros(1, dtype="int64"))
        switch_diff = pm.Data("switch_diff", np.zeros(1, dtype="float64"))
        alt2_diff = pm.Data("alt2_diff", np.zeros(1, dtype="float64"))
        gam_diff = pm.Data("gam_diff", np.zeros(1, dtype="float64"))
        participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))
        seq_lmark = pm.Data("seq_lmark", np.zeros(1, dtype="float64"))
        seq_heads = pm.Data("seq_heads", np.zeros(1, dtype="float64"))
        seq_len = pm.Data("seq_len", np.full(1, 2.0))
        seq_cmp = {p: pm.Data(f"seq_cmp{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
        seq_mis = {p: pm.Data(f"seq_mis{p}", np.zeros(1, dtype="float64")) for p in PERIODS}
        pid = participant_id

        def per_person(mechanism, mu, sigma_name, sigma_prior, z):
            """Each trial's value: ``mu + sigma * z[participant]`` when ``mechanism`` is on, else ``mu``.
            ``z`` makes the person-level vector (called only when on)."""
            if mechanism not in on:
                return mu
            sigma = pm.HalfNormal(sigma_name, sigma=sigma_prior)
            return (mu + sigma * z())[pid]

        def shared(mechanism, name, mu, sigma):
            return pm.Normal(name, mu=mu, sigma=sigma) if mechanism in on else 0.0

        # Log marginal likelihood of each distinct sequence under each regular generator.
        if "motif" in on:
            logit_eps = pm.Normal("logit_eps", mu=-2.0, sigma=1.0)
            eps = pm.math.sigmoid(logit_eps)
            log_eps = pt.log(eps)
            log_1m_eps = pt.log1p(-eps)
            motif_terms = []
            for p in PERIODS:
                free = pt.minimum(float(p), seq_len)
                motif_terms.append(
                    -free * math.log(2.0)
                    + (seq_cmp[p] - seq_mis[p]) * log_1m_eps
                    + seq_mis[p] * log_eps
                )
            log_motif = pm.math.logsumexp(pt.stack(motif_terms, axis=0), axis=0, keepdims=False) - math.log(len(PERIODS))
        if "trick_coin" in on:
            kappa = pt.exp(pm.Normal("log_kappa", mu=1.5, sigma=0.75)) if "trick_coin_spread" in on else pt.as_tensor(2.0)
            a_h = 0.5 * kappa
            a_t = 0.5 * kappa
            seq_lbias = (
                pt.gammaln(seq_heads + a_h) + pt.gammaln(seq_len - seq_heads + a_t)
                - pt.gammaln(seq_len + kappa)
                - pt.gammaln(a_h) - pt.gammaln(a_t) + pt.gammaln(kappa)
            )
        if "motif" in on:
            # Prior weight of the motif generator among the regular explanations
            # (weights 1 : 1 : exp(a)); the normaliser cancels in the difference.
            a_motif = (
                per_person("motif_weight_personal", pm.Normal("mu_a", mu=0.0, sigma=1.0), "sigma_a", 1.0,
                           lambda: pm.Normal("z_a", mu=0.0, sigma=1.0, shape=N_SLOTS))
                if "motif_weight" in on else 0.0
            )

        def _log_regular(idx):
            terms = []
            if "markov" in on:
                terms.append(seq_lmark[idx])
            if "trick_coin" in on:
                terms.append(seq_lbias[idx])
            if "motif" in on:
                terms.append(log_motif[idx] + a_motif)
            return pm.math.logsumexp(pt.stack(terms, axis=0), axis=0, keepdims=False)

        log_regular_diff = _log_regular(idx_a) - _log_regular(idx_b)

        # Signed sensitivity (personal: a heavy-tailed population).
        mu_b = pm.Normal("mu_b", mu=1.0, sigma=0.5)
        beta = per_person("beta_personal", mu_b, "sigma_b", 0.5,
                          lambda: pm.StudentT("z_beta", nu=4.0, mu=0.0, sigma=1.0, shape=N_SLOTS))

        # Length normalisation of the evidence (gamma = 0: total evidence, 1: per flip).
        if "length_scaling" in on:
            gamma = pm.Normal("gamma", mu=0.5, sigma=0.5)
            trial_len = seq_len[idx_a]
            c_pad = pt.exp(pm.Normal("log_c", mu=0.5, sigma=1.0)) if "length_padding" in on else 0.0
            length_scale = pt.exp(-gamma * pt.log((trial_len + c_pad) / (5.0 + c_pad)))
        else:
            length_scale = 1.0

        # Believed switch rate of a fair coin, as the product w = beta * logit_q.
        w = (
            per_person("switch_belief_personal", pm.Normal("mu_w", mu=0.0, sigma=1.0), "sigma_w", 0.5,
                       lambda: pm.Normal("z_w", mu=0.0, sigma=1.0, shape=N_SLOTS))
            if "switch_belief" in on else 0.0
        )
        d_switch2 = shared("second_order", "d_switch2", 0.0, 1.0)
        g_run = shared("gamblers_run", "g_run", 0.0, 0.3)
        log_chance2_diff = 0.5 * d_switch2 * alt2_diff + 0.5 * g_run * gam_diff
        eta = (w * switch_diff + beta * (log_chance2_diff - log_regular_diff)) * length_scale

        side = (
            per_person("side_personal", pm.Normal("mu_side", mu=0.0, sigma=0.5), "sigma_side", 0.3,
                       lambda: pm.Normal("z_side", mu=0.0, sigma=1.0, shape=N_SLOTS))
            if "side_bias" in on else 0.0
        )
        p_left = pm.Deterministic("p_left", pm.math.sigmoid(eta + side))

        chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=p_left, observed=chose_left)
    return model


# --- this variant (src.subjective_randomness.ablation.variants) ---
MECHANISMS_ON = ('markov', 'trick_coin', 'motif', 'motif_weight', 'motif_weight_personal', 'beta_personal', 'switch_belief', 'switch_belief_personal', 'length_scaling', 'side_bias', 'side_personal')
model = build(MECHANISMS_ON)
