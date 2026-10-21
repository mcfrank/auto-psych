"""Run 2's final, edge_streak_primacy_recency_pattern_rule, with switchable mechanisms.

The core, never removed: each sequence's tally span (the range of its running
heads-minus-tails count, as a share of its length) judged against an ideal
span, by squared distance weighted by a sensitivity; a logistic choice.

The mechanisms (``MECHANISMS``); "personal" means one value per participant
from a population (non-centred), and its removal gives everyone the
population value:

* ``span_ideal_personal``, ``span_beta_personal``: a personal ideal span and
  sensitivity.
* ``length_scaling``: the sensitivity scaled by a power of the length (off:
  the same at every length).
* ``switch`` (``switch_ideal_personal``, ``switch_weight_personal``): the
  switching rate judged against an ideal rate, with its own weight.
* ``variety``: a reward on the share of distinct three-flip chunks.
* ``rule_built``: a penalty on a sequence built by a rule (a repeated unit,
  or equal to its reversed complement).
* ``pattern_gain_personal``: a personal gain (median 1) scaling both.
* ``final_run``, ``opening_run``: a penalty per flip of the last / first run
  beyond the first.
* ``side_personal``: a personal left/right lean.
"""

import numpy as np
import pymc as pm

FINAL = "edge_streak_primacy_recency_pattern_rule"
N_SLOTS = 400
MECHANISMS = (
    "span_ideal_personal", "span_beta_personal", "length_scaling",
    "switch", "switch_ideal_personal", "switch_weight_personal",
    "variety", "rule_built", "pattern_gain_personal", "final_run", "opening_run", "side_personal",
)
REQUIRES = {
    "switch_ideal_personal": ("switch",),
    "switch_weight_personal": ("switch",),
    "pattern_gain_personal": ("variety", "rule_built"),
}
KEEP_ONE_OF = ()
# For variant names (``variants.variant_name``).
SHORT = {
    "span_ideal_personal": "idealP", "span_beta_personal": "betaP", "length_scaling": "len", "switch": "sw",
    "switch_ideal_personal": "swidealP", "switch_weight_personal": "swweightP", "variety": "var",
    "rule_built": "rule", "pattern_gain_personal": "patP", "final_run": "fin", "opening_run": "open",
    "side_personal": "sideP",
}
NEUTRAL = {
    "span_ideal_personal": {"sigma_ideal": 0.0},
    "span_beta_personal": {"sigma_log_beta": 0.0},
    "length_scaling": {"lam": 0.0},
    "switch": {"mu_kappa": 0.0},
    "switch_ideal_personal": {"sigma_sw": 0.0},
    "switch_weight_personal": {"sigma_kappa": 0.0},
    "variety": {"gamma": 0.0},
    "rule_built": {"rho": 0.0},
    "pattern_gain_personal": {"sigma_pat": 0.0},
    "final_run": {"tau": 0.0},
    "opening_run": {"tau_open": 0.0},
    "side_personal": {"sigma_side": 0.0},
}


def compute_features(sequence_a, sequence_b):
    def span_share(seq):
        tally, hi, lo = 0, 0, 0
        for c in seq:
            tally += 1 if c == "H" else -1
            hi = max(hi, tally)
            lo = min(lo, tally)
        return (hi - lo) / len(seq)

    def switch_rate(seq):
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def triplet_variety(seq):
        n_windows = len(seq) - 2
        if n_windows < 1:
            return 0.0
        distinct = len({seq[i:i + 3] for i in range(n_windows)})
        return distinct / min(8, n_windows)

    def rule_built(seq):
        # A unit of two or more flips repeated at least twice, or a sequence
        # equal to its own reversed complement.
        n = len(seq)
        if len(set(seq)) < 2:
            return 0.0
        for p in range(2, n // 2 + 1):
            if all(seq[i] == seq[i + p] for i in range(n - p)):
                return 1.0
        swapped = seq.translate(str.maketrans("HT", "TH"))
        return 1.0 if seq == swapped[::-1] else 0.0

    def terminal_excess(seq):
        # Flips of the final run beyond the first (0 when it ends on a switch).
        k = 1
        while k < len(seq) and seq[-1 - k] == seq[-1]:
            k += 1
        return float(k - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "span_a": span_share(a),
        "span_b": span_share(b),
        "sw_a": switch_rate(a),
        "sw_b": switch_rate(b),
        "trip_diff": triplet_variety(a) - triplet_variety(b),
        "rule_diff": rule_built(a) - rule_built(b),
        "term_diff": terminal_excess(a) - terminal_excess(b),
        "init_diff": terminal_excess(a[::-1]) - terminal_excess(b[::-1]),
        "log_len": float(np.log(len(a) / 6.0)),
    }


def build(on):
    on = frozenset(on)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")

    with pm.Model() as model:
        span_a = pm.Data("span_a", np.zeros(1, dtype="float64"))
        span_b = pm.Data("span_b", np.zeros(1, dtype="float64"))
        sw_a = pm.Data("sw_a", np.zeros(1, dtype="float64"))
        sw_b = pm.Data("sw_b", np.zeros(1, dtype="float64"))
        trip_diff = pm.Data("trip_diff", np.zeros(1, dtype="float64"))
        rule_diff = pm.Data("rule_diff", np.zeros(1, dtype="float64"))
        term_diff = pm.Data("term_diff", np.zeros(1, dtype="float64"))
        init_diff = pm.Data("init_diff", np.zeros(1, dtype="float64"))
        log_len = pm.Data("log_len", np.zeros(1, dtype="float64"))
        participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

        def per_person(mechanism, mu, sigma_name, sigma_prior, z_name):
            """Each trial's value: ``mu + sigma * z[participant]`` when ``mechanism`` is on, else ``mu``."""
            if mechanism not in on:
                return mu
            sigma = pm.HalfNormal(sigma_name, sigma=sigma_prior)
            z = pm.Normal(z_name, mu=0.0, sigma=1.0, shape=N_SLOTS)
            return (mu + sigma * z)[participant_id]

        def shared(mechanism, name, sigma):
            return pm.Normal(name, mu=0.0, sigma=sigma) if mechanism in on else 0.0

        mu_ideal = pm.Normal("mu_ideal", mu=-0.5, sigma=1.0)
        theta = pm.math.sigmoid(per_person("span_ideal_personal", mu_ideal, "sigma_ideal", 1.0, "z_ideal"))
        mu_log_beta = pm.Normal("mu_log_beta", mu=2.0, sigma=1.5)
        beta = pm.math.exp(per_person("span_beta_personal", mu_log_beta, "sigma_log_beta", 0.7, "z_beta"))
        lam = shared("length_scaling", "lam", 1.0)

        if "switch" in on:
            mu_sw = pm.Normal("mu_sw", mu=0.4, sigma=0.7)
            sw_p = pm.math.sigmoid(per_person("switch_ideal_personal", mu_sw, "sigma_sw", 0.7, "z_sw"))
            mu_kappa = pm.Normal("mu_kappa", mu=0.0, sigma=1.5)
            kappa = per_person("switch_weight_personal", mu_kappa, "sigma_kappa", 1.0, "z_kappa")
            sw_dist = (sw_b - sw_p) ** 2 - (sw_a - sw_p) ** 2
            switching = 4.0 * kappa * sw_dist
        else:
            switching = 0.0

        gamma = shared("variety", "gamma", 2.0)
        rho = shared("rule_built", "rho", 1.0)
        pat = pm.math.exp(per_person("pattern_gain_personal", 0.0, "sigma_pat", 0.7, "z_pat"))
        tau = shared("final_run", "tau", 1.0)
        tau_open = shared("opening_run", "tau_open", 1.0)
        side = per_person("side_personal", 0.0, "sigma_side", 0.3, "z_side")

        dist_a = (span_a - theta) ** 2
        dist_b = (span_b - theta) ** 2
        sens = beta * pm.math.exp(lam * log_len)
        logit = (
            sens * (dist_b - dist_a)
            + switching
            + pat * (gamma * trip_diff - rho * rule_diff)
            - tau * term_diff
            - tau_open * init_diff
            + side
        )
        p_left = pm.Deterministic("p_left", pm.math.sigmoid(logit))

        chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=p_left, observed=chose_left)
    return model
