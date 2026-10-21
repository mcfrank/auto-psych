"""Run 1's final, personal_pattern_detection_gain, with switchable mechanisms.

The core, never removed: each sequence's switching rate judged against an
ideal rate (squared distance, weighted by a sensitivity), a logistic choice.

The mechanisms (``MECHANISMS``); "personal" means one value per participant
from a population (non-centred), and its removal gives everyone the
population value:

* ``ideal_personal``, ``beta_personal``: a personal ideal rate and sensitivity.
* ``asymmetry``: over-alternation penalised by a fitted fraction of the
  under-alternation penalty (off: symmetric).
* ``imbalance`` (``imbalance_personal``): a weight on H/T imbalance.
* ``longest_run`` (``longest_run_personal``): a weight on the longest run.
* structural cues of design, shared weights: ``final_run``, ``palindrome``
  (mirror symmetry), ``variety`` (run-length variety), ``lopsided`` (the most
  lopsided stretch), ``antisymmetry`` (complement symmetry).
* ``gain_personal``: a personal gain (median 1) scaling every structural cue.
* ``lapse`` (``lapse_personal``): a guessing rate (off: none).
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt

FINAL = "personal_pattern_detection_gain"
MAX_PARTICIPANTS = 400
STRUCTURE = ("final_run", "palindrome", "variety", "lopsided", "antisymmetry")
MECHANISMS = (
    "ideal_personal", "beta_personal", "asymmetry", "imbalance", "imbalance_personal",
    "longest_run", "longest_run_personal", *STRUCTURE, "gain_personal", "lapse", "lapse_personal",
)
REQUIRES = {
    "imbalance_personal": ("imbalance",),
    "longest_run_personal": ("longest_run",),
    "gain_personal": STRUCTURE,
    "lapse_personal": ("lapse",),
}
KEEP_ONE_OF = ()
# For variant names (``variants.variant_name``).
SHORT = {
    "ideal_personal": "idealP", "beta_personal": "betaP", "asymmetry": "asym", "imbalance": "imb",
    "imbalance_personal": "imbP", "longest_run": "lrun", "longest_run_personal": "lrunP", "final_run": "fin",
    "palindrome": "pal", "variety": "var", "lopsided": "lop", "antisymmetry": "anti", "gain_personal": "gainP",
    "lapse": "lapse", "lapse_personal": "lapseP",
}
NEUTRAL = {
    "ideal_personal": {"sigma_ideal": 0.0},
    "beta_personal": {"sigma_beta": 0.0},
    "asymmetry": {"log_over": 0.0},
    "imbalance": {"mu_gamma": 0.0},
    "imbalance_personal": {"sigma_gamma": 0.0},
    "longest_run": {"mu_delta": 0.0},
    "longest_run_personal": {"sigma_delta": 0.0},
    "final_run": {"kappa": 0.0},
    "palindrome": {"rho": 0.0},
    "variety": {"omega": 0.0},
    "lopsided": {"eta": 0.0},
    "antisymmetry": {"chi": 0.0},
    "gain_personal": {"sigma_gain": 0.0},
    "lapse": {"mu_lapse": -np.inf},
    "lapse_personal": {"sigma_lapse": 0.0},
}


def compute_features(sequence_a, sequence_b):
    def alternation_rate(seq):
        if len(seq) < 2:
            raise ValueError(f"sequence too short: {seq!r}")
        return sum(1 for x, y in zip(seq, seq[1:]) if x != y) / (len(seq) - 1)

    def imbalance(seq):
        return abs(seq.count("H") - seq.count("T")) / len(seq)

    def longest_run_share(seq):
        best = cur = 1
        for x, y in zip(seq, seq[1:]):
            cur = cur + 1 if x == y else 1
            best = max(best, cur)
        return (best - 1) / (len(seq) - 1)

    def final_run_share(seq):
        cur = 1
        for x, y in zip(seq[-2::-1], seq[:0:-1]):
            if x != y:
                break
            cur += 1
        return (cur - 1) / (len(seq) - 1)

    def palindrome(seq):
        return 1.0 if seq == seq[::-1] else 0.0

    def run_variety(seq):
        runs = []
        cur = 1
        for x, y in zip(seq, seq[1:]):
            if x == y:
                cur += 1
            else:
                runs.append(cur)
                cur = 1
        runs.append(cur)
        n = len(seq)
        kmax = 1
        while (kmax + 1) * (kmax + 2) // 2 <= n:
            kmax += 1
        if kmax <= 1:
            return 0.0
        return (len(set(runs)) - 1) / (kmax - 1)

    def complement_symmetric(seq):
        if len(seq) < 4:
            return 0.0
        flipped = "".join("T" if c == "H" else "H" for c in seq[::-1])
        return 1.0 if seq == flipped else 0.0

    def lopsided_stretch(seq):
        # Largest |#H - #T| over contiguous stretches = range of the running tally.
        tally, lo, hi = 0, 0, 0
        for x in seq:
            tally += 1 if x == "H" else -1
            lo = min(lo, tally)
            hi = max(hi, tally)
        return (hi - lo - 1) / (len(seq) - 1)

    a = sequence_a.strip().upper()
    b = sequence_b.strip().upper()
    return {
        "alt_rate_a": alternation_rate(a),
        "alt_rate_b": alternation_rate(b),
        "imbalance_a": imbalance(a),
        "imbalance_b": imbalance(b),
        "maxrun_a": longest_run_share(a),
        "maxrun_b": longest_run_share(b),
        "endrun_a": final_run_share(a),
        "endrun_b": final_run_share(b),
        "palin_a": palindrome(a),
        "palin_b": palindrome(b),
        "variety_a": run_variety(a),
        "variety_b": run_variety(b),
        "lopsided_a": lopsided_stretch(a),
        "lopsided_b": lopsided_stretch(b),
        "antisym_a": complement_symmetric(a),
        "antisym_b": complement_symmetric(b),
    }


def build(on):
    on = frozenset(on)
    unknown = sorted(on - set(MECHANISMS))
    if unknown:
        raise ValueError(f"unknown mechanisms {unknown}; known: {list(MECHANISMS)}")

    with pm.Model() as model:
        alt_rate_a = pm.Data("alt_rate_a", np.zeros(1, dtype="float64"))
        alt_rate_b = pm.Data("alt_rate_b", np.zeros(1, dtype="float64"))
        imbalance_a = pm.Data("imbalance_a", np.zeros(1, dtype="float64"))
        imbalance_b = pm.Data("imbalance_b", np.zeros(1, dtype="float64"))
        maxrun_a = pm.Data("maxrun_a", np.zeros(1, dtype="float64"))
        maxrun_b = pm.Data("maxrun_b", np.zeros(1, dtype="float64"))
        endrun_a = pm.Data("endrun_a", np.zeros(1, dtype="float64"))
        endrun_b = pm.Data("endrun_b", np.zeros(1, dtype="float64"))
        palin_a = pm.Data("palin_a", np.zeros(1, dtype="float64"))
        palin_b = pm.Data("palin_b", np.zeros(1, dtype="float64"))
        variety_a = pm.Data("variety_a", np.zeros(1, dtype="float64"))
        variety_b = pm.Data("variety_b", np.zeros(1, dtype="float64"))
        lopsided_a = pm.Data("lopsided_a", np.zeros(1, dtype="float64"))
        lopsided_b = pm.Data("lopsided_b", np.zeros(1, dtype="float64"))
        antisym_a = pm.Data("antisym_a", np.zeros(1, dtype="float64"))
        antisym_b = pm.Data("antisym_b", np.zeros(1, dtype="float64"))
        participant_id = pm.Data("participant_id", np.zeros(1, dtype="int64"))

        def per_person(mechanism, mu, suffix, sigma_prior):
            """Each trial's value: ``mu + sigma * z[participant]`` when ``mechanism`` is on, else ``mu``."""
            if mechanism not in on:
                return mu
            sigma = pm.HalfNormal(f"sigma_{suffix}", sigma=sigma_prior)
            z = pm.Normal(f"z_{suffix}", mu=0.0, sigma=1.0, shape=MAX_PARTICIPANTS)
            return (mu + sigma * z)[participant_id]

        def shared(mechanism, name, sigma):
            return pm.Normal(name, mu=0.0, sigma=sigma) if mechanism in on else 0.0

        mu_ideal = pm.Normal("mu_ideal", mu=0.4, sigma=1.0)
        theta = pm.math.sigmoid(per_person("ideal_personal", mu_ideal, "ideal", 1.0))
        mu_beta = pm.Normal("mu_beta", mu=2.5, sigma=0.5)
        b = pm.math.exp(per_person("beta_personal", mu_beta, "beta", 0.7))

        g = per_person("imbalance_personal", pm.Normal("mu_gamma", mu=0.0, sigma=2.0), "gamma", 1.5) if "imbalance" in on else 0.0
        d = per_person("longest_run_personal", pm.Normal("mu_delta", mu=0.0, sigma=2.0), "delta", 1.5) if "longest_run" in on else 0.0

        kappa = shared("final_run", "kappa", 1.5)
        rho = shared("palindrome", "rho", 1.5)
        omega = shared("variety", "omega", 1.5)
        eta = shared("lopsided", "eta", 1.5)
        chi = shared("antisymmetry", "chi", 1.5)
        s = pm.math.exp(per_person("gain_personal", 0.0, "gain", 0.7))

        over_frac = pm.math.exp(pm.Normal("log_over", mu=0.0, sigma=1.0)) if "asymmetry" in on else 1.0

        def structure(endrun, palin, variety, lopsided, antisym):
            return -kappa * endrun - rho * palin + omega * variety - eta * lopsided - chi * antisym

        def rate_penalty(rate):
            dev = rate - theta
            return pt.sqr(dev) * pt.switch(dev > 0, over_frac, 1.0)

        score_a = -b * rate_penalty(alt_rate_a) - g * imbalance_a - d * maxrun_a + s * structure(endrun_a, palin_a, variety_a, lopsided_a, antisym_a)
        score_b = -b * rate_penalty(alt_rate_b) - g * imbalance_b - d * maxrun_b + s * structure(endrun_b, palin_b, variety_b, lopsided_b, antisym_b)
        p_engaged = pm.math.sigmoid(score_a - score_b)
        if "lapse" in on:
            lam = pm.math.sigmoid(per_person("lapse_personal", pm.Normal("mu_lapse", mu=-2.0, sigma=1.0), "lapse", 1.0))
            p_engaged = 0.5 * lam + (1.0 - lam) * p_engaged
        p_left = pm.Deterministic("p_left", pt.clip(p_engaged, 1e-6, 1 - 1e-6))

        chose_left = pm.Data("chose_left", np.zeros(1, dtype="int64"))
        pm.Bernoulli("response", p=p_left, observed=chose_left)
    return model
