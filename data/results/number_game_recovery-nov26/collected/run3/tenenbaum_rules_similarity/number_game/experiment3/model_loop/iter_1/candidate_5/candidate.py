"""Refinement of individual_rule_vs_range_tendency: people do the same Bayesian
concept learning over 45 rules (family-organised prior, exception rules as their
own family) and all intervals of 1..100 (beta prior over the share of the line a
range covers, size principle with examples from the concept with probability
0.9). The single change is what people differ on: not their rule-vs-range odds,
but how wide they expect a range to be. Each participant has their own expected
share of the line (logit-normal around an inferred population mean, with an
inferred spread; the beta's sharpness is shared), so a "broad" person extends
"yes" by magnitude well past the examples and a "narrow" person does not, while
rule members are judged alike. Participants are identified from session order
(trial_index restarts at 0); rows without session information use the
population-mean person.
"""

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from scipy.special import logsumexp

ALPHA = 0.9
NUMBERS = np.arange(1, 101)


def rule_concepts():
    rules = {
        "odd": [n for n in NUMBERS if n % 2 == 1],
        "primes": [n for n in NUMBERS if n > 1 and all(n % a for a in range(2, int(n**0.5) + 1))],
        "squares": [n * n for n in range(1, 11)],
        "cubes": [n**3 for n in range(1, 5)],
        "powers of 2, not 32": [2, 4, 8, 16, 64],
        "powers of 2, or 37": [2, 4, 8, 16, 32, 37, 64],
        "even": list(range(2, 101, 2)),
    }
    rules.update({f"multiples of {m}": list(range(m, 101, m)) for m in [3, 4, 5, 6, 7, 8, 9, 11, 12]})
    rules.update({f"powers of {m}": [m**k for k in range(1, 8) if m**k <= 100] for m in range(2, 11)})
    rules.update({f"ends in {d}": [n for n in NUMBERS if n % 10 == d] for d in range(10)})
    rules.update({f"contains digit {d}": [n for n in NUMBERS if str(d) in str(n)] for d in range(10)})
    return rules


_RULES = rule_concepts()
RULE_NAMES, RULES = list(_RULES), list(_RULES.values())
N_RULES = len(RULES)
INTERVALS = [list(range(m, n + 1)) for m in NUMBERS for n in range(m, 101)]
MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES + INTERVALS], dtype="float64")
EXAMPLE_LOG_LIK = np.log(ALPHA * MEMBERSHIP / MEMBERSHIP.sum(axis=1, keepdims=True) + (1 - ALPHA) / 100)
RULE_MEMBERSHIP = MEMBERSHIP[:N_RULES]
INTERVAL_MEMBERSHIP = MEMBERSHIP[N_RULES:]
INTERVAL_SIZE_IDX = (INTERVAL_MEMBERSHIP.sum(axis=1) - 1).astype("int64")  # L - 1
SIZES = np.arange(1, 101, dtype="float64")
LOG_N_OF_SIZE = np.log(101.0 - SIZES)  # number of intervals of each size
SHARE = (SIZES - 0.5) / 100.0  # share of the 1..100 line covered, in (0, 1)


def rule_family(name):
    if name in ("odd", "even"):
        return 0
    if name == "primes":
        return 1
    if name in ("squares", "cubes"):
        return 2
    if name.startswith("powers of 2,"):
        return 7
    if name.startswith("powers"):
        return 3
    if name.startswith("multiples"):
        return 4
    if name.startswith("ends in"):
        return 5
    return 6


FAMILY = np.array([rule_family(n) for n in RULE_NAMES])
N_FAMILIES = FAMILY.max() + 1
FAMILY_SIZE = np.bincount(FAMILY)
N_MAX_PEOPLE = 24  # row index N_MAX_PEOPLE = population-mean person

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    set_size_evidence = pm.Data("set_size_evidence", np.zeros((1, 100)))  # (sets, sizes), scaled
    set_log_scale = pm.Data("set_log_scale", np.zeros(1))  # (sets,): log of that scaling
    st_set = pm.Data("st_set", np.zeros(1, dtype="int64"))  # (sts,) distinct (set, target)
    st_rule_member = pm.Data("st_rule_member", np.zeros((1, N_RULES)))  # (sts, rules)
    st_size_evidence = pm.Data("st_size_evidence", np.zeros((1, 100)))  # (sts, sizes), scaled
    row_person = pm.Data("row_person", np.zeros(1, dtype="int64"))  # (trials,)
    row_set = pm.Data("row_set", np.zeros(1, dtype="int64"))  # (trials,)
    row_st = pm.Data("row_st", np.zeros(1, dtype="int64"))  # (trials,)
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Each person's expected share of the line a range covers (non-centred);
    # the last row is the population-mean person.
    lambda_0 = pm.Beta("lambda_0", 2, 2)  # prior share of rules over ranges
    share_logit_mean = pm.Normal("share_logit_mean", mu=-0.4, sigma=1.0)
    share_logit_sd = pm.HalfNormal("share_logit_sd", sigma=0.5)
    person_z = pm.Normal("person_z", mu=0.0, sigma=1.0, shape=N_MAX_PEOPLE)
    person_share = pm.math.sigmoid(
        share_logit_mean + share_logit_sd * pt.concatenate([person_z, pt.zeros(1)]))
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    # (people+1, sizes): each person's prior on one interval of size L.
    a = person_share[:, None] * concentration
    b = (1.0 - person_share[:, None]) * concentration
    log_size_prior = (a - 1.0) * np.log(SHARE)[None, :] + (b - 1.0) * np.log1p(-SHARE)[None, :]
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior, axis=1, keepdims=True)
    per_interval = pt.exp(log_size_prior - LOG_N_OF_SIZE[None, :])

    log_rules = rule_log_prior[None, :] + rule_log_lik  # (sets, rules), person-free
    z_rules = pt.logsumexp(log_rules, axis=1)
    post_rules = pt.exp(log_rules - z_rules[:, None])[st_set]
    in_if_rule = (post_rules * st_rule_member).sum(axis=1)  # (sts,)

    # Range evidence per (person, set) and its part from ranges holding the target.
    range_evidence = pt.maximum(per_interval @ set_size_evidence.T, 1e-300)  # (people+1, sets)
    range_in = per_interval @ st_size_evidence.T  # (people+1, sts)
    ev_row = range_evidence[row_person, row_set]
    in_if_range = pt.clip(range_in[row_person, row_st] / ev_row, 0.0, 1.0)
    log_z_ranges = pt.log(ev_row) + set_log_scale[row_set]

    w_rule = pm.math.sigmoid(pt.log(lambda_0) - pt.log1p(-lambda_0)
                             + z_rules[row_set] - log_z_ranges)
    p_row = w_rule * in_if_rule[row_st] + (1.0 - w_rule) * in_if_range

    p_yes = pm.Deterministic("p_yes", pt.clip(p_row, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def participant_index(rows):
    """Participant of each row from session order (trial_index restarts at 0);
    N_MAX_PEOPLE (the population-mean person) when unavailable."""
    if not rows or any(str(r.get("trial_index", "")).strip() in ("", "nan", "None") for r in rows):
        return np.full(len(rows), N_MAX_PEOPLE, dtype="int64")
    person, current = [], -1
    for r in rows:
        if int(float(r["trial_index"])) == 0:
            current += 1
        person.append(max(current, 0))
    person = np.array(person, dtype="int64")
    if person.max() >= N_MAX_PEOPLE:
        return np.full(len(rows), N_MAX_PEOPLE, dtype="int64")
    return person


def prepare_observed(rows):
    """Rows carry `number_set` ("16,8", in presented order), `target`, `rating`
    and, when available, `trial_index`."""
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    set_log_lik = np.stack([EXAMPLE_LOG_LIK[:, np.array(s) - 1].sum(axis=1) for s in unique])
    n_sets = len(unique)
    size_ev = np.zeros((n_sets, 100))
    size_member = np.zeros((n_sets, 100, 100))
    for i in range(n_sets):
        ll = set_log_lik[i, N_RULES:]
        for L in range(100):
            sel = INTERVAL_SIZE_IDX == L
            ev = logsumexp(ll[sel])
            size_ev[i, L] = ev
            size_member[i, L] = np.exp(ll[sel] - ev) @ INTERVAL_MEMBERSHIP[sel]
    person = participant_index(rows)
    st_keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    sts = list(dict.fromkeys(st_keys))
    st_index = {c: i for i, c in enumerate(sts)}
    st_s = np.array([c[0] for c in sts], dtype="int64")
    st_t = np.array([c[1] - 1 for c in sts], dtype="int64")
    scale = size_ev.max(axis=1)
    scaled = np.exp(size_ev - scale[:, None])
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "set_size_evidence": scaled,
        "set_log_scale": scale,
        "st_set": st_s,
        "st_rule_member": RULE_MEMBERSHIP[:, st_t].T.copy(),
        "st_size_evidence": scaled[st_s] * np.clip(size_member[st_s, :, st_t], 0.0, 1.0),
        "row_person": person,
        "row_set": np.array([index[s] for s in sets], dtype="int64"),
        "row_st": np.array([st_index[k] for k in st_keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
