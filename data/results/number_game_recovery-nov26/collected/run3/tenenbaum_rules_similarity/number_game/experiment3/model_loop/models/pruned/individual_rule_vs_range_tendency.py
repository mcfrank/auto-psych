"""Refinement of beta_share_width_exception_concepts: people do the same Bayesian
concept learning over 45 rules (family-organised prior, exception rules as their
own family) and all intervals of 1..100 (beta prior over the share of the line a
range covers, size principle with examples from the concept with probability
0.9). The single change is that people differ in how readily they think of the
hidden program as following an arithmetic rule rather than a magnitude range:
each participant has their own prior rule-vs-range odds, drawn from a population
distribution (inferred mean and spread). A "rule person" confines yes to rule
members, a "range person" spreads yes over nearby numbers, for the same examples.
Participants are identified from the session order (trial_index restarts at 0);
rows with no session information get the population-average prediction.
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

# All intervals of one size share their prior, so prepare_observed sums each
# set's interval likelihood within every size (parameter-free) and records, per
# size, the share of that evidence from intervals containing each target.
N_MAX_PEOPLE = 12

with pm.Model() as model:
    rule_log_lik = pm.Data("rule_log_lik", np.zeros((1, N_RULES)))  # (sets, rules)
    size_log_evidence = pm.Data("size_log_evidence", np.zeros((1, 100)))  # (sets, sizes)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))  # (cells,)
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))  # (cells, rules)
    cell_size_member = pm.Data("cell_size_member", np.zeros((1, 100)))  # (cells, sizes)
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))  # (trials,)
    row_person = pm.Data("row_person", np.zeros(1, dtype="int64"))  # (trials,), -1 = unknown
    n_people = pm.Data("n_people", np.ones(1, dtype="int64"))  # (1,): participants seen
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    # Each person's prior log-odds of a rule over a range (non-centred).
    rule_logit_mean = pm.Normal("rule_logit_mean", mu=0.0, sigma=1.5)
    rule_logit_sd = pm.HalfNormal("rule_logit_sd", sigma=0.5)
    person_z = pm.Normal("person_z", mu=0.0, sigma=1.0, shape=N_MAX_PEOPLE)
    person_logit = rule_logit_mean + rule_logit_sd * person_z

    mean_share = pm.Beta("mean_share", 2, 3)  # mean share of 1..100 a range covers
    concentration = pm.LogNormal("concentration", mu=np.log(5.0), sigma=1.0)
    family_weights = pm.Dirichlet("family_weights", a=np.full(N_FAMILIES, 2.0))
    rule_log_prior = pt.log(family_weights)[FAMILY] - np.log(FAMILY_SIZE[FAMILY])

    log_size_prior = ((mean_share * concentration - 1.0) * np.log(SHARE)
                      + ((1.0 - mean_share) * concentration - 1.0) * np.log1p(-SHARE))
    log_size_prior = log_size_prior - pt.logsumexp(log_size_prior)
    log_per_interval = log_size_prior - LOG_N_OF_SIZE  # prior of one interval of size L

    # Within rules and within ranges the posterior does not depend on the
    # person; only the rule-vs-range split does.
    log_rules = rule_log_prior[None, :] + rule_log_lik
    log_sizes = log_per_interval[None, :] + size_log_evidence
    z_rules = pt.logsumexp(log_rules, axis=1, keepdims=True)
    z_sizes = pt.logsumexp(log_sizes, axis=1, keepdims=True)
    post_rules = pt.exp(log_rules - z_rules)[cell_set]
    post_sizes = pt.exp(log_sizes - z_sizes)[cell_set]
    in_if_rule = (post_rules * cell_rule_member).sum(axis=1)  # (cells,)
    in_if_range = (post_sizes * cell_size_member).sum(axis=1)
    evidence_logit = (z_rules - z_sizes)[:, 0][cell_set]  # (cells,)

    # (people, cells): posterior share of rules for each person.
    w_rule = pm.math.sigmoid(person_logit[:, None] + evidence_logit[None, :])
    p_person = w_rule * in_if_rule[None, :] + (1.0 - w_rule) * in_if_range[None, :]
    known = pt.arange(N_MAX_PEOPLE) < n_people[0]
    p_average = (p_person * known[:, None]).sum(axis=0) / pt.maximum(n_people[0], 1)
    p_table = pt.concatenate([p_person, p_average[None, :]], axis=0)  # row -1 = average
    p_row = p_table[pt.switch(row_person < 0, N_MAX_PEOPLE, row_person), row_cell]

    p_yes = pm.Deterministic("p_yes", pt.clip(p_row, 1e-6, 1 - 1e-6))
    pm.Bernoulli("response", p=p_yes, observed=rating)


def participant_index(rows):
    """Participant of each row from session order (trial_index restarts at 0)."""
    if not rows or any(str(r.get("trial_index", "")).strip() in ("", "nan", "None") for r in rows):
        return np.full(len(rows), -1, dtype="int64"), 0
    person, current = [], -1
    for r in rows:
        if int(float(r["trial_index"])) == 0:
            current += 1
        person.append(max(current, 0))
    person = np.array(person, dtype="int64")
    n = int(person.max()) + 1
    if n > N_MAX_PEOPLE:
        return np.full(len(rows), -1, dtype="int64"), 0
    return person, n


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
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: i for i, c in enumerate(cells)}
    cell_s = np.array([c[0] for c in cells], dtype="int64")
    cell_t = np.array([c[1] - 1 for c in cells], dtype="int64")
    person, n_people = participant_index(rows)
    return {
        "rule_log_lik": set_log_lik[:, :N_RULES],
        "size_log_evidence": size_ev,
        "cell_set": cell_s,
        "cell_rule_member": RULE_MEMBERSHIP[:, cell_t].T.copy(),
        "cell_size_member": np.clip(size_member[cell_s, :, cell_t], 0.0, 1.0),
        "row_cell": np.array([cell_index[k] for k in keys], dtype="int64"),
        "row_person": person,
        "n_people": np.array([n_people], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
