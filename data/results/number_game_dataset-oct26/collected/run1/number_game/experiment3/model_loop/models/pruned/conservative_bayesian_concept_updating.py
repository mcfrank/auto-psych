"""Conservative Bayesian concept learning over familiar number concepts.

People generalize as an ideal Bayesian concept learner whose candidate concepts
are "ends in digit d", "even"/"odd", "multiples of k", a band of consecutive
decades, and "any number at all", weighing each by its prior and by the size
principle. The one distortion is conservatism: belief is updated less than
Bayes' rule prescribes, each example counting as only an inferred fraction of
a full random draw from the concept, so broad concepts retain more belief after
several examples than an ideal learner would leave them.
"""
import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMS = np.arange(1, 101)
DECADE = np.minimum(NUMS // 10, 9)  # 100 joins the 90s

_members, _family = [], []
for d in range(10):  # family 0: ends in digit d
    _members.append(NUMS % 10 == d); _family.append(0)
for r in range(2):  # family 1: even / odd
    _members.append(NUMS % 2 == r); _family.append(1)
for k in range(3, 51):  # family 2: multiples of k
    _members.append(NUMS % k == 0); _family.append(2)
for a in range(10):  # family 3: decade bands a..b
    for b in range(a, 10):
        _members.append((DECADE >= a) & (DECADE <= b)); _family.append(3)
_members.append(np.ones(100, dtype=bool)); _family.append(4)  # family 4: anything

MEMBER = np.array(_members, dtype="float64")  # (K, 100)
FAMILY = np.array(_family, dtype="int64")
N_FAM = 5
FAMILY_ONEHOT = np.eye(N_FAM)[FAMILY]  # (K, F)
LOG_FAMILY_COUNT = np.log(np.bincount(FAMILY, minlength=N_FAM).astype("float64"))[FAMILY]
LOG_SIZE = np.log(MEMBER.sum(axis=1))


def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    targets = np.array([int(r["target"]) for r in rows], dtype="int64")
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    consistent = np.array(
        [MEMBER[:, np.array(s) - 1].min(axis=1) for s in unique], dtype="float64"
    )
    return {
        "n_examples": np.array([len(s) for s in unique], dtype="float64"),
        # log of consistency: 0 if every example is in the concept, else -1e30
        "log_consistent": np.where(consistent > 0, 0.0, -1e30),
        "set_index": np.array([index[s] for s in sets], dtype="int64"),
        "target_member": MEMBER[:, targets - 1].T.copy(),  # (N, K)
        "rating": np.array([int(r.get("rating", 0) or 0) for r in rows], dtype="int64"),
    }


_PH = prepare_observed([{"number_set": "16,8", "target": "4", "rating": 0}])

with pm.Model() as model:
    n_examples = pm.Data("n_examples", _PH["n_examples"])
    log_consistent = pm.Data("log_consistent", _PH["log_consistent"])
    set_index = pm.Data("set_index", _PH["set_index"])
    target_member = pm.Data("target_member", _PH["target_member"])

    # Prior share of each concept family, spread evenly over its concepts.
    family_weight = pm.Dirichlet("family_weight", a=np.ones(N_FAM))
    log_prior = pt.dot(FAMILY_ONEHOT, pt.log(family_weight)) - LOG_FAMILY_COUNT

    # The distortion: each example counts as only `conservatism` of a draw
    # (1 = normative size principle, 0 = examples ignored beyond consistency).
    conservatism = pm.Beta("conservatism", alpha=2.0, beta=2.0)
    log_lik = -conservatism * n_examples[:, None] * LOG_SIZE[None, :]
    log_post = log_prior[None, :] + log_lik + log_consistent
    posterior = pt.special.softmax(log_post, axis=1)  # (sets, K)

    in_concept = (posterior[set_index] * target_member).sum(axis=1)

    # Response noise: on a fraction of trials an unconsidered, biased guess.
    lapse = pm.Beta("lapse", alpha=1.5, beta=8.0)
    guess_yes = pm.Beta("guess_yes", alpha=2.0, beta=2.0)
    p_yes = pm.Deterministic(
        "p_yes",
        pt.clip((1.0 - lapse) * in_concept + lapse * guess_yes, 1e-6, 1 - 1e-6),
    )

    rating = pm.Data("rating", _PH["rating"])
    pm.Bernoulli("response", p=p_yes, observed=rating)
