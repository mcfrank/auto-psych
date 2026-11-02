"""Win-stay/lose-sample running guess on the variable rule-or-range learner.

People take the examples one at a time, in the order shown, and hold a single
working guess about the program rather than the full posterior. After the first
example they pick a guess (a rule, a rule held within a range, or a magnitude
range of 1-100) from their Bayesian posterior: equally plausible rules, the size
principle, an uncertain stray rate with familiar strays, and a rule-versus-range
expectation that varies from judgement to judgement (as in
`variable_rule_range_expectation`). At each later example they keep the current
guess with an inferred probability ("stay") if it contains that example, and
otherwise re-pick from the posterior given the examples seen so far. The target
is endorsed if the final guess contains it. stay = 0 recovers the incumbent.

The guess distribution is tracked exactly: D_t = stay * c_t * D_{t-1} + R_t P_t,
with R_t = 1 - stay * sum(c_t D_{t-1}). Every concept's likelihood depends only on
(which shown examples it contains, its size), so all sums over intervals and
range-limited rules reduce to (subset, j) tables of size^-j built once.
"""


import math

import numpy as np
import pymc as pm
import pytensor.tensor as pt

NUMBERS = np.arange(1, 101)
MAX_N = 4


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
RULE_MEMBERSHIP = np.array([np.isin(NUMBERS, c) for c in RULES], dtype="float64")
RULE_SIZE = RULE_MEMBERSHIP.sum(axis=1)
INT_LO = np.array([m for m in NUMBERS for n in range(m, 101)])
INT_HI = np.array([n for m in NUMBERS for n in range(m, 101)])
INT_SIZE_IDX = INT_HI - INT_LO  # size - 1
RULE_MEMBER_RATE = RULE_MEMBERSHIP / RULE_SIZE[:, None]  # (rules, 100): pick a member of the rule
# Interval prior predictive: an interval drawn uniformly, then one of its members.
STRAY_INTERVAL_GEN = np.array([((INT_LO <= x) & (x <= INT_HI)) @ (1.0 / (INT_SIZE_IDX + 1)) for x in NUMBERS]) / len(INT_LO)


SUBSET_IN = np.array([[(m >> i) & 1 for i in range(MAX_N)] for m in range(2**MAX_N)], dtype="float64")
RULE_SIZE_IDX = RULE_SIZE.astype("int64") - 1

# Runs of sorted examples an interval can contain: (first, last) with first <= last.
RUNS = [(a, b) for a in range(MAX_N) for b in range(a, MAX_N)]
N_RUNS = len(RUNS)
RUN_INDEX = {r: i for i, r in enumerate(RUNS)}
RUN_SUBSET = np.array([sum(1 << i for i in range(a, b + 1)) for a, b in RUNS] + [0])


_N = np.arange(1, MAX_N + 1, dtype="float64")[:, None]
_J = np.arange(MAX_N + 1, dtype="float64")[None, :]
_VALID = _J <= _N
_N_MINUS_J = np.maximum(_N - _J, 0.0)
_SIZES = np.arange(1, 101, dtype="float64")
_GH_Z, _GH_W = np.polynomial.hermite_e.hermegauss(9)
_GH_W = _GH_W / _GH_W.sum()  # standard-normal quadrature
_SIZE_POW = _SIZES[:, None] ** -np.arange(MAX_N + 1)[None, :]  # (size, j)



def _betaln(a, b):
    return pt.gammaln(a) + pt.gammaln(b) - pt.gammaln(a + b)


def _elementary(weights):
    """Elementary symmetric polynomials e_0..e_4 over the last axis (4 examples)."""
    e = [pt.ones_like(weights[..., 0])] + [pt.zeros_like(weights[..., 0])] * MAX_N
    for i in range(MAX_N):
        w = weights[..., i]
        e = [e[0]] + [e[j] + w * e[j - 1] for j in range(1, MAX_N + 1)]
    return pt.stack(e, axis=-1)


N_SUB = 2**MAX_N
# Subset of shown positions restricted to the first t+1 examples.
SUBPREF = np.array([[s & ((1 << (t + 1)) - 1) for s in range(N_SUB)] for t in range(MAX_N)])
# BMASK[t0, t, s]: concept subset s contains shown positions t0+1..t (0-indexed).
BMASK = np.array(
    [[[float(all((s >> p) & 1 for p in range(t0 + 1, t + 1))) for s in range(N_SUB)] for t in range(MAX_N)] for t0 in range(MAX_N)]
)
RULE_POW = RULE_SIZE[:, None] ** -np.arange(MAX_N + 1)[None, :]  # (rules, j)
N_INT = len(INT_LO)

with pm.Model() as model:
    example_idx = pm.Data("example_idx", np.zeros((1, MAX_N), dtype="int64"))  # (sets, 4): x - 1, shown order
    example_mask = pm.Data("example_mask", np.zeros((1, MAX_N)))
    rule_sub = pm.Data("rule_sub", np.zeros((1, N_RULES), dtype="int64"))  # shown-order subset in each rule
    set_n = pm.Data("set_n", np.ones(1, dtype="int64"))  # (sets,): n
    fin_mask = pm.Data("fin_mask", np.zeros((1, MAX_N, N_SUB)))  # (sets, t, s): s contains t+1..n-1, t < n
    int_pow = pm.Data("int_pow", np.zeros((1, N_SUB, MAX_N + 1)))  # (sets, s, j): intervals, sum size^-j
    rr_pow = pm.Data("rr_pow", np.zeros((1, N_SUB, MAX_N + 1)))  # same, range-limited rules (x multiplicity)
    cell_set = pm.Data("cell_set", np.zeros(1, dtype="int64"))
    cell_rule_member = pm.Data("cell_rule_member", np.zeros((1, N_RULES)))
    cell_int_pow = pm.Data("cell_int_pow", np.zeros((1, N_SUB, MAX_N + 1)))  # intervals containing the target
    cell_rr_pow = pm.Data("cell_rr_pow", np.zeros((1, N_SUB, MAX_N + 1)))
    row_cell = pm.Data("row_cell", np.zeros(1, dtype="int64"))
    rating = pm.Data("rating", np.zeros(1, dtype="int64"))

    lambda_mu = pm.Normal("lambda_mu", mu=0.0, sigma=1.5)
    lambda_sd = pm.HalfNormal("lambda_sd", sigma=1.0)
    lam_k = pm.math.sigmoid(lambda_mu + lambda_sd * _GH_Z)  # (K,)
    alpha_logit = pm.Normal("alpha_logit", mu=2.2, sigma=1.5)
    alpha = pm.Deterministic("alpha", pm.math.sigmoid(alpha_logit))
    log_kappa = pm.Normal("log_kappa", mu=np.log(10.0), sigma=1.0)
    kappa = pm.Deterministic("kappa", pt.exp(log_kappa))
    u, v = alpha * kappa, (1 - alpha) * kappa
    mix_logit = pm.Normal("mix_logit", mu=0.0, sigma=1.5)
    mix = pm.Deterministic("mix", pm.math.sigmoid(mix_logit))
    rho_logit = pm.Normal("rho_logit", mu=-2.0, sigma=1.0)
    rho = pm.Deterministic("rho", pm.math.sigmoid(rho_logit))
    # Probability of keeping a working guess that contains the new example.
    stay_logit = pm.Normal("stay_logit", mu=-1.0, sigma=1.5)
    stay = pm.Deterministic("stay", pm.math.sigmoid(stay_logit))

    q = (1 - mix) * 0.01 + mix * (RULE_MEMBERSHIP.sum(axis=0) / RULE_MEMBERSHIP.sum())
    inv_q = example_mask / q[example_idx]  # (sets, 4)
    c_table = pt.where(_VALID, pt.exp(_betaln(u + _J, v + _N_MINUS_J) - _betaln(u, v)), 0.0)  # (n, j)
    e_sub = _elementary(SUBSET_IN[None, :, :] * inv_q[:, None, :])  # (sets, s, j) over shown positions
    # Prior weight per unit of each kind, at each quadrature node: (K, 3) = rule, interval, restricted.
    pk = pt.stack([lam_k * (1 - rho) / N_RULES, (1 - lam_k) / N_INT, lam_k * rho / (N_RULES * N_INT)], axis=1)

    S_idx = pt.arange(rule_sub.shape[0])[:, None]
    E, Wi, Wr, Lr = [], [], [], []
    for t in range(MAX_N):  # prefix of the first t+1 shown examples
        E_t = e_sub[:, SUBPREF[t], :] * c_table[t][None, None, :]  # (sets, s, j)
        E.append(E_t)
        Wi.append((E_t * int_pow).sum(axis=2))  # (sets, s)
        Wr.append((E_t * rr_pow).sum(axis=2))
        Lr.append((E_t[S_idx, rule_sub, :] * RULE_POW[None, :, :]).sum(axis=2))  # (sets, rules)

    def kind_mass(t, m_sub):
        """Prior x prefix-t likelihood mass of concepts whose subset passes m_sub (sets, s): (sets, K)."""
        rule_m = m_sub[S_idx, rule_sub] if m_sub.ndim == 2 else m_sub[rule_sub]
        sums = pt.stack([(Lr[t] * rule_m).sum(axis=1), (Wi[t] * m_sub).sum(axis=-1), (Wr[t] * m_sub).sum(axis=-1)], axis=1)
        return pt.dot(sums, pk.T)

    ones = pt.ones((N_SUB,))
    Z = [kind_mass(t, pt.as_tensor_variable(np.ones(N_SUB))) for t in range(MAX_N)]  # (sets, K)
    R = [pt.ones_like(Z[0])]
    for t in range(1, MAX_N):
        S_t = sum(R[t0] * stay ** (t - 1 - t0) * kind_mass(t0, pt.as_tensor_variable(BMASK[t0, t])) / Z[t0] for t0 in range(t))
        R.append(1 - stay * S_t)

    # Final guess contains the target.
    n_c = set_n[cell_set]
    in_k = 0.0
    for t in range(MAX_N):
        m_c = fin_mask[cell_set, t, :]  # (cells, s)
        E_c = E[t][cell_set]  # (cells, s, j)
        rule_m = m_c[pt.arange(cell_set.shape[0])[:, None], rule_sub[cell_set]]
        sums = pt.stack(
            [
                (Lr[t][cell_set] * rule_m * cell_rule_member).sum(axis=1),
                ((E_c * cell_int_pow).sum(axis=2) * m_c).sum(axis=1),
                ((E_c * cell_rr_pow).sum(axis=2) * m_c).sum(axis=1),
            ],
            axis=1,
        )
        expo = pt.maximum(n_c - 1 - t, 0).astype("float64")
        in_k = in_k + (R[t] / Z[t])[cell_set] * (stay ** expo)[:, None] * pt.dot(sums, pk.T)
    in_concept = pt.dot(in_k, _GH_W)
    p_yes = pm.Deterministic("p_yes", pt.clip(in_concept, 1e-9, 1 - 1e-9)[row_cell])
    pm.Bernoulli("response", p=p_yes, observed=rating)


def _restricted_blocks():
    """Every proper, non-empty block of a rule's sorted members a range of 1-100 can cut out."""
    rule, lo, hi, size, mult = [], [], [], [], []
    for r, c in enumerate(RULES):
        m = np.sort(np.array(c))
        k = len(m)
        gaps = np.diff(np.concatenate([[0], m, [101]]))  # gaps[i]: choices of boundary before member i
        for a in range(k):
            for b in range(a, k):
                if a == 0 and b == k - 1:
                    continue  # the whole rule is the plain rule
                rule.append(r)
                lo.append(m[a])
                hi.append(m[b])
                size.append(b - a + 1)
                mult.append(gaps[a] * gaps[b + 1])
    return (np.array(rule), np.array(lo), np.array(hi), np.array(size) - 1, np.array(mult, dtype="float64"))


BLK_RULE, BLK_LO, BLK_HI, BLK_SIZE_IDX, BLK_MULT = _restricted_blocks()




def prepare_observed(rows):
    sets = [tuple(int(x) for x in str(r["number_set"]).split(",")) for r in rows]
    unique = list(dict.fromkeys(sets))
    index = {s: i for i, s in enumerate(unique)}
    S = len(unique)
    example_idx = np.zeros((S, MAX_N), dtype="int64")
    example_mask = np.zeros((S, MAX_N))
    set_n = np.zeros(S, dtype="int64")
    fin_mask = np.zeros((S, MAX_N, N_SUB))
    int_count = np.zeros((S, N_SUB, 100))
    rr_count = np.zeros((S, N_SUB, 100))
    set_int_sub, set_blk_sub = [], []
    for i, s in enumerate(unique):
        ex = np.array(s)
        n = len(ex)
        set_n[i] = n
        example_idx[i, :n] = ex - 1
        example_mask[i, :n] = 1.0
        for t in range(n):
            fin_mask[i, t] = [float(all((sb >> p) & 1 for p in range(t + 1, n))) for sb in range(N_SUB)]
        isub = np.zeros(len(INT_LO), dtype="int64")
        bsub = np.zeros(len(BLK_RULE), dtype="int64")
        for e_i, x in enumerate(ex):
            isub += ((INT_LO <= x) & (x <= INT_HI)).astype("int64") << e_i
            inb = (RULE_MEMBERSHIP[BLK_RULE, x - 1] > 0) & (BLK_LO <= x) & (x <= BLK_HI)
            bsub += inb.astype("int64") << e_i
        set_int_sub.append(isub)
        set_blk_sub.append(bsub)
        np.add.at(int_count[i], (isub, INT_SIZE_IDX), 1.0)
        np.add.at(rr_count[i], (bsub, BLK_SIZE_IDX), BLK_MULT)
    keys = [(index[s], int(r["target"])) for s, r in zip(sets, rows)]
    cells = list(dict.fromkeys(keys))
    cell_index = {c: j for j, c in enumerate(cells)}
    cell_int_count = np.zeros((len(cells), N_SUB, 100))
    cell_rr_count = np.zeros((len(cells), N_SUB, 100))
    for j, (i, t) in enumerate(cells):
        hit = (RULE_MEMBERSHIP[BLK_RULE, t - 1] > 0) & (BLK_LO <= t) & (t <= BLK_HI)
        np.add.at(cell_rr_count[j], (set_blk_sub[i][hit], BLK_SIZE_IDX[hit]), BLK_MULT[hit])
        contains = (INT_LO <= t) & (t <= INT_HI)
        np.add.at(cell_int_count[j], (set_int_sub[i][contains], INT_SIZE_IDX[contains]), 1.0)
    return {
        "example_idx": example_idx,
        "example_mask": example_mask,
        "rule_sub": (
            (RULE_MEMBERSHIP[:, example_idx].transpose(1, 0, 2) * example_mask[:, None, :]) @ (2 ** np.arange(MAX_N))
        ).astype("int64"),
        "set_n": set_n,
        "fin_mask": fin_mask,
        "int_pow": int_count @ _SIZE_POW,
        "rr_pow": rr_count @ _SIZE_POW,
        "cell_set": np.array([i for i, _ in cells], dtype="int64"),
        "cell_rule_member": np.stack([RULE_MEMBERSHIP[:, t - 1] for _, t in cells]),
        "cell_int_pow": cell_int_count @ _SIZE_POW,
        "cell_rr_pow": cell_rr_count @ _SIZE_POW,
        "row_cell": np.array([cell_index[c] for c in keys], dtype="int64"),
        "rating": np.array([int(r["rating"]) for r in rows], dtype="int64"),
    }
