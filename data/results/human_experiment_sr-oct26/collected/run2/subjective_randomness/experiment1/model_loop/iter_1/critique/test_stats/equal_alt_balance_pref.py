# name: equal_alt_balance_pref
# description: Among trials whose two sequences have identical alternation rates, the proportion choosing the sequence with the more balanced H/T count (ties in balance excluded); the model predicts 0.5, so t_observed > null_mean means people use H/T balance, which the model ignores.
def _feat(s, fn):
    u = pd.unique(s)
    m = {x: fn(x) for x in u}
    return s.map(m).to_numpy(dtype=float)


def _alt(x):
    return sum(a != b for a, b in zip(x, x[1:])) / (len(x) - 1)


def _imb(x):
    return abs(x.count("H") - len(x) / 2.0) / len(x)


def _longest(x):
    best = cur = 1
    for a, b in zip(x, x[1:]):
        cur = cur + 1 if a == b else 1
        best = max(best, cur)
    return best / len(x)


def _periodic(x):
    n = len(x)
    for p in range(1, n // 2 + 1):
        if all(x[i] == x[i - p] for i in range(p, n)):
            return 1.0
    return 0.0


def _ols_coef(y, cols, k):
    X = np.column_stack([np.ones(len(y))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return float(b[k])


def test_statistic(df):
    aa = _feat(df["sequence_a"], _alt); ab = _feat(df["sequence_b"], _alt)
    ia = _feat(df["sequence_a"], _imb); ib = _feat(df["sequence_b"], _imb)
    m = (np.abs(aa - ab) < 1e-9) & (np.abs(ia - ib) > 1e-9)
    if m.sum() == 0:
        return 0.5
    y = df["chose_left"].to_numpy(dtype=float)[m]
    left_bal = (ia[m] < ib[m]).astype(float)
    return float(np.mean(y * left_bal + (1 - y) * (1 - left_bal)))
