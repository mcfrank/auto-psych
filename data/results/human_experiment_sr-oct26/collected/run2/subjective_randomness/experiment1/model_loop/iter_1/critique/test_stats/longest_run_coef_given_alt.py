# name: longest_run_coef_given_alt
# description: OLS coefficient of chose_left on the longest-run-length difference (normalised by length, right minus left), controlling for alternation difference and squared-distance difference; t_observed > null_mean means long streaks are penalised beyond what alternation rate explains.
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
    ra = _feat(df["sequence_a"], _longest); rb = _feat(df["sequence_b"], _longest)
    y = df["chose_left"].to_numpy(dtype=float)
    q = (ab - 0.6) ** 2 - (aa - 0.6) ** 2
    return _ols_coef(y, [rb - ra, aa - ab, q], 1)
