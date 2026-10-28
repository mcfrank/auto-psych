# name: nonparity_rule_far_gap
# description: In sets of >=2, among targets >10 outside the span, yes-rate for targets that share a non-parity rule (multiple of 3-12, last digit, contains digit, prime, square, cube) with all examples minus targets that share none; observed above null means people generalize by specific arithmetic rules far from the examples more than the model, below null means the model over-credits such rules.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    N = np.arange(101)
    rules = [N % k == 0 for k in range(3, 13)]
    rules += [N % 10 == d for d in range(1, 10)]
    s = np.array([str(i) for i in N])
    rules += [np.char.find(s, str(d)) >= 0 for d in range(1, 10)]
    pr = np.array([i > 1 and all(i % j for j in range(2, int(i ** 0.5) + 1)) for i in N])
    rules += [pr, np.isin(N, [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]), np.isin(N, [1, 8, 27, 64])]
    M = np.array(rules)
    M[:, 0] = True
    Xi = np.nan_to_num(X, nan=0).astype(int)
    cons = M[:, Xi].all(axis=2)
    hit = (cons & M[:, t]).any(axis=0)
    dist = np.maximum(mn - t, t - mx)
    g = (n >= 2) & (dist > 10)
    a = g & hit
    b = g & ~hit
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
