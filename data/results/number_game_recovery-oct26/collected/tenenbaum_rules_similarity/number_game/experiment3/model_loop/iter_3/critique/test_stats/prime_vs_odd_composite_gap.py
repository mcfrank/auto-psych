# name: prime_vs_odd_composite_gap
# description: In sets of >=2 examples that are all primes, yes-rate for prime non-example targets minus odd composite targets (excluding 1); observed above null means people favour primality more sharply than the model's prime-vs-odd competition predicts, below null means people treat such sets as 'odd numbers' more than the model.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    N = np.arange(101)
    isp = np.ones(101, dtype=bool); isp[:2] = False
    for k in range(2, 11):
        isp[k * k::k] = False
    Xf = np.nan_to_num(X, nan=0.0).astype(np.int64)
    allp = (isp[Xf] | np.isnan(X)).all(axis=1) & (n >= 2)
    tp = isp[t]
    oddc = (np.mod(t, 2) == 1) & ~tp & (t > 1)
    a = allp & ~is_ex & tp
    b = allp & ~is_ex & oddc
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
