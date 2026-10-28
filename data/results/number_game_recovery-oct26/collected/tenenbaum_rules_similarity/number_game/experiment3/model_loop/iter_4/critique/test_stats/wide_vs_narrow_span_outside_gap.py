# name: wide_vs_narrow_span_outside_gap
# description: For sets of >=2 examples, yes-rate on non-example targets 1-10 outside the span when the span is >=40 minus when the span is <=15; observed above null means people scale outward generalization with the set's spread more than the model does, below null less.
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
    span = mx - mn
    near = (n >= 2) & (dist >= 1) & (dist <= 10) & ~is_ex
    a = near & (span >= 40)
    b = near & (span <= 15)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
