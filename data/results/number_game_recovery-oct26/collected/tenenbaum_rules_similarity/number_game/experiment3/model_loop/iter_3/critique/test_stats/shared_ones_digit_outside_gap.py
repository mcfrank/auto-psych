# name: shared_ones_digit_outside_gap
# description: In sets of >=2 examples, among non-example targets 1-20 outside the span, yes-rate for targets sharing a ones digit with some example minus those sharing none; observed above null means people use digit-level similarity to examples beyond what the model's ends-in-d rules (which need all examples to share the digit) capture.
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
    od = np.mod(X, 10)
    share = (od == np.mod(t, 10)[:, None]).any(axis=1)
    near = (dist >= 1) & (dist <= 20) & (n >= 2) & ~is_ex
    a = near & share
    b = near & ~share
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
