# name: scale_extreme_target_yes
# description: Yes-rate for non-example targets at the ends of the scale (1-5 or 96-100) lying at least 15 beyond the example span; observed above null means people treat scale endpoints (e.g. 100, 1) as special inclusions the model's interval/rule hypotheses do not produce, below null means the model over-extends ranges to the scale ends.
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
    m = ((t <= 5) | (t >= 96)) & (dist >= 15) & ~is_ex
    return float(r[m].mean()) if m.any() else 0.0
