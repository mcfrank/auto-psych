# name: far_outside_span_yes
# description: Yes-rate for targets more than 20 away from the nearest edge of the example span (sets of >=2); observed above null means people generalize far beyond the examples (broad rule-like concepts) more than the model predicts, below null means the model over-produces far generalization.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    last = X[np.arange(len(X)), n - 1]
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    dist = np.maximum(mn - t, t - mx)
    m = (n >= 2) & (dist > 20)
    return float(r[m].mean()) if m.any() else 0.0
