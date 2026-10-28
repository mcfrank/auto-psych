# name: inside_span_width_corr
# description: Correlation between example-span width (max-min) and yes response for non-example targets strictly inside the span (sets of >=2); observed below null means people reject interior gaps of wide spans more than the model (they distrust wide ranges / fall back on rules), above null means they fill wide spans more readily than the model.
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
    m = (n >= 2) & (t > mn) & (t < mx) & ~is_ex
    w = (mx - mn)[m]
    y = r[m]
    if m.sum() < 3 or w.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(w, y)[0, 1])
