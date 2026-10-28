# name: min_first_below_contrast
# description: Among sets of >=2 examples, yes-rate for targets 1-10 below the minimum when the minimum is shown first minus when it is shown later; observed less negative than null means the incumbent's first-shown-bottom credit exaggerates the opening-order effect, more negative means people show a stronger order effect than the model.
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
    first_min = X[:, 0] == mn
    below = (mn - t >= 1) & (mn - t <= 10) & (n >= 2)
    a = below & first_min
    b = below & ~first_min
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
