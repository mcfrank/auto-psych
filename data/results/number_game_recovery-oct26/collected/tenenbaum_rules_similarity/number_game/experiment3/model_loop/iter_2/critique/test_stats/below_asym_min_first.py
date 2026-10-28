# name: below_asym_min_first
# description: In sets of >=2, yes-rate for targets 1-10 below the min when the smallest example is shown first minus when it is not; the model has no first-shown cue, so observed below null means a first-shown minimum marks the range bottom (restricts downward extension), above null means it invites downward extension.
def test_statistic(df):
    X = df["number_set"].astype(str).str.replace(" ", "", regex=False).str.split(",", expand=True)
    X = X.apply(pd.to_numeric, errors="coerce").values.astype(float)
    n = (~np.isnan(X)).sum(axis=1)
    mx = np.nanmax(X, axis=1)
    mn = np.nanmin(X, axis=1)
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    is_ex = (X == t[:, None]).any(axis=1)
    first = X[:, 0]
    dn = (n >= 2) & (t < mn) & (t >= mn - 10)
    a = dn & (first == mn)
    b = dn & (first != mn)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
