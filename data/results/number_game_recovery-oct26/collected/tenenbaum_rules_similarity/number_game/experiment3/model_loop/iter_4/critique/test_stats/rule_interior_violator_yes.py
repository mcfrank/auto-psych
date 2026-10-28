# name: rule_interior_violator_yes
# description: Yes-rate on interior (inside min-max span) non-example targets that violate a non-parity arithmetic rule (power series, multiple of 3-12, ends-in-d) shared by all of a >=2-example set's examples; observed above null means people let range-based similarity override such a rule inside the span more than the model, below null that the rule is held more strictly.
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
    Xi = np.nan_to_num(X, nan=0).astype(int)
    valid = ~np.isnan(X)
    rules = [[2, 4, 8, 16, 32, 64], [3, 9, 27, 81], [4, 16, 64], [5, 25], [6, 36], [7, 49], [8, 64], [9, 81], [10, 100]]
    M = np.zeros((len(rules) + 19, 101), dtype=bool)
    for i, rr in enumerate(rules):
        M[i, rr] = True
    k = len(rules)
    nums = np.arange(101)
    for m in range(3, 13):
        M[k] = (nums % m == 0) & (nums > 0); k += 1
    for dd in range(1, 10):
        M[k] = (nums % 10 == dd); k += 1
    memb = M[:, Xi]  # rules x rows x slots
    allm = (memb | ~valid[None, :, :]).all(axis=2)  # rules x rows
    tm = M[:, t]  # rules x rows
    viol = (allm & ~tm).any(axis=0)
    a = (n >= 2) & (t > mn) & (t < mx) & ~is_ex & viol
    if a.sum() == 0:
        return 0.0
    return float(r[a].mean())
