# name: above_below_asym_max_not_last
# description: In sets of >=2 whose largest example is NOT shown last, yes-rate for targets 1-10 above the max minus yes-rate for targets 1-10 below the min; the model gives these sets no top credit (symmetric size-principle extension), so observed below null means people still restrict upward extension without the max-last cue, above null means they extend upward more than the model.
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
    g = (n >= 2) & (last != mx)
    up = g & (t > mx) & (t <= mx + 10)
    dn = g & (t < mn) & (t >= mn - 10)
    if up.sum() == 0 or dn.sum() == 0:
        return 0.0
    return float(r[up].mean() - r[dn].mean())
