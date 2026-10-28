# name: round_target_outside_gap
# description: Among targets outside the example span (all sets) whose example set contains no multiple of 10, yes-rate for round targets (multiples of 10) minus non-round targets; observed above null means people favour round numbers beyond what the model's rules/decade ranges produce, below null means the model over-credits round targets.
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
    has10 = (np.mod(np.nan_to_num(X, nan=1.0), 10) == 0).any(axis=1)
    out = ((t < mn) | (t > mx)) & ~has10
    a = out & (t % 10 == 0)
    b = out & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
