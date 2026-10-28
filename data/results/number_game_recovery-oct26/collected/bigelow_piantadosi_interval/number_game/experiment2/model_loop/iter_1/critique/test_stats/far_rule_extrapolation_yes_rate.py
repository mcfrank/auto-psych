# name: far_rule_extrapolation_yes_rate
# description: Among multi-example sets whose examples share a common divisor >= 2, yes-rate for non-example targets divisible by that divisor lying more than 10 beyond the examples' range; observed above the null means people extrapolate rules farther outside the examples' span than the model predicts, below means they confine rules to the examples' range more than the model does.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    k = (~np.isnan(ex)).sum(axis=1)
    exi = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.gcd.reduce(exi, axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(np.int64).to_numpy()
    is_ex = (ex == t[:, None]).any(axis=1)
    m = (k >= 2) & (g >= 2) & (t % np.maximum(g, 1) == 0) & ((t < lo - 10) | (t > hi + 10)) & ~is_ex
    r = df["rating"].astype(float).to_numpy()
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
