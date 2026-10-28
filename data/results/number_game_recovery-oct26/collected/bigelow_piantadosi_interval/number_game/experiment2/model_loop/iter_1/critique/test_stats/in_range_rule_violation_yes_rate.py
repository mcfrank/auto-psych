# name: in_range_rule_violation_yes_rate
# description: Among multi-example sets whose examples share a common divisor >= 2, yes-rate for targets inside the examples' [min, max] range that are NOT divisible by that divisor; observed above the null means people generalize by numeric proximity (ranges) over a clear rule more than the model does, below means the model over-weights ranges against rules.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    k = (~np.isnan(ex)).sum(axis=1)
    exi = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.gcd.reduce(exi, axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(np.int64).to_numpy()
    m = (k >= 2) & (g >= 2) & (t % np.maximum(g, 1) != 0) & (t >= lo) & (t <= hi)
    r = df["rating"].astype(float).to_numpy()
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
