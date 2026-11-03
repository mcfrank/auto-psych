# name: two_example_interpolation_vs_extrapolation
# description: In two-example sets whose examples are >= 6 apart, yes rate for non-example targets strictly between them minus yes rate for targets outside their range but within one gap-length of the nearer example; observed > null means people interpolate between examples more sharply than they extrapolate beyond them, beyond what the model's interval hypotheses produce; observed < null means extrapolation is stronger than modelled.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    d = dist.min(axis=1)
    lo = np.where(valid, ex, np.inf).min(axis=1)
    hi = np.where(valid, ex, -np.inf).max(axis=1)
    gap = hi - lo
    base = (n == 2) & (gap >= 6) & (d > 0)
    a = base & (t > lo) & (t < hi)
    b = base & ((t < lo) | (t > hi)) & (d <= gap)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
