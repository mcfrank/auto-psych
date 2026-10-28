# name: ruleless_inside_minus_outside_range
# description: Among multi-example sets with no common divisor (gcd = 1), yes-rate for non-example targets inside the examples' [min, max] range minus that for targets outside it; observed above the null means people's range-based (similarity) generalization is stronger/sharper-edged than the model predicts, below means weaker.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    k = (~np.isnan(ex)).sum(axis=1)
    exi = np.nan_to_num(ex, nan=0.0).astype(np.int64)
    g = np.gcd.reduce(exi, axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    t = df["target"].astype(float).to_numpy()
    is_ex = (ex == t[:, None]).any(axis=1)
    base = (k >= 2) & (g == 1) & ~is_ex
    inside = base & (t >= lo) & (t <= hi)
    outside = base & ((t < lo) | (t > hi))
    r = df["rating"].astype(float).to_numpy()
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
