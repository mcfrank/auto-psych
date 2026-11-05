# name: four_example_above_minus_below_range
# description: On 4-example sets, yes rate for non-example targets above the examples' maximum minus the rate for targets below their minimum; observed above null means people generalize upward (to larger numbers) more than the model, whose concepts extend symmetrically.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).reindex(columns=range(4)).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(E)
    n = valid.sum(axis=1)
    Ez = np.where(valid, E, 0.0)
    lo = np.where(valid, E, np.inf).min(axis=1)
    hi = np.where(valid, E, -np.inf).max(axis=1)
    is_ex = (valid & (E == t[:, None])).any(axis=1)
    odd = (np.where(valid, Ez % 2, 1) == 1).all(axis=1)
    even = (np.where(valid, Ez % 2, 0) == 0).all(axis=1)
    def rate(m):
        return float(y[m].mean()) if m.sum() > 0 else 0.0
    base = (n == 4) & ~is_ex
    return rate(base & (t > hi)) - rate(base & (t < lo))
