# name: recency_last_vs_second_example
# description: For 3-4 example sets and non-example targets within 10 of some example, yes rate when the nearest example is the LAST shown minus when it is the SECOND shown; observed above null_mean means a recency advantage for the final example that the model (which only singles out the first example) fails to produce, below means the opposite.
def test_statistic(df):
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = (s.str.count(",") + 1).to_numpy()
    t = df["target"].astype(int).to_numpy()[:, None]
    d = np.abs(ex - t)
    d = np.where(np.isnan(d), 1e9, d)
    dmin = d.min(axis=1)
    nearest = d.argmin(axis=1)
    ties = (d == dmin[:, None]).sum(axis=1) > 1
    r = df["rating"].to_numpy().astype(float)
    base = (n >= 3) & (dmin >= 1) & (dmin <= 10) & ~ties
    last = base & (nearest == n - 1)
    second = base & (nearest == 1)
    if last.sum() == 0 or second.sum() == 0:
        return 0.0
    return float(r[last].mean() - r[second].mean())
