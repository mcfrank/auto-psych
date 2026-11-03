# name: inside_vs_outside_range_gap
# description: Among multi-example sets, yes-rate for non-example targets inside the examples' [min, max] range minus yes-rate for targets outside it; observed above null means the model under-produces interval-like (range-filling) generalisation, below means it over-produces it.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ok = ~np.isnan(ex)
    n = ok.sum(axis=1)
    mn = np.nanmin(ex, axis=1)
    mx = np.nanmax(ex, axis=1)
    inset = np.where(ok, ex == t[:, None], False).any(axis=1)
    m = (n >= 2) & ~inset
    inside = (t >= mn) & (t <= mx)
    a, b = m & inside, m & ~inside
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
