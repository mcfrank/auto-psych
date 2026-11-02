# name: same_row_vs_adjacent_row_near
# description: Among non-example targets 1-5 away from the nearest example, yes rate when the target is in the same hundred-chart row (1-10, 11-20, ...) as that nearest example minus when it is in a different row; observed above null means chart-row boundaries matter more than the model's row-range bonus implies, below means less.
def _ex(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = (~np.isnan(E)).sum(1)
    isex = (E == t[:, None]).any(1)
    return E, t, r, n, isex

def _rate(r, m):
    return float(r[m].mean()) if m.sum() > 0 else 0.0

def test_statistic(df):
    E, t, r, n, isex = _ex(df)
    dist = np.abs(E - t[:, None])
    dmin = np.nanmin(dist, 1)
    j = np.nanargmin(np.where(np.isnan(dist), 1e9, dist), 1)
    near = E[np.arange(len(t)), j]
    row = lambda x: np.floor((x - 1) / 10)
    m = ~isex & (dmin >= 1) & (dmin <= 5)
    same = row(near) == row(t)
    return _rate(r, m & same) - _rate(r, m & ~same)
