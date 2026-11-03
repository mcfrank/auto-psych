# name: inside_gap_width_effect
# description: Among non-member targets strictly inside the range of 3-4-example sets, Pearson correlation between rating and the width of the gap (between consecutive sorted examples) that contains the target; under the model's intervals every inside target is covered equally, so observed below null means people fill wide gaps between examples less than narrow ones (local similarity), above null the reverse.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    s = np.sort(np.where(np.isnan(ex), np.inf, ex), axis=1)
    lo = s[:, 0]; hi = np.nanmax(ex, axis=1)
    member = np.nansum(ex == t[:, None], axis=1) > 0
    below = np.where(s <= t[:, None], s, -np.inf).max(axis=1)
    above = np.where(s >= t[:, None], s, np.inf).min(axis=1)
    keep = (n >= 3) & (t > lo) & (t < hi) & ~member
    if keep.sum() < 3:
        return 0.0
    g = (above - below)[keep]; y = r[keep]
    if np.std(g) == 0 or np.std(y) == 0:
        return 0.0
    return float(np.corrcoef(g, y)[0, 1])
