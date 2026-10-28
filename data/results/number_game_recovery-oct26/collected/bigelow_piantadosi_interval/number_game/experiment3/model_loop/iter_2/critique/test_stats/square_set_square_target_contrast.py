# name: square_set_square_target_contrast
# description: For sets of 2-4 examples that are all perfect squares above 1 (e.g. 4,64,25,81; 81,25; 81,9), yes rate on square non-example targets minus non-square targets; observed above null_mean means people generalize to the "squares" rule more than the model predicts, below means less.
def test_statistic(df):
    sq = np.array([4, 9, 16, 25, 36, 49, 64, 81, 100], dtype=float)
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    alls = np.all(np.isin(ex, sq) | np.isnan(ex), axis=1) & (n >= 2)
    sel = alls & ~np.any(ex == t[:, None], axis=1)
    ts = np.isin(t, np.append(sq, 1.0))
    a, b = sel & ts, sel & ~ts
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
