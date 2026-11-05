# name: large_target_yes_gap
# description: Yes-rate on non-example targets outside the examples' range [min,max] that lie above the max minus those below the min (multi-example sets); observed above null means people extend generalization upward more readily than downward beyond what the model's fitted bound saliences predict.
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(E), axis=1)
    inset = np.any(E == t[:, None], axis=1)
    lo = np.nanmin(E, axis=1); hi = np.nanmax(E, axis=1)
    ok = (n >= 2) & (~inset)
    a = ok & (t > hi)
    b = ok & (t < lo)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
