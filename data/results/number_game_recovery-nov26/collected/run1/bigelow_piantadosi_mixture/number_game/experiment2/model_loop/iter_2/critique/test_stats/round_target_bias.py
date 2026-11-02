# name: round_target_bias
# description: Among sets with no multiple-of-10 example, yes-rate for multiple-of-10 targets minus other targets outside the example range; observed above null means people over-accept round numbers relative to the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    has10 = np.any(np.mod(np.nan_to_num(ex, nan=1.0), 10) == 0, axis=1)
    out = (~has10) & ((t < lo) | (t > hi))
    return rate(out & (np.mod(t, 10) == 0)) - rate(out & (np.mod(t, 10) != 0))
