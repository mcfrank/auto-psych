# name: arithmetic_rule_outside_boost
# description: Sets of >=2 all-even examples: yes-rate for even minus odd targets outside the example range; observed above null means people use parity rules more than the model.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    alleven = np.all(np.mod(np.nan_to_num(ex, nan=2.0), 2) == 0, axis=1) & (n >= 2)
    out = alleven & ((t < lo) | (t > hi))
    return rate(out & (np.mod(t, 2) == 0)) - rate(out & (np.mod(t, 2) == 1))
