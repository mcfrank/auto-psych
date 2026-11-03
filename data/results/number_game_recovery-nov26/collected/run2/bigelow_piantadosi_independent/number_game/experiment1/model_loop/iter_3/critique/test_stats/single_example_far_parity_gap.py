# name: single_example_far_parity_gap
# description: In single-example sets, yes rate for targets >10 from the example with the same parity minus with the opposite parity; t_observed above null_mean means one example triggers parity generalization more than the model predicts, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    v = ~np.isnan(s); n = v.sum(axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].to_numpy(dtype=float)
    sv = np.where(v, s, np.nan)
    is_ex = (np.where(v, s, -999) == t[:, None]).any(axis=1)
    d = np.nanmin(np.abs(sv - t[:, None]), axis=1)
    def rate(m):
        return float(r[m].mean()) if m.sum() > 0 else 0.0
    base = (n == 1) & ~is_ex & (d > 10)
    sp = (t % 2) == (s[:, 0] % 2)
    return rate(base & sp) - rate(base & ~sp)
