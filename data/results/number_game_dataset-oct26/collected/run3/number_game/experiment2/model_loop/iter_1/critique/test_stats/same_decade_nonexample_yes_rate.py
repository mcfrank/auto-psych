# name: same_decade_nonexample_yes_rate
# description: Among sets of 1-2 examples, yes-rate on non-example targets in the same decade (same tens digit, 100 counted with 90s) as some example; observed > model means people generalize to decade-mates after little evidence more than the model predicts, observed < model means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = (~np.isnan(ex)).sum(axis=1)
    t = df["target"].astype(float).to_numpy()
    dec_e = np.floor(np.minimum(ex, 99) / 10.0)
    dec_t = np.floor(np.minimum(t, 99) / 10.0)[:, None]
    same = (dec_e == dec_t).any(axis=1)
    isex = (ex == t[:, None]).any(axis=1)
    m = (n <= 2) & same & ~isex
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy(dtype=float)[m].mean())
