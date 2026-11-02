# name: mult4_set_mult4_vs_even_gap
# description: Among sets of >=2 examples that are all multiples of 4, yes rate for non-example multiple-of-4 targets minus yes rate for even non-multiple-of-4 targets; t_observed below null_mean means people generalize to the broader 'even' concept more (less specific) than the model's size principle predicts, above means more specific.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    m4 = np.all(np.isnan(ex) | (ex % 4 == 0), axis=1) & (n >= 2)
    isex = np.any(ex == t[:, None], axis=1)
    r = df["rating"].to_numpy(dtype=float)
    a = m4 & ~isex & (t % 4 == 0)
    b = m4 & ~isex & (t % 2 == 0) & (t % 4 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
