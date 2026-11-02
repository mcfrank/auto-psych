# name: mult5_set_mult5_target_gap
# description: In sets of >=2 examples that are all multiples of 5, yes rate for non-example multiple-of-5 targets minus for non-multiple-of-5 targets; t_observed above null_mean means people use the "multiples of 5/10" rule more than the model, below means less.
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
    all5 = np.where(v, s % 5 == 0, True).all(axis=1)
    base = (n >= 2) & all5 & ~is_ex
    return rate(base & (t % 5 == 0)) - rate(base & (t % 5 != 0))
