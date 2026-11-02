# name: odd_set_far_odd_yes_rate
# description: In all-odd sets (>=2 examples), yes rate for odd non-example targets >10 from every example and sharing no final digit with any example; t_observed above null_mean means people generalize the "odd numbers" rule more than the model, below means less.
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
    par = np.where(v, s % 2, np.nan)
    allodd = np.nanmin(par, axis=1) == 1
    ld = (np.where(v, s % 10, -1) == (t % 10)[:, None]).any(axis=1)
    m = (n >= 2) & allodd & ~is_ex & (t % 2 == 1) & (d > 10) & ~ld
    return rate(m)
