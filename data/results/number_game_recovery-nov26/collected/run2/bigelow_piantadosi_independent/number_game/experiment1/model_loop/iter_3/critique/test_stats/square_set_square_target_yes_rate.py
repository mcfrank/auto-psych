# name: square_set_square_target_yes_rate
# description: In sets of >=2 examples that are all perfect squares, yes rate for non-example perfect-square targets; t_observed above null_mean means people generalize the "squares" rule more than the model, below means less.
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
    sq = np.zeros(102, dtype=bool); sq[[k * k for k in range(1, 11)]] = True
    si = np.where(v, s, 0).astype(int)
    allsq = np.where(v, sq[si], True).all(axis=1)
    m = (n >= 2) & allsq & ~is_ex & sq[t.astype(int)]
    return rate(m)
