# name: extrapolation_above_minus_below
# description: Yes rate for non-example targets above the largest example minus for targets below the smallest example (multi-example sets); t_observed above null_mean means people extrapolate upward more readily than the model (a magnitude/direction asymmetry), below means downward more.
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
    mx = np.nanmax(sv, axis=1); mn = np.nanmin(sv, axis=1)
    base = (n >= 2) & ~is_ex
    return rate(base & (t > mx)) - rate(base & (t < mn))
