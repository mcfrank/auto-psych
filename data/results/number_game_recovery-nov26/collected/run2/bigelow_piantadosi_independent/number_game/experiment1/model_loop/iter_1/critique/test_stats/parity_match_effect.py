# name: parity_match_effect
# description: Among multi-example sets whose examples all share parity, yes rate for same-parity targets minus opposite-parity targets; observed > null means people weight parity more than the model, observed < null means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    valid = ~np.isnan(s)
    n = valid.sum(axis=1)
    p = np.where(valid, s % 2, np.nan)
    pmin = np.nanmin(p, axis=1); pmax = np.nanmax(p, axis=1)
    homog = (n >= 2) & (pmin == pmax)
    t = df["target"].astype(int).to_numpy()
    same = (t % 2) == pmin
    y = df["rating"].to_numpy(dtype=float)
    a = homog & same; b = homog & ~same
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
