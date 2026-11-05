# name: round_set_near_miss_rate
# description: In sets of 2+ examples that are all multiples of 10, yes-rate for non-multiple-of-10 targets within 10 of the nearest example; observed < null means people reject near-but-off-rule targets more strictly than the model's (sharpened) magnitude similarity allows, observed > null means the model is too strict.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t1 = df["target"].astype(int).to_numpy().astype(float)
    t = t1[:, None]
    pres = ~np.isnan(ex)
    n = pres.sum(1)
    e = np.where(pres, ex, 1e9)
    e1 = np.where(pres, ex, 1.0)
    r = df["rating"].to_numpy(dtype=float)
    d = np.abs(t - e).min(1)
    digmatch = ((np.mod(t, 10) == np.mod(e1, 10)) & pres).any(1)
    def diff(a, b):
        if a.sum() == 0 or b.sum() == 0:
            return 0.0
        return float(r[a].mean() - r[b].mean())
    allr = (~pres | (np.mod(e1, 10) == 0)).all(1) & (n >= 2)
    a = allr & (np.mod(t1, 10) != 0) & (d <= 10)
    if a.sum() == 0:
        return 0.0
    return float(r[a].mean())
