# name: size_principle_four_vs_three
# description: Among sets whose examples all share one final digit, yes-rate on targets breaking that digit for 4-example sets minus for 3-example sets; observed > null means the model's linear per-example size penalty over-rejects at 4 examples (human rule confidence saturates by 3); observed < null means people's rejection keeps growing faster than the model's linear penalty.
def test_statistic(df):
    X = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy(float)
    valid = ~np.isnan(X)
    n = valid.sum(1)
    d = np.where(valid, np.mod(np.nan_to_num(X), 10), np.nan)
    dmin = np.nanmin(d, axis=1); dmax = np.nanmax(d, axis=1)
    uniform = (n >= 2) & (dmin == dmax)
    viol = uniform & ((t % 10) != dmin)
    a = viol & (n == 4); b = viol & (n == 3)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
