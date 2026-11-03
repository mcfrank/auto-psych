# name: interior_nondigit_yes_rate
# description: Yes rate for targets strictly inside the examples' range (min<t<max, multi-example sets, range>=10) that share no digit with any example and are not examples; observed above the null means the model under-produces interval (magnitude) generalization when a digit rule competes, below means it over-produces it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    td = np.array([[str(v).count(str(d)) > 0 for d in range(10)] for v in range(101)])
    exi = np.nan_to_num(ex, nan=0).astype(int)
    exd = np.zeros((len(df), 10), bool)
    for j in range(ex.shape[1]):
        exd |= td[exi[:, j]] & ~np.isnan(ex[:, j])[:, None]
    share = np.any(exd & td[t], axis=1)
    inset = np.any(ex == t[:, None], axis=1)
    m = (n >= 2) & (hi - lo >= 10) & (t > lo) & (t < hi) & ~share & ~inset
    return float(y[m].mean()) if m.any() else 0.0
