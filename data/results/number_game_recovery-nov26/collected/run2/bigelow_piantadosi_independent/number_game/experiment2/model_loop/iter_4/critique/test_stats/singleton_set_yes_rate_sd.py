# name: singleton_set_yes_rate_sd
# description: Standard deviation, across 1-example sets, of each set's yes rate on non-example targets; observed above null means people's breadth of generalization varies across single examples more than the model's rule familiarities and ranges predict (below: less).
def test_statistic(df):
    E = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    pad = np.isnan(E)
    n = (~pad).sum(axis=1)
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    in_set = (E == t[:, None]).any(axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    def rate(mask):
        return float(y[mask].mean()) if mask.any() else 0.0
    m = (n == 1) & ~in_set
    g = pd.Series(y[m]).groupby(df["number_set"].to_numpy()[m]).mean()
    return float(g.std()) if len(g) > 1 else 0.0
