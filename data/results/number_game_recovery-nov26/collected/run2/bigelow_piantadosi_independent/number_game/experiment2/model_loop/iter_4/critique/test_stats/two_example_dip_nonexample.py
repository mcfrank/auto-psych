# name: two_example_dip_nonexample
# description: Yes rate on non-example targets for 2-example sets minus the mean of the yes rates for 3- and 4-example sets; observed below null means people are more conservative after two examples than the model's monotone capacity limit predicts.
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
    ne = ~in_set
    return rate(ne & (n == 2)) - 0.5 * (rate(ne & (n == 3)) + rate(ne & (n == 4)))
