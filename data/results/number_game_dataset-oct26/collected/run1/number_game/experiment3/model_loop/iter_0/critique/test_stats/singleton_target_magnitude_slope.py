# name: singleton_target_magnitude_slope
# description: For one-example sets, the OLS slope of yes (0/1) on |target - example|/100; observed more negative than null means the model under-produces the decline of acceptance with numeric distance from a lone example.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",").to_numpy()
    if single.sum() < 3:
        return 0.0
    x = s[single].astype(int).to_numpy()
    t = df["target"].astype(int).to_numpy()[single]
    d = np.abs(t - x) / 100.0
    r = df["rating"].to_numpy().astype(float)[single]
    v = d.var()
    if v <= 0:
        return 0.0
    return float(((d - d.mean()) * (r - r.mean())).mean() / v)
