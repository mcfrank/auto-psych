# name: singleton_set_yes_rate
# description: Yes rate on sets with exactly one example, for targets other than that example; observed above null means the model under-generalizes from a single example (size principle too sharp), below null means it over-generalizes.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    n = np.sum(~np.isnan(ex), axis=1)
    t = df["target"].astype(float).to_numpy()
    m = (n == 1) & (ex[:, 0] != t)
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].to_numpy()[m].mean())
