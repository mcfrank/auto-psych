# name: single_example_far_yes_rate
# description: For single-example sets, yes rate for targets more than 20 away from the example; observed above null means people generalize far from a lone example (via rules) more than the model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    sel = (n == 1) & (np.abs(t - ex[:, 0]) > 20)
    return float(r[sel].mean()) if sel.any() else 0.0
