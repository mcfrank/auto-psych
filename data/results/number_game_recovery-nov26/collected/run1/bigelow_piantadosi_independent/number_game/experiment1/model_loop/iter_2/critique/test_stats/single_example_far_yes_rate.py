# name: single_example_far_yes_rate
# description: For sets with exactly one example, yes rate among targets more than 10 from the example; observed above null means people generalise from a single example to distant numbers more broadly than the model predicts, below null means more narrowly.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    m = (n == 1) & nonmember & (nearest > 10)
    return float(r[m].mean()) if m.any() else 0.0
