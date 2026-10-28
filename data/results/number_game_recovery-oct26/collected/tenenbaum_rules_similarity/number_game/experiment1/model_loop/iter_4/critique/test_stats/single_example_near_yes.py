# name: single_example_near_yes
# description: For single-example sets, yes rate on targets 1-5 away from the example; observed above null_mean means people generalize to close neighbours of a lone example more than the model's interval/rule posterior predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    sel = (n_ex == 1) & (dist >= 1) & (dist <= 5)
    if sel.sum() == 0:
        return 0.0
    return float(y[sel].mean())
