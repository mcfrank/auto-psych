# name: multiple_of_five_generalization_after_mult5_sets
# description: On sets of >=2 examples that are all multiples of 5, yes rate to non-example targets that are multiples of 5 and more than 10 from every example; observed above null means people extend the multiples-of-5/10 rule farther than the model (model too interval-bound for round-number sets), below null means the model over-extends it.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    mx = np.nanmax(ex, axis=1)
    mn = np.nanmin(ex, axis=1)
    is_ex = np.any(ex == t[:, None], axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    exz = np.where(np.isnan(ex), 0.0, ex)
    all5 = np.all(exz % 5 == 0, axis=1) & (n >= 2)
    m = all5 & ~is_ex & (t % 5 == 0) & (dist > 10)
    if m.sum() == 0:
        return 0.0
    return float(y[m].mean())
