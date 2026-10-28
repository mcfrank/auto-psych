# name: multiple_of_ten_set_rule_yes
# description: For sets whose examples are all multiples of 10, yes rate on non-example targets that are multiples of 10; observed above null_mean means people favour the multiples-of-10 rule more than the model's kind-pooled rule prior (which makes it no likelier than multiples of 7) predicts, below means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n_ex = valid.sum(axis=1)
    lo = np.nanmin(ex, axis=1); hi = np.nanmax(ex, axis=1)
    dist = np.nanmin(np.abs(ex - t[:, None]), axis=1)
    member = dist == 0
    allten = np.all(np.where(valid, np.mod(ex, 10) == 0, True), axis=1)
    sel = allten & (~member) & (np.mod(t, 10) == 0)
    if sel.sum() == 0:
        return 0.0
    return float(y[sel].mean())
