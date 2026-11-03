# name: singleton_yes_rate_far_targets
# description: For single-example sets, yes-rate for targets more than 10 away from the example; observed > null means people generalize from one example more broadly (via rules/broad similarity) than the model predicts, observed < null means more narrowly.
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    x = pd.to_numeric(s.where(single, "0")).to_numpy(dtype=float)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = single.to_numpy() & (np.abs(t - x) > 10)
    if sel.sum() == 0:
        return 0.0
    return float(r[sel].mean())
