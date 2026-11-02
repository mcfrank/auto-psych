# name: set_size_yes_gap
# description: Yes-rate for non-example targets after 4-example sets minus after 1-example sets; observed above null means the model over-tightens generalisation as examples accumulate (size-principle too strong), below means too weak.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ok = ~np.isnan(ex)
    n = ok.sum(axis=1)
    inset = np.where(ok, ex == t[:, None], False).any(axis=1)
    a = (n == 4) & ~inset
    b = (n == 1) & ~inset
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
