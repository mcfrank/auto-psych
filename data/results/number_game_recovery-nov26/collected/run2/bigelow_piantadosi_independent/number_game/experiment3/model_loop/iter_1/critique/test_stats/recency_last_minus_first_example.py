# name: recency_last_minus_first_example
# description: Among non-example targets for sets of 2+ examples, yes-rate when the target is closer to the last-shown example than to the first-shown one minus yes-rate when closer to the first; observed away from null_mean (model is order-invariant) means presentation order biases generalization (positive = recency).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    u = df["number_set"].unique()
    first = {s: int(s.split(",")[0]) for s in u}
    last = {s: int(s.split(",")[-1]) for s in u}
    nn = {s: s.count(",") + 1 for s in u}
    f = df["number_set"].map(first).to_numpy(); l = df["number_set"].map(last).to_numpy()
    n = df["number_set"].map(nn).to_numpy()
    exm = np.array([str(tt) in s.split(",") for s, tt in zip(df["number_set"].to_numpy(), t)])
    df_ = np.abs(t - f); dl = np.abs(t - l)
    r = df["rating"].astype(float).to_numpy()
    base = (n >= 2) & ~exm
    a = base & (dl < df_); b = base & (df_ < dl)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
