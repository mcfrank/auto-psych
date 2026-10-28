# name: set_size_two_minus_four
# description: Yes rate to non-example targets for 2-example sets minus that for 4-example sets; observed above the null means people tighten generalization with more examples more steeply than the model (or are broader after 2 examples), below means the model's size-principle tightening is too strong.
def test_statistic(df):
    ex = df["number_set"].str.split(",")
    n = df["number_set"].str.count(",") + 1
    tgt = df["target"].astype(int).astype(str)
    is_ex = np.array([t in e for t, e in zip(tgt, ex)])
    r = df["rating"].to_numpy().astype(float)
    a = (n.to_numpy() == 2) & ~is_ex
    b = (n.to_numpy() == 4) & ~is_ex
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
