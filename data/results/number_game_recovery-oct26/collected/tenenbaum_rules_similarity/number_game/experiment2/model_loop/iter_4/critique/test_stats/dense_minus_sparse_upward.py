# name: dense_minus_sparse_upward
# description: Yes-rate to targets 1-10 above the largest example on sets of >=3 examples minus the same on single-example sets; the model predicts top-marking credit fades with more examples (so this is positive), so observed below null means people's upward generalization does not recover with denser sets as much as the model claims, above null means it recovers more.
def test_statistic(df):
    sets = df["number_set"].astype(str).str.replace(" ", "")
    n = sets.str.count(",").to_numpy() + 1
    lut = {k: max(int(z) for z in k.split(",")) for k in sets.unique()}
    mx = sets.map(lut).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    d = t - mx
    above = (d >= 1) & (d <= 10)
    a = above & (n >= 3)
    b = above & (n == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
