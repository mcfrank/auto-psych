# name: mixed_parity_even_target_bias
# description: Among sets containing both odd and even examples, yes rate for even non-example targets minus odd non-example targets; the model has no parity preference once examples mix parity, so observed away from the null means people carry a residual parity bias (above = even favoured, below = odd favoured) the model misses.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    mixed = {s: len({int(x) % 2 for x in s.split(",")}) == 2 for s in sets.unique()}
    mx = sets.map(mixed).to_numpy().astype(bool)
    t = df["target"].astype(int).to_numpy()
    inset = np.array([str(tt) in s.split(",") for tt, s in zip(t, sets)])
    ok = mx & ~inset
    y = df["rating"].to_numpy().astype(float)
    a, b = ok & (t % 2 == 0), ok & (t % 2 == 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
