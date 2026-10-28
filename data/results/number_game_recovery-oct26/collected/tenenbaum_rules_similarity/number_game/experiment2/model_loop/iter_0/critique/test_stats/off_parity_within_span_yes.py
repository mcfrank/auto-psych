# name: off_parity_within_span_yes
# description: Among multi-example sets whose examples all share parity, yes-rate to opposite-parity targets lying inside the examples' [min,max] span; observed above null means the model under-weights range/similarity relative to parity rules for in-span targets.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",").apply(lambda l: np.array([int(x) for x in l]))
    t = df["target"].astype(int).values
    n = ex.apply(len).values
    mn = ex.apply(np.min).values; mx = ex.apply(np.max).values
    par = ex.apply(lambda a: int(a[0] % 2) if np.all(a % 2 == a[0] % 2) else -1).values
    mask = (n >= 2) & (par >= 0) & (t > mn) & (t < mx) & (t % 2 != par)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
