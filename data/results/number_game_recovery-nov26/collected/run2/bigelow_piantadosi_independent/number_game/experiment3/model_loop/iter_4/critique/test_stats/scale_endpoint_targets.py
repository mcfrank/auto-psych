# name: scale_endpoint_targets
# description: Yes rate for out-of-span targets 1 or 100 minus yes rate for all other out-of-span targets on the same sets; observed above null means the scale endpoints are endorsed as special/anchor numbers more than the model's concepts imply (below: less).
def test_statistic(df):
    t = df["target"].astype(int).to_numpy()
    exl = df["number_set"].astype(str).str.split(",")
    lo = exl.apply(lambda x: min(int(v) for v in x)).to_numpy()
    hi = exl.apply(lambda x: max(int(v) for v in x)).to_numpy()
    out = (t < lo) | (t > hi)
    end = (t == 1) | (t == 100)
    sets = df["number_set"].to_numpy()
    with_end = np.isin(sets, np.unique(sets[out & end]))
    r = df["rating"].to_numpy().astype(float)
    a, b = out & end, out & ~end & with_end
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
