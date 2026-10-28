# name: late_minus_early_outside_breadth
# description: Yes-rate for non-example targets outside the example range on trials in the second half of the session minus the first half; observed above null means generalization beyond the examples broadens over the session (the static model under-produces late yeses), below means it narrows.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"])
    E = np.full((len(uniq), 4), -1000)
    for i, s in enumerate(uniq):
        v = [int(x) for x in str(s).split(",")]
        E[i, : len(v)] = v
    arr = E[codes]
    valid = arr > 0
    n_ex = valid.sum(axis=1)
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].to_numpy().astype(float)
    is_ex = (arr == t[:, None]).any(axis=1)
    lo = np.where(valid, arr, 1000).min(axis=1)
    hi = np.where(valid, arr, -1000).max(axis=1)
    dist = np.abs(np.where(valid, arr, 10000) - t[:, None]).min(axis=1)
    ti = df["trial_index"].to_numpy()
    late = ti >= np.median(ti)
    out = ((t < lo) | (t > hi)) & ~is_ex
    a, b = r[out & late], r[out & ~late]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
