# name: magnitude_scaled_generalization
# description: Yes-rate for non-example targets 3-20 outside the example range whose nearest example is <= 30 minus those whose nearest example is >= 60; observed below null means people generalize less around small numbers than large ones (Weber-like log-scale similarity) beyond what the model's translation-invariant intervals produce, above null the reverse.
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
    nearest = np.where(t < lo, lo, hi)
    out = ((t < lo) | (t > hi)) & (dist >= 3) & (dist <= 20) & ~is_ex
    a, b = r[out & (nearest <= 30)], r[out & (nearest >= 60)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
