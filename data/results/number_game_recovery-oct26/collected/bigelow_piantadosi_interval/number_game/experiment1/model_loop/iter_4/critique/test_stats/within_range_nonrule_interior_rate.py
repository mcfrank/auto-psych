# name: within_range_nonrule_interior_rate
# description: In sets of 3-4 examples spanning >= 30, yes-rate for non-example targets strictly inside the range that share neither parity-with-all-examples nor a final digit with any example; observed below null means people reject unrelated interior numbers more than the model's interval hypotheses predict (model over-produces interior yeses), above means they accept them more.
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
    par = np.where(valid, arr % 2, -1)
    all_same_par = ((par == (t % 2)[:, None]) | ~valid).all(axis=1)
    share_ld = (valid & ((arr % 10) == (t % 10)[:, None])).any(axis=1)
    m = (n_ex >= 3) & (hi - lo >= 30) & (t > lo) & (t < hi) & ~is_ex & ~all_same_par & ~share_ld
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
