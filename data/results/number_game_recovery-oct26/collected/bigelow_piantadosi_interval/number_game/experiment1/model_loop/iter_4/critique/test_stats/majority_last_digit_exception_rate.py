# name: majority_last_digit_exception_rate
# description: In sets of 3-4 examples where all but exactly one example share a final digit, the yes-rate for non-example targets with that final digit; observed above null means people adopt a rule despite one exception more readily than the model's 10%-noise likelihood allows (model under-produces these yeses), below means less.
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
    ld = np.where(valid, arr % 10, -1)
    cnt = (ld[:, :, None] == np.arange(10)[None, None, :]).sum(axis=1)
    maj = (cnt == (n_ex - 1)[:, None]) & (n_ex >= 3)[:, None]
    hit = maj[np.arange(len(t)), t % 10] & ~is_ex
    if hit.sum() == 0:
        return 0.0
    return float(r[hit].mean())
