# name: single_example_shared_digit_or_multiple
# description: For one-example sets, yes-rate on non-example targets that are a multiple/divisor of the example or share its final digit, minus yes-rate on other targets; observed above null means people generalize from a single number by arithmetic/digit relations more sharply than the model, below means less sharply.
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
    e = arr[:, 0]
    one = (n_ex == 1) & ~is_ex
    rel = (t % np.maximum(e, 1) == 0) | (e % np.maximum(t, 1) == 0) | (t % 10 == e % 10)
    a, b = r[one & rel], r[one & ~rel]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
