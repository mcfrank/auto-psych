# name: single_digit_target_yes_rate
# description: Yes-rate for single-digit (1-9) non-example targets in sets whose smallest example is >= 10; observed above null means people accept small numbers for large-number sets more than the model predicts, below means they reject them more.
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
    m = (t <= 9) & (lo >= 10) & ~is_ex
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
