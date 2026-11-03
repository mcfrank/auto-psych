# name: shared_last_digit_match
# description: Among sets of 2+ examples that all end in the same digit, yes rate for targets ending in that digit (and not equal to an example); observed < null means the model over-predicts generalization of "ends in d" rules, observed > null means it under-predicts.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    n = np.sum(~np.isnan(ex), axis=1)
    ld = np.mod(ex, 10)
    first = ld[:, 0]
    same = np.all(np.isnan(ld) | (ld == first[:, None]), axis=1) & (n >= 2)
    isex = np.any(ex == t[:, None], axis=1)
    m = same & (np.mod(t, 10) == first) & ~isex
    r = df["rating"].astype(float).to_numpy()
    if m.sum() == 0:
        return 0.0
    return float(r[m].mean())
