# name: majority_last_digit_far_target
# description: For sets of >=3 examples where exactly n-1 examples share a last digit d (one exception), yes rate among non-example targets ending in d that are >4 from every example; observed above null means people extend a partly shared last digit to distant targets more than the fitted exception-tolerant model predicts, below null means less.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    r = df["rating"].astype(float).to_numpy()
    valid = ~np.isnan(ex)
    n = valid.sum(axis=1)
    dist = np.where(valid, np.abs(ex - t[:, None]), np.inf)
    nearest = dist.min(axis=1)
    nonmember = nearest > 0
    ld = np.where(valid, np.mod(ex, 10), np.nan)
    tl = np.mod(t, 10)
    cnt = np.zeros((len(t), 10))
    for d in range(10):
        cnt[:, d] = (ld == d).sum(axis=1)
    maj = cnt.argmax(axis=1)
    majc = cnt.max(axis=1)
    m = (n >= 3) & (majc == n - 1) & (tl == maj) & nonmember & (nearest > 4)
    return float(r[m].mean()) if m.any() else 0.0
