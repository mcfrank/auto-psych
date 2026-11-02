# name: consistent_last_digit_far
# description: For sets of >=2 examples that all share one last digit, yes rate among non-example targets more than 5 from every example that end in that same digit; observed above null means people adopt the crisp shared-final-digit rule and extend it to distant numbers more than the model's graded profile predicts, below null means less.
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
    ld0 = ld[:, 0]
    allsame = (n >= 2) & (np.nansum(ld == ld0[:, None], axis=1) == n)
    m = allsame & nonmember & (nearest > 5) & (np.mod(t, 10) == ld0)
    return float(r[m].mean()) if m.any() else 0.0
