# name: prime_target_in_prime_sets
# description: In sets of 2+ examples that are all prime, yes-rate on non-example prime targets minus yes-rate on odd composite targets; observed > null means people use a "prime numbers" concept that the model (factor-sharing similarity favours composites like 21 for 3,7) under-produces.
def test_statistic(df):
    P = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97}
    s = df["number_set"].astype(str)
    ex = s.str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    k = ex.notna().sum(axis=1).to_numpy()
    isp = np.isin(np.nan_to_num(exv, nan=-1).astype(int), list(P)) | np.isnan(exv)
    allp = isp.all(axis=1) & (k >= 2)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    isex = (exv == t[:, None]).any(axis=1)
    tp = np.isin(t, list(P))
    a = allp & ~isex & tp
    b = allp & ~isex & ~tp & (t % 2 == 1) & (t > 1)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
