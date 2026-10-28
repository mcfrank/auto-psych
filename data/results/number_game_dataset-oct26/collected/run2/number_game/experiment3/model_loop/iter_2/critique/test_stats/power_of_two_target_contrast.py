# name: power_of_two_target_contrast
# description: In sets of 2+ examples that are all powers of two, yes-rate on non-example power-of-two targets minus yes-rate on other even targets; observed > null means people use a "powers of 2" concept beyond multiples-of-g/factor similarity, which the model under-produces.
def test_statistic(df):
    P2 = [1, 2, 4, 8, 16, 32, 64]
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    exv = ex.to_numpy(dtype=float)
    k = ex.notna().sum(axis=1).to_numpy()
    ok = (np.isin(np.nan_to_num(exv, nan=-1).astype(int), P2) | np.isnan(exv)).all(axis=1) & (k >= 2)
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    isex = (exv == t[:, None]).any(axis=1)
    tp = np.isin(t, P2)
    a = ok & ~isex & tp
    b = ok & ~isex & ~tp & (t % 2 == 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(y[a].mean() - y[b].mean())
