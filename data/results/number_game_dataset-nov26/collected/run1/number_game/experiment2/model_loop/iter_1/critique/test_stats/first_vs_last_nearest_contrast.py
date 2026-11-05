# name: first_vs_last_nearest_contrast
# description: In multi-example sets, among non-example targets, yes-rate when the magnitude-nearest example is the first one shown minus when it is the last one shown; observed > null means a primacy advantage (early examples weigh more), observed < null means a recency advantage, either of which the model's order-symmetric summed similarity cannot produce.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    M = np.full((len(uniq), 4), np.nan)
    for i, s in enumerate(uniq):
        l = [int(v) for v in s.split(",")][:4]
        M[i, :len(l)] = l
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sets].to_numpy()
    E = M[idx]
    n = (~np.isnan(E)).sum(1)
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    D = np.abs(E - t[:, None])
    isex = np.nanmin(D, axis=1) == 0
    Dm = np.where(np.isnan(D), np.inf, D)
    near = np.argmin(Dm, axis=1)
    mn = Dm.min(1)
    ties = (Dm == mn[:, None]).sum(1) > 1
    base = (n >= 2) & ~isex & ~ties
    first = base & (near == 0)
    last = base & (near == n - 1)
    if first.sum() == 0 or last.sum() == 0:
        return 0.0
    return float(r[first].mean() - r[last].mean())
