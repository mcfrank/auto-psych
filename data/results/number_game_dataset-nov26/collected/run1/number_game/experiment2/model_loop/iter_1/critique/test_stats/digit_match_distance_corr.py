# name: digit_match_distance_corr
# description: In multi-example sets where >=2 examples share a final digit d, among non-example targets ending in d, Pearson correlation between distance to the nearest example and the yes response; observed > null (closer to 0) means people apply the shared-digit rule flatly regardless of magnitude, whereas the model's distance-decaying similarity predicts a negative correlation; observed < null means steeper decay.
def test_statistic(df):
    from collections import Counter
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    M = np.full((len(uniq), 4), np.nan)
    rd = np.full(len(uniq), -1)
    for i, s in enumerate(uniq):
        l = [int(v) for v in s.split(",")][:4]
        M[i, :len(l)] = l
        c = Counter(e % 10 for e in l).most_common(1)[0]
        if len(l) >= 2 and c[1] >= 2:
            rd[i] = c[0]
    idx = pd.Series(np.arange(len(uniq)), index=uniq)[sets].to_numpy()
    E = M[idx]
    d = rd[idx]
    t = df["target"].astype(float).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    D = np.where(np.isnan(E), np.inf, np.abs(E - t[:, None])).min(1)
    m = (d >= 0) & ((t % 10) == d) & (D > 0)
    if m.sum() < 3:
        return 0.0
    x, y = D[m], r[m]
    if x.std() == 0 or y.std() == 0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
