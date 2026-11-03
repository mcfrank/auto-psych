# name: multiples_set_rule_contrast
# description: Among multi-example sets whose examples are all multiples of some k in 3..12 (largest such k), yes rate for targets that are multiples of k minus yes rate for other targets; observed below null means the model over-sharpens multiples rules (people discriminate rule members less), above means people follow multiples rules more sharply than predicted.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    kk = np.zeros(len(uniq), int)
    for i, s in enumerate(uniq):
        v = [int(x) for x in s.split(",")]
        if len(v) > 1:
            for k in range(12, 2, -1):
                if all(e % k == 0 for e in v):
                    kk[i] = k; break
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    k = kk[codes]
    m = k > 0
    inr = np.zeros(len(t), bool)
    inr[m] = (t[m] % k[m]) == 0
    a, b = m & inr, m & ~inr
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
