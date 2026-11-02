# name: prime_set_prime_target_contrast
# description: Among multi-example sets whose examples are all prime, yes rate for prime targets minus yes rate for non-prime targets; observed below null means the model over-produces prime-rule generalisation, above means people follow the prime rule more sharply than predicted.
def test_statistic(df):
    pr = np.zeros(101, bool)
    for k in range(2, 101):
        pr[k] = all(k % a for a in range(2, int(k ** 0.5) + 1))
    sets = df["number_set"].astype(str)
    uniq = pd.unique(sets)
    codes = pd.Categorical(sets, categories=uniq).codes
    ok = np.array([(s.count(",") > 0) and all(pr[int(x)] for x in s.split(",")) for s in uniq])
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = ok[codes]
    a, b = m & pr[t], m & ~pr[t]
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
