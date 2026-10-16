# name: balance_preference_similar_switches
# description: Among pairs whose switch counts differ by at most 1 and whose head-count imbalances |#H - n/2| differ, the proportion of choices for the more balanced sequence; observed above null_mean means people weight H/T balance more than the model's biased-coin alternative implies, below means less.
def test_statistic(df):
    a = df["sequence_a"]; b = df["sequence_b"]
    n = a.str.len().to_numpy()
    ia = np.abs(a.str.count("H").to_numpy() - n / 2)
    ib = np.abs(b.str.count("H").to_numpy() - n / 2)
    def sw(s):
        return sum(1 for x, y in zip(s, s[1:]) if x != y)
    u = pd.unique(pd.concat([a, b]))
    m = {s: sw(s) for s in u}
    swa = a.map(m).to_numpy(); swb = b.map(m).to_numpy()
    sel = (np.abs(swa - swb) <= 1) & (ia != ib)
    if sel.sum() == 0:
        return 0.5
    cl = df["chose_left"].to_numpy()[sel]
    c = np.where(ia[sel] < ib[sel], cl, 1 - cl)
    return float(np.mean(c))
