# name: within_range_gap_yes_rate
# description: Yes-rate for non-example targets strictly between the smallest and largest example of multi-example sets; observed above null means people fill in the example range (interval-like) more than the model predicts, below means less.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    lo = sets.map({s: a.min() for s, a in parsed.items()}).to_numpy()
    hi = sets.map({s: a.max() for s, a in parsed.items()}).to_numpy()
    n = sets.map({s: len(a) for s, a in parsed.items()}).to_numpy()
    isex = np.zeros(len(df), bool)
    for s, a in parsed.items():
        m = (sets == s).to_numpy()
        isex[m] = np.isin(t[m], a)
    sel = (n >= 2) & (t > lo) & (t < hi) & ~isex
    return float(r[sel].mean()) if sel.any() else 0.0
