# name: majority_final_digit_effect
# description: In sets where a final digit is shared by a strict majority (>=2, >half) but not all examples (e.g. 69,9,39,21 -> '9'), yes-rate for targets ending in that digit minus other targets in those sets; observed > null means the model (which only counts properties shared by ALL examples) under-produces partial/majority-feature similarity generalization.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].values.astype(float)
    a = np.zeros(len(t), bool); b = np.zeros(len(t), bool)
    for s in pd.unique(sets):
        xs = [int(x) for x in s.split(",")]
        n = len(xs)
        if n < 3:
            continue
        fds = [x % 10 for x in xs]
        maj = [d for d in set(fds) if fds.count(d) >= 2 and fds.count(d) * 2 > n and fds.count(d) < n]
        if not maj:
            continue
        idx = sets == s
        has = np.isin(t % 10, maj)
        a |= idx & has
        b |= idx & ~has
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
