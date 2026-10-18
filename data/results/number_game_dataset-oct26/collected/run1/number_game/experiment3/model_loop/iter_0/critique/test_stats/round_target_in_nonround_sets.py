# name: round_target_in_nonround_sets
# description: In multi-example sets with no example ending in 0, yes-rate on targets that are multiples of 10 minus yes-rate on other even targets not matching any example's final digit; observed != null means the model mis-handles the special status of round numbers (it treats 0 as just another even digit).
def test_statistic(df):
    sets = df["number_set"].astype(str)
    codes, cats = pd.factorize(sets)
    ok = np.array([("," in s) and all(int(x) % 10 != 0 for x in s.split(",")) for s in cats])
    digs = [set(int(x) % 10 for x in s.split(",")) for s in cats]
    t = df["target"].astype(int).to_numpy()
    td = t % 10
    match = np.zeros(len(df), dtype=bool)
    for i, ds in enumerate(digs):
        m = codes == i
        match[m] = np.isin(td[m], list(ds))
    inset = ok[codes]
    r = df["rating"].to_numpy().astype(float)
    a = inset & (td == 0)
    b = inset & (td % 2 == 0) & (td != 0) & ~match
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
