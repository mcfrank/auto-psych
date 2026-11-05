# name: shared_decade_bonus
# description: Among multi-example sets, yes-rate for targets whose tens digit equals some example's tens digit minus yes-rate for targets that share no example's tens digit (targets that are examples excluded); observed > null means the model under-produces endorsement driven by same-decade similarity, which it has no feature for.
def test_statistic(df):
    sets = df["number_set"].unique()
    tens = {}
    for s in sets:
        ex = [int(x) for x in str(s).split(",")]
        if len(ex) >= 2:
            tens[s] = set(e // 10 for e in ex)
    sub = df[df["number_set"].isin(list(tens.keys()))]
    if len(sub) == 0:
        return 0.0
    t = sub["target"].astype(int).values
    keys = sub["number_set"].values
    # vectorised via a lookup table: set index x decade
    idx = {s: i for i, s in enumerate(tens)}
    tab = np.zeros((len(idx), 11), dtype=bool)
    for s, i in idx.items():
        for d in tens[s]:
            tab[i, d] = True
    si = np.array([idx[s] for s in pd.unique(keys)])
    codes = pd.Series(keys).map(idx).values
    hit = tab[codes, t // 10]
    r = sub["rating"].values.astype(float)
    if hit.sum() == 0 or (~hit).sum() == 0:
        return 0.0
    return float(r[hit].mean() - r[~hit].mean())
