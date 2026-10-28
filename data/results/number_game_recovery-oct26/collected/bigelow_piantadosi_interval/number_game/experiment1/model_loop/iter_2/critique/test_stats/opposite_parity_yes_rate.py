# name: opposite_parity_yes_rate
# description: Yes rate on targets whose parity differs from all examples, for multi-example sets whose examples are all even or all odd; observed below null means the model's guessing floor over-produces yeses on clearly rule-violating targets, above means it under-produces them.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    par = {}
    for s in uniq:
        a = np.array([int(x) for x in s.split(",")])
        if len(a) >= 2 and np.all(a % 2 == a[0] % 2):
            par[s] = float(a[0] % 2)
        else:
            par[s] = -1.0
    p = sets.map(par).to_numpy()
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    sel = (p >= 0) & ((t % 2) != p)
    return float(r[sel].mean()) if sel.any() else 0.0
