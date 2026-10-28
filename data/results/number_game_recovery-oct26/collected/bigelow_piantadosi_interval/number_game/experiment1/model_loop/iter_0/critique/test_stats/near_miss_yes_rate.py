# name: near_miss_yes_rate
# description: Yes-rate for targets 1-3 away from the nearest example (and not an example itself); observed above null means people generalize by numerical proximity more than the rule/interval posterior predicts.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = sets.unique()
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in uniq}
    t = df["target"].astype(int).to_numpy()
    r = df["rating"].astype(float).to_numpy()
    out = np.full(len(df), np.inf)
    for s, a in parsed.items():
        m = (sets == s).to_numpy()
        out[m] = np.abs(t[m][:, None] - a[None, :]).min(axis=1)
    sel = (out >= 1) & (out <= 3)
    return float(r[sel].mean()) if sel.any() else 0.0
