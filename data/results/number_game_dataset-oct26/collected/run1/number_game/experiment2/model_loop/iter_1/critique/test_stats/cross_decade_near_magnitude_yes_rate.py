# name: cross_decade_near_magnitude_yes_rate
# description: Yes rate for targets within 1-4 of some example numerically but in a different decade and sharing no final digit with any example (e.g. 29 vs 31); t_observed > null_mean means the model under-produces magnitude-proximity generalization that ignores decade boundaries.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    parsed = {s: np.array([int(x) for x in s.split(",")]) for s in sets.unique()}
    codes, uniq = pd.factorize(sets)
    maxk = 4
    X = np.full((len(uniq), maxk), -1000)
    for i, s in enumerate(uniq):
        xs = parsed[s][:maxk]
        X[i, :len(xs)] = xs
    E = X[codes]  # n x 4
    valid = E > -1000
    d = np.abs(E - t[:, None])
    near_other_dec = valid & (d >= 1) & (d <= 4) & ((E // 10) != (t // 10)[:, None])
    same_dec = valid & ((E // 10) == (t // 10)[:, None])
    same_dig = valid & ((E % 10) == (t % 10)[:, None])
    mask = near_other_dec.any(1) & ~same_dec.any(1) & ~same_dig.any(1)
    if mask.sum() == 0:
        return 0.0
    return float(df["rating"].values[mask].mean())
