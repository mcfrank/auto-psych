# name: parity_set_distance_corr
# description: In sets of >=3 examples all of one parity but not all one final digit, among parity-matching targets that share no example's final digit, the correlation of yes with distance to the nearest example; observed closer to 0 (or above null) means people treat such sets as a parity rule with flat endorsement, while the model's magnitude similarity predicts a more negative distance gradient.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).to_numpy(dtype=float)
    t = df["target"].astype(int).to_numpy().astype(float)
    n = (~np.isnan(ex)).sum(1)
    par = np.mod(ex, 2); dig = np.mod(ex, 10)
    samepar = (np.nanmax(par, 1) == np.nanmin(par, 1)) & (n >= 3)
    samedig = np.nanmax(dig, 1) == np.nanmin(dig, 1)
    tpar = np.mod(t, 2) == np.nanmin(par, 1)
    anydig = np.nansum((dig == np.mod(t, 10)[:, None]).astype(float), 1) > 0
    dmin = np.nanmin(np.abs(ex - t[:, None]), 1)
    m = samepar & ~samedig & tpar & ~anydig
    r = df["rating"].to_numpy().astype(float)[m]; d = dmin[m]
    if m.sum() < 3 or r.std() == 0 or d.std() == 0:
        return 0.0
    return float(np.corrcoef(r, d)[0, 1])
