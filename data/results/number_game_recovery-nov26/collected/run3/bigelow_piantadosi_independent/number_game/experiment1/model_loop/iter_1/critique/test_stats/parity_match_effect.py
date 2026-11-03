# name: parity_match_effect
# description: Among sets whose examples all share parity, yes rate for parity-matching targets minus parity-mismatching targets; observed < null means the model over-commits to parity (odd/even) rules relative to people, observed > null means people use parity more.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).astype(float).to_numpy()
    t = df["target"].astype(float).to_numpy()
    par = np.mod(ex, 2)
    n = np.sum(~np.isnan(ex), axis=1)
    s = np.nansum(par, axis=1)
    alleven = s == 0; allodd = s == n
    uni = (alleven | allodd) & (n >= 2)
    setpar = np.where(allodd, 1.0, 0.0)
    match = uni & (np.mod(t, 2) == setpar)
    mis = uni & (np.mod(t, 2) != setpar)
    r = df["rating"].astype(float).to_numpy()
    if match.sum() == 0 or mis.sum() == 0:
        return 0.0
    return float(r[match].mean() - r[mis].mean())
