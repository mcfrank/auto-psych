# name: parity_match_effect
# description: Among sets of 2+ examples that all share one parity (target not in set), yes rate for targets of that parity minus targets of the other parity; observed < null means the model over-weights the odd/even rules relative to people, observed > null under-weights them.
def test_statistic(df):
    a = df["number_set"].astype(str).str.split(",", expand=True).astype(float).values
    t = df["target"].astype(float).values
    r = df["rating"].astype(float).values
    valid = ~np.isnan(a)
    n = valid.sum(axis=1)
    par = np.where(valid, np.mod(a, 2), np.nan)
    allodd = np.nansum(par == 1, axis=1) == n
    alleven = np.nansum(par == 0, axis=1) == n
    inset = (a == t[:, None]).any(axis=1)
    elig = (n >= 2) & (allodd | alleven) & ~inset
    setpar = np.where(allodd, 1, 0)
    match = elig & (np.mod(t, 2) == setpar)
    mis = elig & (np.mod(t, 2) != setpar)
    if match.sum() == 0 or mis.sum() == 0:
        return 0.0
    return float(r[match].mean() - r[mis].mean())
