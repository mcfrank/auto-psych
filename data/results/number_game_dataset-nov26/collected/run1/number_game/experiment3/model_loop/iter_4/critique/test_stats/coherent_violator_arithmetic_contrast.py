# name: coherent_violator_arithmetic_contrast
# description: Among targets that break the coincidence rule of a 2-4-example coherent set (wrong final digit for same-digit sets, else not a multiple of 5 for all-multiple-of-5 sets, else wrong parity for same-parity sets), yes-rate for targets that are a multiple (>1, not equal) or factor-kin (gcd>=3, neither divides the other) of some example minus yes-rate for violators with neither relation; observed > null means arithmetic kinship rescues rule-breakers more than the model's uniform share-of-support cut allows, observed < null means it rescues them less.
def test_statistic(df):
    s = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric)
    S = s.to_numpy(dtype=float)
    n = s.notna().sum(axis=1).to_numpy()
    t = df["target"].astype(int).to_numpy()
    T = t[:, None].astype(float)
    dig = s % 10
    same_dig = (dig.max(axis=1) == dig.min(axis=1)).to_numpy()
    f5 = ((s % 5 == 0) | s.isna()).all(axis=1).to_numpy()
    par = s % 2
    same_par = (par.max(axis=1) == par.min(axis=1)).to_numpy()
    e0 = S[:, 0].astype(int)
    viol = np.where(same_dig, t % 10 != e0 % 10,
           np.where(f5, t % 5 != 0, np.where(same_par, t % 2 != e0 % 2, False))) & (n >= 2)
    valid = ~np.isnan(S)
    Si = np.where(valid, S, 1).astype(int)
    Ti = np.broadcast_to(t[:, None], Si.shape)
    mult = valid & (Si > 1) & (Ti % Si == 0) & (Ti != Si)
    g = np.gcd(Ti, Si)
    fk = valid & (g >= 3) & (Ti % Si != 0) & (Si % Ti != 0)
    rel = (mult | fk).any(axis=1)
    r = df["rating"].to_numpy(dtype=float)
    a = viol & rel
    b = viol & ~rel
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
