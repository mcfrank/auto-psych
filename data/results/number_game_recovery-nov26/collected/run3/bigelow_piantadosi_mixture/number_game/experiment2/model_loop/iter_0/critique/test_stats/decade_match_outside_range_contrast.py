# name: decade_match_outside_range_contrast
# description: Among targets >5 outside the set range that share no final digit with any example, yes rate for targets sharing a tens digit with some example minus yes rate for those that do not; observed above null means people use decade similarity the model lacks.
def test_statistic(df):
    s = df["number_set"].astype(str)
    u = pd.unique(s)
    codes = pd.Categorical(s, categories=u).codes
    U = np.full((len(u), 4), np.nan)
    for i, x in enumerate(u):
        v = [int(float(z)) for z in x.split(",") if z.strip() != ""][:4]
        U[i, :len(v)] = v
    E = U[codes]
    n = np.sum(~np.isnan(E), axis=1)
    t = df["target"].astype(int).to_numpy().astype(float)
    y = df["rating"].astype(float).to_numpy()
    lo = np.nanmin(E, axis=1)
    hi = np.nanmax(E, axis=1)
    Ed = np.mod(E, 10)
    td = np.mod(t, 10)
    shares_last = np.any(Ed == td[:, None], axis=1)
    dist = np.nanmin(np.abs(E - t[:, None]), axis=1)
    in_set = np.any(E == t[:, None], axis=1)
    first = Ed[:, 0]
    all_same_last = np.all((Ed == first[:, None]) | np.isnan(Ed), axis=1)
    def tens(a):
        return np.where(a >= 100, 10, np.floor(a / 10))
    sd = np.any(tens(E) == tens(t)[:, None], axis=1)
    base = ((t < lo - 5) | (t > hi + 5)) & ~shares_last
    a = base & sd
    b = base & ~sd
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
