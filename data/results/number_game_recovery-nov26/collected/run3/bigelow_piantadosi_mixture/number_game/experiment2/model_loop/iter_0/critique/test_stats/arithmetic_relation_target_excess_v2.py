# name: arithmetic_relation_target_excess_v2
# description: Yes rate for non-example targets equal to a sum, absolute difference, double or half of examples minus yes rate of other targets (all restricted to no shared final digit and >3 from every example); observed above null means people use arithmetic relations between examples that the model lacks.
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
    rel = np.zeros(len(t), bool)
    for i in range(4):
        rel |= (E[:, i] * 2 == t) | (E[:, i] / 2 == t)
        for j in range(i + 1, 4):
            rel |= (E[:, i] + E[:, j] == t) | (np.abs(E[:, i] - E[:, j]) == t)
    base = ~in_set & ~shares_last & (dist > 3)
    a = base & rel
    b = base & ~rel
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
