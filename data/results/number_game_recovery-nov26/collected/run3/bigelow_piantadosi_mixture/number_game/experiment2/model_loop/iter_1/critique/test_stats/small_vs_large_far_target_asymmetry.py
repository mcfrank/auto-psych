# name: small_vs_large_far_target_asymmetry
# description: Among targets >15 from every example that share no final digit with any example, yes rate for small targets (<=15) minus large targets (>=85); observed above null means people favour small numbers far from the examples more than the model predicts (a small-number / compressed-scale bias the model lacks).
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
    Ep = np.mod(E, 2)
    n_even = np.sum(Ep == 0, axis=1)
    n_odd = np.sum(Ep == 1, axis=1)
    lastE = E[np.arange(len(E)), np.maximum(n - 1, 0)]

    base = ~in_set & ~shares_last & (dist > 15)
    a = base & (t <= 15)
    b = base & (t >= 85)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
