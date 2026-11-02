# name: stray_parity_generalisation
# description: In >=3-example sets where exactly one example differs in parity from the rest, yes rate for non-example targets of the STRAY's parity that are >3 from every example and share no final digit with any; observed above null means people let the stray broaden the concept beyond the majority rule more than the model predicts, below means they discount it more.
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

    one_out = (n >= 3) & ((n_even == 1) | (n_odd == 1))
    out_par = np.where(n_even > n_odd, 1, 0)
    m = one_out & ~in_set & (np.mod(t, 2) == out_par) & (dist > 3) & ~shares_last
    return float(y[m].mean()) if m.any() else 0.0
