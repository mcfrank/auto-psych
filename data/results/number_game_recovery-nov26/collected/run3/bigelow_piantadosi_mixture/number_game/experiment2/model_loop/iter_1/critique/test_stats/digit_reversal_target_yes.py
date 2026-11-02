# name: digit_reversal_target_yes
# description: Yes rate for targets that are the digit reversal of a two-digit example (e.g. 38 for 83, not itself an example) minus yes rate of other non-example targets at a similar distance (3-30 from every example) sharing no final digit with any example; observed above null means people use digit-reversal similarity the model lacks.
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

    rev = np.mod(E, 10) * 10 + np.floor(E / 10)
    ok = (E >= 10) & (E <= 99) & (np.mod(E, 10) != 0) & (rev != E)
    is_rev = np.any(ok & (rev == t[:, None]), axis=1) & ~in_set
    base = ~in_set & ~shares_last & (dist >= 3) & (dist <= 30)
    a = is_rev
    b = base & ~is_rev
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
