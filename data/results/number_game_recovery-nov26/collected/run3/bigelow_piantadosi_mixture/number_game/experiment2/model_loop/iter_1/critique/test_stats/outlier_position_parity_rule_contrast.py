# name: outlier_position_parity_rule_contrast
# description: In >=3-example sets where exactly one example differs in parity from the rest, yes rate for non-example targets of the majority parity when the odd-one-out is shown FIRST minus when it is shown LAST; observed above null means a first-position stray undermines the rule less (relative to a last-position one) for people than the anchored-first-example model predicts, below means more.
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
    maj = np.where(n_even > n_odd, 0, 1)
    out_par = 1 - maj
    first_out = one_out & (Ep[:, 0] == out_par)
    last_par = np.mod(lastE, 2)
    last_out = one_out & (last_par == out_par)
    tgt = ~in_set & (np.mod(t, 2) == maj)
    a = first_out & tgt
    b = last_out & tgt
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
