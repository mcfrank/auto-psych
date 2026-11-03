# name: random_set_inrange_breadth
# description: In 3-4 example sets spanning >=40 where no final digit and no parity is shared by more than n-2 examples (no near-rule), yes rate for non-example targets inside the set's range; observed above null means people generalise broadly ('anything in range') from rule-less noisy sets more than the model predicts, below means less.
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

    cnt = np.stack([np.sum(Ed == d, axis=1) for d in range(10)], axis=1).max(axis=1)
    par = np.maximum(n_even, n_odd)
    noisy = (n >= 3) & (hi - lo >= 40) & (cnt <= n - 2) & (par <= n - 2)
    m = noisy & ~in_set & (t > lo) & (t < hi)
    return float(y[m].mean()) if m.any() else 0.0
