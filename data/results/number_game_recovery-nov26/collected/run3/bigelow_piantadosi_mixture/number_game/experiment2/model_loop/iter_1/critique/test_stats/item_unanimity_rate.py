# name: item_unanimity_rate
# description: Fraction of (set, target) items answered >=3 times on which every answer is identical; observed above null means people are more consistent/extreme per item than the model's graded yes-probabilities predict (model too moderate), below means the model is overconfident.
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

    key = s + "|" + df["target"].astype(int).astype(str)
    g = pd.DataFrame({"k": key.to_numpy(), "y": y}).groupby("k")["y"].agg(["size", "mean"])
    g = g[g["size"] >= 3]
    if len(g) == 0:
        return 0.0
    return float(((g["mean"] == 0) | (g["mean"] == 1)).mean())
