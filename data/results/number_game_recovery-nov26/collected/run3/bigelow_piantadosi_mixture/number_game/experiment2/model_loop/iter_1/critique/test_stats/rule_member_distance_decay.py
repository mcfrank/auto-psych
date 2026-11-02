# name: rule_member_distance_decay
# description: In >=2-example sets whose examples all share a final digit, yes rate for non-example targets with that final digit lying >25 from every example minus those within 10 of an example; observed below null means people's rule generalisation decays with distance more than the model's rule-plus-interval mixture predicts, above means flatter.
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

    same = np.all((Ed == Ed[:, [0]]) | np.isnan(Ed), axis=1) & (n >= 2)
    tgt = same & ~in_set & (td == Ed[:, 0])
    a = tgt & (dist > 25)
    b = tgt & (dist <= 10)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
