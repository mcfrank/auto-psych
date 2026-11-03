# name: hull_gap_interior_yes_rate
# description: Yes rate for targets strictly inside the [min,max] range of a >=2-example set spanning >=40 that are >=6 from every example and share no final digit with any; observed above null means people fill wide gaps between examples more than the interval+stray model predicts, below means less.
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
    m = (n >= 2) & (hi - lo >= 40) & (t > lo) & (t < hi) & (dist >= 6) & ~shares_last & ~in_set
    return float(y[m].mean()) if m.any() else 0.0
