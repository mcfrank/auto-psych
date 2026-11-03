# name: single_example_magnitude_gradient
# description: For single-example sets, yes rate for targets within 10 of the example minus yes rate for targets more than 30 away (excluding same-final-digit targets); observed above null means people generalise by magnitude similarity more steeply than the model predicts.
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
    base = (n == 1) & ~in_set & ~shares_last
    a = base & (dist <= 10)
    b = base & (dist > 30)
    if not a.any() or not b.any():
        return 0.0
    return float(y[a].mean() - y[b].mean())
