# name: shared_contained_digit_nonfinal_yes
# description: Yes rate for targets containing (in any position) a digit present in every example of a >=2-example set, but sharing no final digit with any example and >5 from every example; observed above null means people generalise contained-digit rules more than the model predicts.
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
    def digs(a):
        a = np.nan_to_num(a, nan=-1).astype(int)
        out = np.zeros(a.shape + (10,), bool)
        for d in range(10):
            out[..., d] = (a >= 0) & ((a % 10 == d) | ((a >= 10) & ((a // 10) % 10 == d)) | ((a >= 100) & (a // 100 == d)))
        return out
    valid = ~np.isnan(E)
    common = np.all(digs(E) | ~valid[..., None], axis=1)
    has = np.any(common & digs(t), axis=1)
    m = (n >= 2) & has & ~shares_last & (dist > 5) & ~in_set
    return float(y[m].mean()) if m.any() else 0.0
