# name: inside_vs_outside_range_yes_gap
# description: In multi-example sets spanning >=20, among targets sharing no final digit with any example and not an example, yes rate for targets strictly inside [min,max] of the examples minus targets outside it; t_observed > null_mean means the model under-produces interval (range-interpolation) generalization relative to extrapolation.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    t = df["target"].astype(int).values
    codes, uniq = pd.factorize(sets)
    X = np.full((len(uniq), 4), -1000)
    lo = np.zeros(len(uniq)); hi = np.zeros(len(uniq)); n = np.zeros(len(uniq))
    for i, s in enumerate(uniq):
        xs = [int(v) for v in s.split(",")][:4]
        X[i, :len(xs)] = xs
        lo[i] = min(xs); hi[i] = max(xs); n[i] = len(xs)
    E = X[codes]
    valid = E > -1000
    same_dig = (valid & ((E % 10) == (t % 10)[:, None])).any(1)
    L = lo[codes]; H = hi[codes]; N = n[codes]
    elig = (N >= 2) & ((H - L) >= 20) & ~same_dig
    inside = elig & (t > L) & (t < H)
    outside = elig & ((t < L) | (t > H))
    r = df["rating"].values.astype(float)
    if inside.sum() == 0 or outside.sum() == 0:
        return 0.0
    return float(r[inside].mean() - r[outside].mean())
