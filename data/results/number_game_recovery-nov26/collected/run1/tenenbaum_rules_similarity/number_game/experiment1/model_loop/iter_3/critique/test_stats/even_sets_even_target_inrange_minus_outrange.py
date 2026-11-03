# name: even_sets_even_target_inrange_minus_outrange
# description: For sets of >=2 examples that are all even, yes-rate on even non-example targets inside [min,max] of the examples minus on even targets outside it; observed > null means people combine the parity rule with a range restriction ("even numbers around these") more than the model, whose even/multiples rules ignore magnitude.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    arrs = [np.array([int(x) for x in s.split(",")]) for s in uniq]
    ok = np.array([len(a) >= 2 and bool(np.all(a % 2 == 0)) for a in arrs])
    lo = np.array([a.min() for a in arrs])[codes]
    hi = np.array([a.max() for a in arrs])[codes]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ins = np.array([str(tt) in s.split(",") for tt, s in zip(t, df["number_set"].astype(str))])
    m = ok[codes] & (t % 2 == 0) & ~ins
    a = m & (t >= lo) & (t <= hi)
    b = m & ((t < lo) | (t > hi))
    ra = y[a].mean() if a.any() else 0.0
    rb = y[b].mean() if b.any() else 0.0
    return float(ra - rb)
