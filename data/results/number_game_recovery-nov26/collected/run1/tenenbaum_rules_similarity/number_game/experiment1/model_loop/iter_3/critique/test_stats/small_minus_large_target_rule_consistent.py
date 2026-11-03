# name: small_minus_large_target_rule_consistent
# description: For sets of >=2 examples that are all even, yes-rate on even non-example targets <=50 minus on even targets >50; observed > null means people favour small numbers within an endorsed rule (a magnitude bias) beyond what the model's examples-driven intervals produce, observed < null the reverse.
def test_statistic(df):
    s = df["number_set"].astype(str)
    codes, uniq = pd.factorize(s)
    arrs = [np.array([int(x) for x in u.split(",")]) for u in uniq]
    ok = np.array([len(a) >= 2 and bool(np.all(a % 2 == 0)) for a in arrs])[codes]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ins = np.array([str(tt) in u.split(",") for tt, u in zip(t, s)])
    m = ok & ~ins & (t % 2 == 0)
    a = m & (t <= 50)
    b = m & (t > 50)
    ra = y[a].mean() if a.any() else 0.0
    rb = y[b].mean() if b.any() else 0.0
    return float(ra - rb)
