# name: recency_last_vs_first_example_proximity
# description: For sets of >=2 examples, yes-rate on non-example targets within 10 of the LAST-shown example (and not within 10 of the first) minus on targets within 10 of the FIRST-shown example (and not within 10 of the last); observed != null (model is order-invariant, so null ~ stimulus-driven difference) means example order matters: > null = recency, < null = primacy.
def test_statistic(df):
    s = df["number_set"].astype(str)
    codes, uniq = pd.factorize(s)
    arrs = [np.array([int(x) for x in u.split(",")]) for u in uniq]
    first = np.array([a[0] for a in arrs])[codes]
    last = np.array([a[-1] for a in arrs])[codes]
    n = np.array([len(a) for a in arrs])[codes]
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    ins = np.array([str(tt) in u.split(",") for tt, u in zip(t, s)])
    nl = np.abs(t - last) <= 10
    nf = np.abs(t - first) <= 10
    base = (n >= 2) & ~ins
    a = base & nl & ~nf
    b = base & nf & ~nl
    ra = y[a].mean() if a.any() else 0.0
    rb = y[b].mean() if b.any() else 0.0
    return float(ra - rb)
