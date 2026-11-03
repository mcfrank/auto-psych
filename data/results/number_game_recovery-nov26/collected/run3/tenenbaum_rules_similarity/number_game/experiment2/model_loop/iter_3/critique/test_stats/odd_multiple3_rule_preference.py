# name: odd_multiple3_rule_preference
# description: For sets of >=2 examples that are all odd multiples of 3 (e.g. 63,27,99,33), yes rate on even multiples of 3 minus yes rate on odd non-multiples of 3 (non-example targets); observed < null means people choose "odd" over the smaller "multiples of 3" rule more than the size principle predicts, observed > null means they prefer the multiplicative rule more.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ok = set()
    for s in pd.unique(sets):
        e = [int(x) for x in s.split(",")]
        if len(e) >= 2 and all(x % 2 == 1 and x % 3 == 0 for x in e):
            ok.add(s)
    m = np.isin(sets, list(ok))
    a = r[m & (t % 2 == 0) & (t % 3 == 0)]
    b = r[m & (t % 2 == 1) & (t % 3 != 0)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
