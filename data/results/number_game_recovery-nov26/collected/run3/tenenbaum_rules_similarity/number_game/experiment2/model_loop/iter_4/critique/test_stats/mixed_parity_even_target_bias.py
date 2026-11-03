# name: mixed_parity_even_target_bias
# description: For sets whose examples include both odd and even numbers, yes rate on even non-example targets minus odd non-example targets; observed > null means people favour even numbers (or reject odd ones) more than the model's concepts imply when parity is uninformative, observed < null means they favour odd numbers more.
def test_statistic(df):
    sets = df["number_set"].astype(str).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    mp = {}
    for s in pd.unique(sets):
        e = [int(x) for x in s.split(",")]
        mp[s] = len(set(x % 2 for x in e)) == 2
    m = pd.Series(sets).map(mp).values.astype(bool)
    isex = np.array([str(x) in s.split(",") for x, s in zip(t, sets)])
    a = r[m & ~isex & (t % 2 == 0)]
    b = r[m & ~isex & (t % 2 == 1)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
