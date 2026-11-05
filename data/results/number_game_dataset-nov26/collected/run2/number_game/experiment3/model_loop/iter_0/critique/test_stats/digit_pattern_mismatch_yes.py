# name: digit_pattern_mismatch_yes
# description: Yes rate for targets whose final digit differs from a final digit shared by all (2+) examples; observed < null means people reject off-pattern targets more strongly than the model (model over-produces yes), observed > null the reverse.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    u, inv = np.unique(sets.values, return_inverse=True)
    E = np.full((len(u), 4), np.nan)
    for i, s in enumerate(u):
        v = [float(x) for x in s.split(",")]
        E[i, :len(v)] = v
    n = (~np.isnan(E)).sum(axis=1)
    ld = np.mod(E, 10)
    first = ld[:, 0]
    shared = (n >= 2) & np.all((ld == first[:, None]) | np.isnan(E), axis=1)
    t = df["target"].astype(int).values
    m = shared[inv] & (np.mod(t, 10) != first[inv])
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values.astype(float)[m].mean())
