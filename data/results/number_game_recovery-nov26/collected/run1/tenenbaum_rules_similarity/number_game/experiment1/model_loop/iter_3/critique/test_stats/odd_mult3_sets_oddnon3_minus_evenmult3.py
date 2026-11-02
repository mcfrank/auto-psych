# name: odd_mult3_sets_oddnon3_minus_evenmult3
# description: For sets of >=3 examples that are all odd multiples of 3, yes-rate on odd non-multiples of 3 minus yes-rate on even multiples of 3; observed > null means people generalize such sets by oddness more (and by "multiples of 3" less) than the model, which lacks the conjunctive "odd multiples of 3" concept and over-weights multiples of 3.
def test_statistic(df):
    codes, uniq = pd.factorize(df["number_set"].astype(str))
    ok = np.array([(lambda a: len(a) >= 3 and bool(np.all(a % 2 == 1)) and bool(np.all(a % 3 == 0)))(np.array([int(x) for x in s.split(",")])) for s in uniq])
    t = df["target"].astype(int).to_numpy()
    y = df["rating"].astype(float).to_numpy()
    m = ok[codes]
    a = m & (t % 2 == 1) & (t % 3 != 0)
    b = m & (t % 2 == 0) & (t % 3 == 0)
    ra = y[a].mean() if a.any() else 0.0
    rb = y[b].mean() if b.any() else 0.0
    return float(ra - rb)
