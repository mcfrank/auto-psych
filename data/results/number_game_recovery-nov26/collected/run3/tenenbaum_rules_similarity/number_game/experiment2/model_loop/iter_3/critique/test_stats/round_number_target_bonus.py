# name: round_number_target_bonus
# description: Among sets with no example divisible by 5, yes rate on even targets divisible by 10 minus yes rate on other even targets (pooled over trials); observed > null means people give a round-number salience bonus the model lacks, observed < null means the model over-generalises to round numbers.
def test_statistic(df):
    sets = df["number_set"].astype(str)
    uniq = pd.Series(sets.unique())
    ok = uniq[uniq.apply(lambda s: all(int(x) % 5 for x in s.split(",")))]
    m = sets.isin(set(ok)).values
    t = df["target"].astype(int).values
    r = df["rating"].astype(float).values
    ev = m & (t % 2 == 0)
    a = r[ev & (t % 10 == 0)]
    b = r[ev & (t % 10 != 0)]
    if len(a) == 0 or len(b) == 0:
        return 0.0
    return float(a.mean() - b.mean())
