# name: single_example_divisor_multiple_bonus
# description: For single-example sets with example >= 2, yes rate on non-example targets that divide or are a multiple of the example minus yes rate on other targets; observed above null means people use multiplicative (divisor/multiple) relations from one example more than the model (which lacks "divisors of n" concepts), below null means less.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = ~s.str.contains(",")
    sub = df[one]
    e = sub["number_set"].astype(int).to_numpy()
    t = sub["target"].astype(int).to_numpy()
    r = sub["rating"].to_numpy()
    keep = (e >= 2) & (t != e)
    rel = ((t % e == 0) | (e % t == 0)) & keep
    oth = (~((t % e == 0) | (e % t == 0))) & keep
    if not rel.any() or not oth.any():
        return 0.0
    return float(r[rel].mean() - r[oth].mean())
