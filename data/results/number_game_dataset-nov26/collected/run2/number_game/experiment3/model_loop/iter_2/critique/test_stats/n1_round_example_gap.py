# name: n1_round_example_gap
# description: For one-example sets, overall yes rate when the lone example is a multiple of 5 minus when it is not; observed < null means people generalize less broadly from a round lone example than the model predicts (round examples read as a narrow multiples-of-5/10 rule), observed > null the reverse.
def test_statistic(df):
    s = df["number_set"].astype(str)
    one = (s.str.count(",") == 0).values
    e = pd.to_numeric(s.where(one, "0"), errors="coerce").fillna(0).astype(int).values
    r = df["rating"].astype(float).values
    rnd = one & (e % 5 == 0)
    non = one & (e % 5 != 0)
    if rnd.sum() == 0 or non.sum() == 0:
        return 0.0
    return float(r[rnd].mean() - r[non].mean())
