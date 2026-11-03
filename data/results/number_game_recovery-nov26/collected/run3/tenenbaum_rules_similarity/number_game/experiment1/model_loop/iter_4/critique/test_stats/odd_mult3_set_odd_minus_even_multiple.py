# name: odd_mult3_set_odd_minus_even_multiple
# description: In sets whose examples are all odd multiples of 3, the "yes" rate for odd multiple-of-3 targets minus that for even multiple-of-3 targets; observed above null = people use a conjunctive "odd multiples of 3" concept the model lacks (model under-produces the contrast), below = model over-weights parity.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    ok = (((ex % 3 == 0) & (ex % 2 == 1)) | ex.isna()).all(axis=1) & (n >= 2)
    t = df["target"].astype(int)
    m3 = ok & (t % 3 == 0)
    odd = m3 & (t % 2 == 1)
    even = m3 & (t % 2 == 0)
    if odd.sum() == 0 or even.sum() == 0:
        return 0.0
    return float(df.loc[odd, "rating"].mean() - df.loc[even, "rating"].mean())
