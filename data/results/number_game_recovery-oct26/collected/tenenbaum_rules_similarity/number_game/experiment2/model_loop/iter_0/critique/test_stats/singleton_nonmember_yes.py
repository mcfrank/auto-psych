# name: singleton_nonmember_yes
# description: Yes-rate to non-example targets after a single-example set; observed above null means people generalize more broadly from one example than the size-principle model (below null: more narrowly).
def test_statistic(df):
    s = df["number_set"].astype(str)
    single = ~s.str.contains(",")
    t = df["target"].astype(int).astype(str)
    m = (single & (s.str.strip() != t)).values
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
