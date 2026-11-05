# name: set_size_four_minus_two
# description: Overall yes rate on 4-example sets minus 2-example sets; observed above null means people tighten generalization with more examples less than the (tempered) size principle predicts, below null means more.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    n = ex.notna().sum(axis=1)
    return m(n == 4) - m(n == 2)
