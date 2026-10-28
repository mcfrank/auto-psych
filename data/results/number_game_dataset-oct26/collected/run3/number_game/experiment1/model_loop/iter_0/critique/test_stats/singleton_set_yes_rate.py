# name: singleton_set_yes_rate
# description: Overall yes rate on trials whose set has a single example; observed above null_mean means people generalize more broadly from one example than the model's size-principle posterior, below means more narrowly.
def test_statistic(df):
    n = df["number_set"].astype(str).str.count(",") + 1
    m = n == 1
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
