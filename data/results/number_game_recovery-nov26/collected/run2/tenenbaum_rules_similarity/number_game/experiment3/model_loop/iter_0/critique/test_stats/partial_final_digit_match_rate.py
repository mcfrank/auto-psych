# name: partial_final_digit_match_rate
# description: In sets whose examples do NOT all share one final digit, yes rate on non-example targets whose final digit matches at least one example's; observed above null means people use loose final-digit similarity that the model's all-or-none "ends in d" rules miss, below means the model over-credits it.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric).values
    t = df["target"].astype(int).values
    valid = ~np.isnan(ex)
    exi = np.nan_to_num(ex, nan=-1).astype(int)
    ones = np.where(valid, exi % 10, -1)
    first = ones[:, 0]
    all_same = ((ones == first[:, None]) | ~valid).all(axis=1)
    match = ((ones == (t % 10)[:, None]) & valid).any(axis=1)
    in_set = (exi == t[:, None]).any(axis=1)
    m = ~all_same & match & ~in_set
    if m.sum() == 0:
        return 0.0
    return float(df["rating"].values[m].mean())
