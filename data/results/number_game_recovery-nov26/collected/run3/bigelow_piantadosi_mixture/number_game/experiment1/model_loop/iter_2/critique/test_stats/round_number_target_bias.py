# name: round_number_target_bias
# description: For sets containing no multiple of 10, yes-rate for non-example targets that are multiples of 10 minus yes-rate for other non-example targets; observed above the null means people over-endorse round numbers relative to the model, below means they under-endorse them.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    has10 = (ex % 10 == 0).any(axis=1)
    is_ex = ex.eq(t, axis=0).any(axis=1)
    base = (~has10) & (~is_ex)
    r = df["rating"].astype(float)
    a = base & (t % 10 == 0)
    b = base & (t % 10 != 0)
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
