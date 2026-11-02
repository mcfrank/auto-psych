# name: shared_tens_digit_outside_yes
# description: Among non-example targets outside the example range and not sharing a last digit with any example, yes-rate when the target shares its tens digit (decade) with some example minus when it does not; observed above the null means people use decade similarity more than the model's digit rules/intervals produce, below means less.
def test_statistic(df):
    ex = df["number_set"].str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    tens_ex = (ex // 10)
    last_ex = ex % 10
    shares_tens = tens_ex.eq(t // 10, axis=0).any(axis=1)
    shares_last = last_ex.eq(t % 10, axis=0).any(axis=1)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    outside = (t < lo) | (t > hi)
    base = outside & ~shares_last
    r = df["rating"].astype(float)
    a = base & shares_tens
    b = base & ~shares_tens
    if a.sum() == 0 or b.sum() == 0:
        return 0.0
    return float(r[a].mean() - r[b].mean())
