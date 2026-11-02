# name: odd_set_far_beyond_range_yes
# description: In sets (>=3 examples) of all-odd numbers, the "yes" rate for odd targets more than 15 beyond the examples' range minus that for odd targets inside the range; observed above null (less negative) = people extend "odd" beyond the range more than the model's odd-within-range concepts allow.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    allodd = ((ex % 2 == 1) | ex.isna()).all(axis=1) & (n >= 3)
    t = df["target"].astype(int)
    lo, hi = ex.min(axis=1), ex.max(axis=1)
    inset = ex.eq(t, axis=0).any(axis=1)
    oddt = allodd & (t % 2 == 1) & ~inset
    far = oddt & ((t > hi + 15) | (t < lo - 15))
    inside = oddt & (t > lo) & (t < hi)
    if far.sum() == 0 or inside.sum() == 0:
        return 0.0
    return float(df.loc[far, "rating"].mean() - df.loc[inside, "rating"].mean())
