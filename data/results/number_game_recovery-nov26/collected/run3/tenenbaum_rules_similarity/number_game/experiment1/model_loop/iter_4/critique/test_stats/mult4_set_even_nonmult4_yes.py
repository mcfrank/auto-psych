# name: mult4_set_even_nonmult4_yes
# description: Among sets (>=2 examples) whose examples are all multiples of 4, the "yes" rate for even targets not divisible by 4 (and not in the set); observed below null = model over-predicts acceptance of merely-even numbers (even/even-in-range concepts) when a multiples-of-4/8 rule fits.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    n = ex.notna().sum(axis=1)
    all4 = ((ex % 4 == 0) | ex.isna()).all(axis=1) & (n >= 2)
    t = df["target"].astype(int)
    inset = ex.eq(t, axis=0).any(axis=1)
    m = all4 & (t % 2 == 0) & (t % 4 != 0) & ~inset
    if m.sum() == 0:
        return 0.0
    return float(df.loc[m, "rating"].mean())
