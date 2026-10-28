# name: shared_last_digit_bonus
# description: Among sets of >=2 examples that all share a last digit, yes-rate for (non-member) targets with that last digit minus yes-rate for targets without it; observed above null means the model under-weights the "ends in d" rule relative to people.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric)
    n = s.notna().sum(axis=1)
    t = df["target"].astype(int)
    ld = s.mod(10)
    first = ld[0]
    same = ld.eq(first, axis=0) | ld.isna()
    uni = same.all(axis=1) & (n >= 2) & ~s.eq(t, axis=0).any(axis=1)
    match = uni & (t % 10 == first)
    miss = uni & (t % 10 != first)
    r = df["rating"].astype(float)
    if not match.any() or not miss.any():
        return 0.0
    return float(r[match].mean() - r[miss].mean())
