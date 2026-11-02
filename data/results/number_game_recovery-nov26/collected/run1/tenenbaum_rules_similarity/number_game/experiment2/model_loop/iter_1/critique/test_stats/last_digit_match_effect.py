# name: last_digit_match_effect
# description: Among non-member targets of sets with 2+ examples, yes rate when the target shares the final digit of all examples minus yes rate otherwise; observed above null means people generalise "ends in d" more than the model, below means less.
def test_statistic(df):
    s = df["number_set"].astype(str).str.split(",", expand=True).astype(float)
    t = df["target"].astype(float)
    k = s.notna().sum(axis=1).values
    member = s.eq(t, axis=0).any(axis=1).values
    match = ((s % 10).eq(t % 10, axis=0) | s.isna()).all(axis=1).values
    r = df["rating"].values.astype(float)
    base = (k >= 2) & ~member
    a, b = base & match, base & ~match
    if not a.any() or not b.any():
        return 0.0
    return float(r[a].mean() - r[b].mean())
