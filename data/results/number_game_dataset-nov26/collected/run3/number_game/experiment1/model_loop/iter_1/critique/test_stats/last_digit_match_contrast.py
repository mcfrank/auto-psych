# name: last_digit_match_contrast
# description: Among non-example targets, yes rate when the target's last digit matches the last digit of some example minus when it matches none; observed below null means the model over-weights ends-in-digit rules relative to people.
def test_statistic(df):
    ex = df["number_set"].astype(str).str.split(",", expand=True).apply(pd.to_numeric, errors="coerce")
    t = df["target"].astype(int)
    r = df["rating"].astype(float)
    inset = ex.eq(t, axis=0).any(axis=1)
    def m(mask):
        return float(r[mask].mean()) if mask.sum() > 0 else 0.0
    match = (ex % 10).eq(t % 10, axis=0).any(axis=1)
    return m((~inset) & match) - m((~inset) & (~match))
